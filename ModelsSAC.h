#pragma once
#include <torch/torch.h>
#include <math.h>

struct Actor: public torch::nn::Module
{
	/* data */
	torch::nn::Linear a_lin1_{nullptr}, a_lin2_{nullptr}, a_lin3_{nullptr};
	torch::Tensor mu_;
	torch::Tensor log_std_;

	Actor(int64_t n_in, int64_t n_out, double std)
		:
          a_lin1_(torch::nn::Linear(n_in, 16)),
          a_lin2_(torch::nn::Linear(16, 32)),
          a_lin3_(torch::nn::Linear(32, n_out)),
          mu_(torch::full(n_out, 0.)),
          log_std_(torch::full(n_out, std))
	{
        // Register the modules.
        register_module("a_lin1", a_lin1_);
        register_module("a_lin2", a_lin2_);
        register_module("a_lin3", a_lin3_);
        register_parameter("log_std", log_std_);		
	}

	auto forward(torch::Tensor x) -> torch::Tensor
	{
		//Actor forward
		mu_ = torch::relu(a_lin1_->forward(x));
		mu_ = torch::relu(a_lin2_->forward(mu_));
		mu_ = torch::tanh(a_lin3_->forward(mu_));

        if (this->is_training()) 
        {
            torch::NoGradGuard no_grad;

            torch::Tensor action = at::normal(mu_, log_std_.exp().expand_as(mu_));
            return action;  
        }

	}

    auto log_prob(torch::Tensor action) -> torch::Tensor
    {
        // Logarithmic probability of taken action, given the current distribution.
        torch::Tensor var = (log_std_+log_std_).exp();

        return -((action - mu_)*(action - mu_))/(2*var) - log_std_ - log(sqrt(2*M_PI));
    }

    auto sample(torch::Tensor state) ->std::tuple<torch::Tensor, torch::Tensor>
    {
         // Get the action and its log probability
        auto mu = forward(state);
        auto action = mu + at::normal(0, 1, mu.sizes()) * log_std_.exp();
        auto log_prob_ = log_prob(mu);
        return std::make_tuple(action, log_prob_);       
    }

};
// TORCH_MODULE(Actor);



struct Critic: public torch::nn::Module
{
    torch::nn::Linear c_lin1_{nullptr}, c_lin2_{nullptr}, c_lin3_{nullptr};

    Critic(int64_t n_in, int64_t n_out, double std)
    :
          c_lin1_(torch::nn::Linear(n_in+n_out, 16)),
          c_lin2_(torch::nn::Linear(16, 32)),
          c_lin3_(torch::nn::Linear(32, n_out))
    {
        // Register the modules.
        register_module("c_lin1", c_lin1_);
        register_module("c_lin2", c_lin2_);
        register_module("c_lin3", c_lin3_);
    }

    auto forward(torch::Tensor state, torch::Tensor action) -> torch::Tensor
    {
        auto x = torch::relu(c_lin1_->forward(torch::cat({state, action}, -1)));
        x = torch::relu(c_lin2_->forward(x));
        return c_lin3_->forward(x);
    }

};

// TORCH_MODULE(Critic);


