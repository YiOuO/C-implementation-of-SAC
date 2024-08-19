#pragma once

#include <torch/torch.h>
#include <random>
#include "ModelsSAC.h"
#include "Replay_buffer.h"

// Vector of tensors.
using VT = std::vector<torch::Tensor>;

// Optimizer.
using OPT = torch::optim::Optimizer;

// Random engine for shuffling memory.
std::random_device rd;
std::mt19937 re(rd());


class SAC
{
	public:
		SAC(int64_t state_dim, int64_t action_dim, double gamma = 0.99, double tau = 0.005, double alpha = 0.2, double lr = 3e-4, double log_std = 0.02,int buffer_capacity = 100000)
			: gamma(gamma), tau(tau), alpha(alpha), log_std(log_std),buffer_capacity(buffer_capacity) {
			
			actor = std::make_shared<Actor>(state_dim, action_dim, log_std);
			critic_1 = std::make_shared<Critic>(state_dim, action_dim, log_std);
			critic_2 = std::make_shared<Critic>(state_dim, action_dim, log_std);
			target_critic_1 = std::make_shared<Critic>(state_dim, action_dim, log_std);
			target_critic_2 = std::make_shared<Critic>(state_dim, action_dim, log_std);

			// 复制权重到目标网络
    		// 手动复制权重到目标网络
			for (size_t i = 0; i < target_critic_1->parameters().size(); ++i) {
				target_critic_1->parameters()[i].copy_(critic_1->parameters()[i]);
			}
			for (size_t i = 0; i < target_critic_2->parameters().size(); ++i) {
				target_critic_2->parameters()[i].copy_(critic_2->parameters()[i]);
			}

			actor_optimizer = std::make_shared<torch::optim::Adam>(actor->parameters(), torch::optim::AdamOptions(lr)); 
			critic_1_optimizer = std::make_shared<torch::optim::Adam>(critic_1->parameters(), lr);
			critic_2_optimizer = std::make_shared<torch::optim::Adam>(critic_2->parameters(), lr);

		};

	public:
		std::shared_ptr<Actor> actor;
		std::shared_ptr<Critic> critic_1, critic_2;
		std::shared_ptr<Critic> target_critic_1, target_critic_2;
		std::shared_ptr<torch::optim::Adam> actor_optimizer;
		std::shared_ptr<torch::optim::Adam> critic_1_optimizer, critic_2_optimizer;

		double gamma, tau, alpha, log_std;		
		int buffer_capacity;
		struct Transition 
		{
			torch::Tensor state;
			torch::Tensor action;
			torch::Tensor reward;
			torch::Tensor next_state;
			torch::Tensor done;
		};

	public:
		//update the network
		void train(int batchsize);
		void update_actor(torch::Tensor states);
		void update_critics(torch::Tensor states, torch::Tensor actions, torch::Tensor rewards, torch::Tensor dones, torch::Tensor next_states);
		void update_target_networks();

		//replay buffer
		std::deque<Transition> replay_buffer;
		std::mt19937 rng;

		std::vector<torch::Tensor> sample_batch(int batch_size);
		void store_transition(const torch::Tensor& state, const torch::Tensor& action, 
							  const torch::Tensor& reward, const torch::Tensor& next_state, 
							  const torch::Tensor& done);
		bool has_enough_samples(int batch_size);

};

bool SAC::has_enough_samples(int batch_size){
	return replay_buffer.size() >= batch_size;
}

std::vector<torch::Tensor> SAC::sample_batch(int batch_size)
{
	std::uniform_int_distribution<> dist(0, replay_buffer.size() - 1);
	std::vector<torch::Tensor> states, actions, rewards, next_states, dones;

	for (int i = 0; i < batch_size; ++i) {
		int index = dist(rng);
		states.push_back(replay_buffer[index].state);
		actions.push_back(replay_buffer[index].action);
		rewards.push_back(replay_buffer[index].reward);
		next_states.push_back(replay_buffer[index].next_state);
		dones.push_back(replay_buffer[index].done);
	}

	return {torch::stack(states), torch::stack(actions), torch::stack(rewards), 
			torch::stack(next_states), torch::stack(dones)};
}

void SAC::store_transition(const torch::Tensor& state, const torch::Tensor& action, 
						   const torch::Tensor& reward, const torch::Tensor& next_state, 
						   const torch::Tensor& done) 
{
	if (replay_buffer.size() == buffer_capacity) {
		replay_buffer.pop_front();
	}
	replay_buffer.push_back({state, action, reward, next_state, done});
}

void SAC::update_actor(torch::Tensor states)
{
	auto [actions, log_probs] = actor->sample(states);

	
	
	auto q_value = torch::min(critic_1->forward(states, actions),
								critic_2->forward(states, actions));
	auto actor_loss = (alpha * log_probs - q_value).mean();

	actor_optimizer->zero_grad();
	actor_loss.backward();
	actor_optimizer->step();
}

void SAC::update_critics(torch::Tensor states, torch::Tensor actions, torch::Tensor rewards, torch::Tensor dones, torch::Tensor next_states)
{
        torch::NoGradGuard no_grad;
        auto [next_actions, next_log_probs] = actor->sample(next_states);
        auto target_q_value = torch::min(target_critic_1->forward(next_states, next_actions),
                                         target_critic_2->forward(next_states, next_actions)) - alpha * next_log_probs;
        target_q_value = rewards + (1 - dones) * gamma * target_q_value;

        auto current_q_value_1 = critic_1->forward(states, actions);
        auto current_q_value_2 = critic_2->forward(states, actions);

        auto critic_1_loss = torch::mse_loss(current_q_value_1, target_q_value);
        auto critic_2_loss = torch::mse_loss(current_q_value_2, target_q_value);

        critic_1_optimizer->zero_grad();
        critic_1_loss.backward();
        critic_1_optimizer->step();

        critic_2_optimizer->zero_grad();
        critic_2_loss.backward();
        critic_2_optimizer->step();
}

void SAC::update_target_networks() {
    // 更新 target_critic_1
    auto params1 = critic_1->parameters();
    auto target_params1 = target_critic_1->parameters();
    for (size_t i = 0; i < params1.size(); ++i) {
        target_params1[i].copy_(tau * params1[i] + (1 - tau) * target_params1[i]);
    }

    // 更新 target_critic_2
    auto params2 = critic_2->parameters();
    auto target_params2 = target_critic_2->parameters();
    for (size_t i = 0; i < params2.size(); ++i) {
        target_params2[i].copy_(tau * params2[i] + (1 - tau) * target_params2[i]);
    }
}

void SAC::train(int batch_size) {
	auto batch = sample_batch(batch_size);
	auto states = batch[0];
	auto actions = batch[1];
	auto rewards = batch[2];
	auto next_states = batch[3];
	auto dones = batch[4];

	update_critics(states, actions, rewards, next_states, dones);
	update_actor(states);
	update_target_networks();
}