#include <fstream>
#include <Eigen/Core>
#include <torch/torch.h>
#include <random>
#include "TestEnvironment.h"
#include "ModelsSAC.h"
#include "SAC.h"

int main(){
    // Random engine.
    std::random_device rd;
    std::mt19937 re(rd());
    std::uniform_int_distribution<> dist(-5, 5);	

    // Environment.
    double x = double(dist(re)); // goal x pos
    double y = double(dist(re)); // goal y pos
    TestEnvironment env(x, y);

    // Model parameters
    int state_dim = 4;
    int action_dim = 2;
    double std = 2e-2;
    int batch_size = 32;
    int max_episodes = 1000;
    int max_timesteps = 10000;
    double gamma = 0.99;
    double tau = 0.005;
    double alpha = 0.2;
    double lr = 3e-4;
    double log_std = 0.02;
    int buffer_capacity = 100000;
    // SAC
    SAC agent(state_dim, action_dim, gamma, tau, alpha, lr, log_std, buffer_capacity);
    // Output.
    std::ofstream out;
    out.open("../data/data.csv");
    // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
    out << 1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";   
	
    // Average reward.
    double best_episode_reward = 0.;
    double episode_reward = 0.;


    // 训练循环
    for (int episode = 0; episode < max_episodes; episode++) {
        env.Reset();
        auto state = env.State();
        std::cout << "Episode " << state << std::endl;
        

        for (int t = 0; t < max_timesteps; t++) {
            // 使用策略网络采样动作
            auto [action, log_prob] = agent.actor->sample(state);
            // // 环境反馈
            double x_act = action[0][0].item<double>();
            double y_act = action[0][1].item<double>();
            auto sd = env.Act(x_act,y_act);
            // New state.
            auto reward = env.Reward(std::get<1>(sd));
            auto done = std::get<2>(sd);
            auto next_state = env.State();
            // // 存储转移到经验池
            agent.store_transition(state,action,reward,next_state,done);
            // SAC agent训练
            if (agent.has_enough_samples(batch_size)) {
                std::cout << "Training..." << std::endl;
                agent.train(batch_size);
            }

            state = next_state;
            episode_reward += reward.item<double>();

            if (done.item<bool>()) 
            {
                std::cout << "finished training" <<" "<<t<< std::endl;
                break;
            }
        }

        // Save the best net.
        if (episode_reward > best_episode_reward) {

            best_episode_reward = episode_reward;
            printf("Best average reward: %f\n", best_episode_reward);
            // torch::save(ac, "best_model.pt");
        }

        episode_reward = 0.;

        // Reset at the end of an epoch.
        double x_new = double(dist(re)); 
        double y_new = double(dist(re));
        env.SetGoal(x_new, y_new);        
        // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
        out << episode << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";
        std::cout << "Episode " << episode << " reward: " << episode_reward << std::endl;
    }	 

	return 0;

}