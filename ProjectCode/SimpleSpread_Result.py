import os
import sys
sys.path.append(r'/home/kimds929/CodePractice')

import ray
ray.init()

from six.moves import cPickle
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO

import supersuit as ss
from mpe2 import simple_spread_v3
from utils_SimpleSpread_Decentralize import CustomObservationWrapper


#############################################################################
def evaluate_single_episode(model_path, max_cycles=100):
    import sys
    sys.path.append('/home/kimds929/CodePractice')

    from stable_baselines3 import PPO
    from utils_SimpleSpread_Decentralize import CustomObservationWrapper
    from mpe2 import simple_spread_v3
    import supersuit as ss
    import numpy as np
    
    raw_env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=max_cycles, continuous_actions=True)
    wrapper_env = CustomObservationWrapper(raw_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(wrapper_env)
    env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=1, base_class='stable_baselines3')
    
    # load_model
    model = PPO.load(model_path, env=env)
    
    env_agents = wrapper_env.possible_agents
    obs, _ = wrapper_env.reset()
    total_reward = 0.0
    
    states, infos = wrapper_env.reset()
    for step_idx in range(max_cycles):
        actions_dict = {}

        # 각 agent별로 개별적으로 action 예측
        for agent in env_agents:
            obs = states[agent]
            action, _ = model.predict(obs, deterministic=True)  # deterministic=True for evaluation
            actions_dict[agent] = action

        # step 진행
        states, rewards, terminations, truncations, infos = wrapper_env.step(actions_dict)
        step_reward = np.sum(list(rewards.values())).item()
        
        total_reward += step_reward
    return total_reward

# ---------------------------------------------------------------------------------------------
# model.predict 기반 평가 함수
def evaluate_model(model_path, n_eval_episodes=10, max_cycles=100):
    episode_rewards = []
    for episode in range(n_eval_episodes):
        total_reward = evaluate_single_episode(model_path=model_path, max_cycles=max_cycles)
        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def evaluate_result(name='',
                    checkpoint_indices=list(range(0, 501, 10)),
                    total_timesteps_per_iter = 2000,
                    model_path_base = '/home/kimds929/CodePractice/SimpleSpread_sb3_mlp',
                    checkpoint_file_basename = 'ppo_decentalize_gnn_checkpoint',
                    eval_episodes=10,
                    max_cycles=100,
                    ray = False):
    raw_env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=max_cycles, continuous_actions=True)
    wrapper_env = CustomObservationWrapper(raw_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(wrapper_env)
    env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=1, base_class='stable_baselines3')
    
    # checkpoint_indices = checkpoint_indices
    # total_timesteps_per_iter = total_timesteps_per_iter
    # model_path_base = '/home/kimds929/CodePractice/SimpleSpread_sb3_mlp'

    env_steps = []
    mean_rewards = []
    std_rewards = []

    for idx in checkpoint_indices:
        model_path = f"{model_path_base}/{checkpoint_file_basename}{idx}.zip"
        print(model_path)
        model = PPO.load(model_path, env=env)

        steps = idx * total_timesteps_per_iter
        env_steps.append(steps)
        
        if ray is True:
            mean_reward, std_reward = ray_evaluate_model(model_path, n_eval_episodes=eval_episodes, max_cycles=max_cycles)
        else:
            mean_reward, std_reward = evaluate_model(model_path, n_eval_episodes=eval_episodes, max_cycles=max_cycles)
            
        
        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)
        print(f"[{name}] Checkpoint {idx}: Steps={steps}, Mean Reward={mean_reward:.2f}, Std Reward={std_reward:.2f}")
    print()
    return env_steps, mean_rewards, std_rewards

##########################################################################################################################################################
##########################################################################################################################################################
# Ray Version ############################################################################################################################################



@ray.remote
def ray_evaluate_single_episode(model_path, max_cycles=100):
    return evaluate_single_episode(model_path=model_path, 
                                   max_cycles=max_cycles)

# ---------------------------------------------------------------------------------------------
def ray_evaluate_model(model_path, n_eval_episodes=10, max_cycles=100):
    ray_data = [ray_evaluate_single_episode.remote(model_path=model_path, 
                                   max_cycles=max_cycles) for i in range(n_eval_episodes)]
    episode_rewards = ray.get(ray_data)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward


@ray.remote
def ray_evaluate_result(name='',
                    checkpoint_indices=list(range(0, 501, 10)),
                    total_timesteps_per_iter = 2000,
                    model_path_base = '/home/kimds929/CodePractice/SimpleSpread_sb3_mlp',
                    checkpoint_file_basename = 'ppo_decentalize_gnn_checkpoint',
                    eval_episodes=10,
                    max_cycles=100):
    return evaluate_model(name=name,
                    checkpoint_indices=checkpoint_indices,
                    total_timesteps_per_iter = total_timesteps_per_iter,
                    model_path_base = model_path_base,
                    checkpoint_file_basename = checkpoint_file_basename,
                    eval_episodes=eval_episodes,
                    max_cycles=max_cycles)



##########################################################################################################################################################
##########################################################################################################################################################







###########################################################################################
# 통합 그래프 그리기
def plot_combined_reward_curve(results_dict, save_path=None):
    color_map = {
        "MLP": "firebrick",
        "GNN": "green",
        "EGNN": "royalblue",
        "E2GN2": "purple",
    }
    title_name = f"(Decentralize) {' vs '.join(list(results_dict.keys()))}: Reward vs Environment Steps"
    fig = plt.figure(figsize=(12, 7))
    plt.title(title_name)
    
    for alg_name, results in results_dict.items():
        steps, reward_means, reward_stds = results
        plt.plot(steps, reward_means, label=alg_name, color=color_map[alg_name])
        plt.fill_between(
            steps,
            np.array(reward_means) - np.array(reward_stds),
            np.array(reward_means) + np.array(reward_stds),
            color=color_map[alg_name],
            alpha=0.2
            )
    # plt.legend(bbox_to_anchor=(1,1))
    plt.legend()
    plt.xlabel("Environment Steps")
    plt.ylabel("Mean Reward (100 episodes)")
    plt.grid(True)
    
    plt.show()
    
    if save_path is not None:
        fig.savefig(f"{save_path}/{title_name}.png")
    
    return fig

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

save_path = '/home/kimds929/CodePractice/SimpleSpread_Results'




max_cycles = 100
eval_episodes = 50
checkpoint_indices = list(range(0, 501, 10))
# checkpoint_indices = list(range(0, 51, 10))
total_timesteps_per_iter = 2000

# raw_env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=max_cycles, continuous_actions=True)
# wrapper_env = CustomObservationWrapper(raw_env)
# vec_env = ss.pettingzoo_env_to_vec_env_v1(wrapper_env)
# env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=1, base_class='stable_baselines3')
# ray_evaluate_model(model_path='/home/kimds929/CodePractice/SimpleSpread_sb3_mlp/ppo_decentalize_gnn_checkpoint0.zip')








######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

# env_steps, mean_rewards, std_rewards
# --------------------------------------------------------------------------------
results_dict = {}
# mlp
results_dict['MLP'] = evaluate_result(name='MLP',
                            checkpoint_indices=checkpoint_indices,
                            total_timesteps_per_iter = total_timesteps_per_iter,
                            model_path_base = '/home/kimds929/CodePractice/SimpleSpread_sb3_mlp',
                            checkpoint_file_basename = 'ppo_decentalize_gnn_checkpoint',
                            eval_episodes=eval_episodes,
                            max_cycles=max_cycles,
                            ray=True)

# gnn
results_dict['GNN'] = evaluate_result(name='GNN',
                            checkpoint_indices=checkpoint_indices,
                            total_timesteps_per_iter = total_timesteps_per_iter,
                            model_path_base = '/home/kimds929/CodePractice/SimpleSpread_sb3_gnn',
                            checkpoint_file_basename = 'ppo_decentalize_gnn_checkpoint',
                            eval_episodes=eval_episodes,
                            max_cycles=max_cycles,
                            ray=True)

# egnn
results_dict['EGNN'] = evaluate_result(name='EGNN',
                            checkpoint_indices=checkpoint_indices,
                            total_timesteps_per_iter = total_timesteps_per_iter,
                            model_path_base = '/home/kimds929/CodePractice/SimpleSpread_sb3_egnn',
                            checkpoint_file_basename = 'ppo_decentalize_egnn_checkpoint',
                            eval_episodes=eval_episodes,
                            max_cycles=max_cycles,
                            ray=True)

# e2gn2
results_dict['E2GN2'] = evaluate_result(name='E2GN2',
                            checkpoint_indices=checkpoint_indices,
                            total_timesteps_per_iter = total_timesteps_per_iter,
                            model_path_base = '/home/kimds929/CodePractice/SimpleSpread_sb3_e2gn2',
                            checkpoint_file_basename = 'ppo_decentalize_e2gn2_checkpoint',
                            eval_episodes=eval_episodes,
                            max_cycles=max_cycles,
                            ray=True)

cPickle.dump(results_dict, open(f"{save_path}/results_dict.pkl", 'wb'))
print('save result.')


# ######################################################################################################################################################

# # env_steps, mean_rewards, std_rewards
# # --------------------------------------------------------------------------------
# # mlp
# future_mlp = ray_evaluate_result.remote(checkpoint_indices=checkpoint_indices,
#                             total_timesteps_per_iter = total_timesteps_per_iter,
#                             model_path_base = '/home/kimds929/CodePractice/SimpleSpread_sb3_mlp',
#                             checkpoint_file_basename = 'ppo_decentalize_gnn_checkpoint',
#                             eval_episodes=eval_episodes,
#                             max_cycles=max_cycles)

# # gnn
# future_gnn = ray_evaluate_result.remote(checkpoint_indices=checkpoint_indices,
#                             total_timesteps_per_iter = total_timesteps_per_iter,
#                             model_path_base = '/home/kimds929/CodePractice/SimpleSpread_sb3_gnn',
#                             checkpoint_file_basename = 'ppo_decentalize_gnn_checkpoint',
#                             eval_episodes=eval_episodes,
#                             max_cycles=max_cycles)

# # egnn
# future_egnn = ray_evaluate_result.remote(checkpoint_indices=checkpoint_indices,
#                             total_timesteps_per_iter = total_timesteps_per_iter,
#                             model_path_base = '/home/kimds929/CodePractice/SimpleSpread_sb3_egnn',
#                             checkpoint_file_basename = 'ppo_decentalize_egnn_checkpoint',
#                             eval_episodes=eval_episodes,
#                             max_cycles=max_cycles)

# # e2gn2
# future_e2gn2 = ray_evaluate_result.remote(checkpoint_indices=checkpoint_indices,
#                             total_timesteps_per_iter = total_timesteps_per_iter,
#                             model_path_base = '/home/kimds929/CodePractice/SimpleSpread_sb3_e2gn2',
#                             checkpoint_file_basename = 'ppo_decentalize_e2gn2_checkpoint',
#                             eval_episodes=eval_episodes,
#                             max_cycles=max_cycles)

# results_list = ray.get([future_mlp, future_gnn, future_egnn, future_e2gn2])
# # Collect results
# results_dict = {}
# results_dict['MLP'], results_dict['GNN'], results_dict['EGNN'], results_dict['E2GN2'] = results_list


# cPickle.dump(results_dict, open(f"{save_path}/results_dict.pkl", 'wb'))
# print('save result.')




# ######################################################################################################################################################
# ######################################################################################################################################################
results_dict = cPickle.load( open(f"{save_path}/results_dict.pkl", 'rb'))
results_dict

figs = plot_combined_reward_curve(results_dict)
figs = plot_combined_reward_curve(results_dict, save_path=save_path)
print('save.pig')
