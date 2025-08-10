import gymnasium as gym

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from mpe2 import simple_spread_v3


from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.utils.wrappers import BaseParallelWrapper
from pettingzoo import AECEnv

import imageio
import supersuit as ss

import torch
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticPolicy

# # 【CustomModel】#################################################################################
class CustomActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                 embed_dim=2, graph_embed_hidden_dim=None, graph_hidden_dim=8, 
                 net_arch=dict(pi=[64, 64], vf=[64, 64])):
        super().__init__()
        
    

class CustomizingPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, 
                 net_arch = dict(pi=[64, 64], vf=[64, 64]),
                 embed_dim = 1, graph_embed_hidden_dim=None, graph_hidden_dim=8,
                 **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        self.custom_model = CustomActorCritic(observation_space=observation_space,
                                            action_space=action_space, 
                                            embed_dim=embed_dim,
                                            graph_embed_hidden_dim=graph_embed_hidden_dim,
                                            graph_hidden_dim=graph_hidden_dim, 
                                            net_arch=net_arch).to(self.device)
    
    def forward(self, obs, deterministic=False):
        return self.custom_model.forward(obs, deterministic)
    
    def evaluate_actions(self, obs, actions):
        return self.custom_model.evaluate_actions(obs, actions)

    def predict_values(self, obs):
        return self.custom_model.predict_values(obs)

# # 【Decentralize】#################################################################################
# # Observation Space를 Customizing 하기 위한 Wrapper 
class CustomObservationWrapper(BaseParallelWrapper):
    """
    PettingZoo parallel_env를 감싸서,
    에이전트별 observation을 원하는 대로 커스터마이징합니다.
    반드시 BaseParallelWrapper.__init__을 호출해야 .agents가 정상 인식됩니다.
    """
    def __init__(self, env):
        # 1) 반드시 super().__init__을 통해 내부 속성(.agents 등)을 초기화하게끔 합니다.
        super().__init__(env)
        self.env = env
        self.env.reset()
        self.agents = self.env.agents
        
        # observation_spaces
        self.origin_observation_spaces = self.observation_spaces.copy()
        self._observation_spaces = {}
        self.set_observation_space()
        # self.discrete_action_space = None
    
    def set_observation_space(self): 
        obs, infos = self.env.reset()
        obs_shape = self.customize_obs(obs, self.agents[0]).shape
        for agent in self.agents:
            origin_obs_space = self.origin_observation_spaces[agent]
            self._observation_spaces[agent] = gym.spaces.Box(
                low = -np.inf,
                high = np.inf,
                shape= obs_shape,
                dtype = origin_obs_space.dtype
                )
    
    @property
    def observation_spaces(self):
        return self._observation_spaces
    
    def observation_space(self, agent):
        return self._observation_spaces[agent]
            
    def set_action_space(self):
        raw_action_space = self.env.action_space(self.agents[0])
        
        if isinstance(raw_action_space, gym.spaces.Discrete):
            self.discrete_action_space = True
        else:
            self.discrete_action_space = False
    
    # ★★★★★★
    def customize_obs(self, obs, agent):
        # obs_position = np.array([])
        obs_velocity = np.array([])
        obs_rel_landmarks = obs[agent][4:10]
        obs_rel_agents = obs[agent][10:14]
        for agent_name in self.agents:
            # obs_position = np.concat([obs_position, obs[agent_name][2:4]], axis=0)
            obs_velocity = np.concat([obs_velocity, obs[agent_name][0:2]], axis=0)
        
        custom_obs = np.concat([ obs_velocity, obs_rel_landmarks, obs_rel_agents])
        return custom_obs
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)  # {'agent_0': obs0, 'agent_1': obs1, ...}
        return {agent: self.customize_obs(obs, agent) for agent in self.agents}, {}

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        # [Customizing] reward setting 
        # ...
        obs = {agent: self.customize_obs(obs, agent) for agent in self.agents}
        return obs, rewards, terminations, truncations, infos

    # def render(self):
    #     return self.env.render()

raw_env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=25)
wrapper_env = CustomObservationWrapper(raw_env)
vec_env = ss.pettingzoo_env_to_vec_env_v1(wrapper_env)
env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=10, base_class='stable_baselines3')

# obs, infos = wrapper_env.reset()
# raw_env.observation_space('agent_0')
# raw_env.observation_spaces
# raw_env.aec_env.world.agents
# wrapper_env.observation_space('agent_0')
# wrapper_env.observation_spaces
# env.observation_space
# env.venv.vec_envs    # 병렬 vector_env 
# env.venv.vec_envs[0].par_env  # WapperEnv
# env.venv.vec_envs[0].par_env.customize_obs()

##########################################################################################

N_ITER = 1000
TOTAL_TIMESTEPS = 500 * 10        # Rollout Timstep
N_REPLAY_STEPS = 500         # Rollout 횟수
N_EPOCHS = 10              # Model Learning 횟수
max_cycles = 50

checkpoint_path = '/home/kimds929/CodePractice/SimpleSpread_sb3'
if "ppo_decentalize_checkpoint.zip" in os.listdir(checkpoint_path):
    model_path = f"{checkpoint_path}/ppo_decentalize_checkpoint.zip"
    model = PPO.load(model_path, env=env)
    
    print('*** Load Model.')
else:
    # PPO 정책 정의
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=N_REPLAY_STEPS,
        batch_size=64,
        n_epochs=N_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )
    print('*** Create New Model.')

# 학습 루프
print(f"Starting training for {N_ITER} iterations")
for i in range(N_ITER):
    print(f"\n--- Iteration {i+1}/{N_ITER} ---")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    mean_reward = model.rollout_buffer.rewards.mean()
    print(f"Total Environment Reward: {mean_reward:.2f}")
    # if i % 10 == 0:
    #     model.save(f"ppo_shared_checkpoint_{i}")
    
    # (gif_save)  ########################################################################################
    if (i % 20 == 0) or (i == N_ITER-1):
        save_path = f"/home/kimds929/CodePractice/SimpleSpread_sb3"
        model.save(f"{save_path}/ppo_decentalize_checkpoint_{i}")   # backup
        model.save(f"{save_path}/ppo_decentalize_checkpoint")   # recent
        writer = imageio.get_writer(f"{save_path}/iter_{i}.gif", fps=20, loop=0)

        states, info = wrapper_env.reset()
        env_agents = wrapper_env.possible_agents
        for step_idx in range(max_cycles):
            actions_dict = {}
            # 각 agent별로 개별적으로 action 예측
            for agent in env_agents:
                obs = states[agent]
                action, _ = model.predict(obs, deterministic=True)  # deterministic=True for evaluation
                actions_dict[agent] = action.item()

            # step 진행
            next_states, rewards, terminations, truncations, infos = wrapper_env.step(actions_dict)
            
            # save_gif
            writer.append_data(wrapper_env.render())
            if np.array(list(terminations.values())).all() or np.array(list(truncations.values())).all():
                break
            else:
                states = next_states

        writer.close()
        print('save gif.')
    ########################################################################################


# ################################################################################################################################
# # Episode Steps -----------------------------------------------------------------
# max_cycles = 50

# env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=max_cycles, continuous_actions=False)
# env_agents = env.possible_agents

# states, info = env.reset()
# for step_idx in range(max_cycles):
#     actions_dict = {}

#     # 각 agent별로 개별적으로 action 예측
#     for agent in env_agents:
#         obs = states[agent]
#         action, _ = model.predict(obs[:4], deterministic=True)  # deterministic=True for evaluation
#         actions_dict[agent] = action.item()

#     # step 진행
#     states, rewards, terminations, truncations, infos = env.step(actions_dict)

#     # 시각화 (3 step마다)
#     if step_idx % 3 == 0:
#         plt.imshow(env.render())
#         plt.axis("off")
#         plt.show()
#         time.sleep(0.05)
#         clear_output(wait=True)

# env.close()


















################################################################################

# from stable_baselines3.common.evaluation import evaluate_policy
# # 체크포인트 평가 및 그래프 그리기
# def plot_reward_curve():
#     # 환경 생성
#     env = MultiAgentWrapper(max_steps=100)

#     # 체크포인트 인덱스 (0, 10, 20, ..., 490)
#     checkpoint_indices = list(range(0, 491, 10))
#     env_steps = []
#     mean_rewards = []
#     std_rewards = []

#     # 각 체크포인트 평가
#     total_timesteps_per_iter = 2000  # 학습 스크립트에서 한 iteration당 2000 timesteps
#     for idx in checkpoint_indices:
#         # 체크포인트 로드
#         model_path = f"./checkpoints/ppo_shared_checkpoint_{idx}.zip"
#         model = PPO.load(model_path, env=env)

#         # 환경 스텝 수 계산
#         steps = idx * total_timesteps_per_iter
#         env_steps.append(steps)

#         # 정책 평가
#         mean_reward, std_reward = evaluate_policy(
#             model,
#             env,
#             n_eval_episodes=100,  # 10번의 에피소드로 평가
#             deterministic=True
#         )

#         mean_rewards.append(mean_reward)
#         std_rewards.append(std_reward)
#         print(f"Checkpoint {idx}: Steps={steps}, Mean Reward={mean_reward:.2f}, Std Reward={std_reward:.2f}")

#     # 그래프 그리기
#     plt.figure(figsize=(10, 6))
#     plt.plot(env_steps, mean_rewards, label="Mean Reward", color="blue")
#     plt.fill_between(
#         env_steps,
#         np.array(mean_rewards) - np.array(std_rewards),
#         np.array(mean_rewards) + np.array(std_rewards),
#         color="blue",
#         alpha=0.2,
#         label="±1 Std"
#     )
#     plt.xlabel("Environment Steps")
#     plt.ylabel("Mean Reward")
#     plt.title("PPO Training: Reward vs Environment Steps")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("ppo_reward_curve.png")
#     plt.show()
