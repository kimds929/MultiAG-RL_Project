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
from torch.distributions import Categorical, Normal, TransformedDistribution, SigmoidTransform, Independent

from stable_baselines3.common.policies import ActorCriticPolicy



# print summary
def print_summary(model, output_keys=None, indent=4):
    # 기본 출력 항목 설정
    default_keys = [
        "eval/mean_ep_length", "eval/mean_reward",
        "rollout/ep_len_mean", "rollout/ep_rew_mean",
        "time/fps", "time/iterations", "time/time_elapsed", "time/total_timesteps",
        "train/approx_kl", "train/clip_fraction", "train/clip_range", "train/entropy_loss",
        "train/explained_variance", "train/learning_rate", "train/loss", "train/n_updates",
        "train/policy_gradient_loss", "train/std", "train/value_loss"
    ]
    
    # 출력 대상 설정 (인자가 없으면 기본값 사용)
    if output_keys is None:
        output_keys = default_keys
        
    if (type(output_keys) == list)  and ('eval' in output_keys):
        output_keys += ["eval/mean_ep_length", "eval/mean_reward"]
        output_keys.remove('eval')
    
    if (type(output_keys) == list)  and ('rollout' in output_keys):
        output_keys += ["rollout/ep_len_mean", "rollout/ep_rew_mean"]
        output_keys.remove('rollout') 
        
    if (type(output_keys) == list)  and ('time' in output_keys):
        output_keys += ["time/fps", "time/iterations", "time/time_elapsed", "time/total_timesteps"]
        output_keys.remove('time')
    
    if (type(output_keys) == list)  and ('train' in output_keys):
        output_keys += ["train/approx_kl", "train/clip_fraction", "train/clip_range", "train/entropy_loss",
        "train/explained_variance", "train/learning_rate", "train/loss", "train/n_updates",
        "train/policy_gradient_loss", "train/std", "train/value_loss"]
        output_keys.remove('train')
        
    logs = model.logger.name_to_value

    # 포맷 설정 (항목명, 값에 대해 정해진 길이로 출력)
    def fmt(k, val, indent=0):
        # 값이 없으면 'N/A' 출력
        if val is None or val == '':
            val = 'N/A'
        # 값이 숫자라면 소수점 4자리로 출력하고, 없으면 그대로
        elif isinstance(val, (int, float)):
            val = f"{val:.4f}" if isinstance(val, float) else f"{val}"

        # indent 값에 따라 들여쓰기 처리, 최대 25자 항목명, 값은 12자리로 고정
        return f"| {' ' * indent}{k:<25} | {val:<12} |"

    print("-" * 55)

    # 평가 정보 출력
    if any(k in output_keys for k in ["eval/mean_ep_length", "eval/mean_reward"]):
        print(f"| {'eval/':<25}{' '*indent} | {'':<12} |")
        if "eval/mean_ep_length" in output_keys:
            print(fmt("mean_ep_length", logs.get("eval/mean_ep_length", ""), indent=indent))
        if "eval/mean_reward" in output_keys:
            print(fmt("mean_reward", logs.get("eval/mean_reward", ""), indent=indent))

    # 롤아웃 정보 출력
    if any(k in output_keys for k in ["rollout/ep_len_mean", "rollout/ep_rew_mean"]):
        print(f"| {'rollout/':<25}{' '*indent} | {'':<12} |")
        if "rollout/ep_len_mean" in output_keys:
            print(fmt("ep_len_mean", logs.get("rollout/ep_len_mean", ""), indent=indent))
        if "rollout/ep_rew_mean" in output_keys:
            print(fmt("ep_rew_mean", logs.get("rollout/ep_rew_mean", ""), indent=indent))

    # 시간 정보 출력
    if any(k in output_keys for k in ["time/fps", "time/iterations", "time/time_elapsed", "time/total_timesteps"]):
        print(f"| {'time/':<25}{' '*indent} | {'':<12} |")
        if "time/fps" in output_keys:
            print(fmt("fps", logs.get("time/fps", ""), indent=indent))
        if "time/iterations" in output_keys:
            print(fmt("iterations", logs.get("time/iterations", ""), indent=indent))
        if "time/time_elapsed" in output_keys:
            print(fmt("time_elapsed", logs.get("time/time_elapsed", ""), indent=indent))
        if "time/total_timesteps" in output_keys:
            print(fmt("total_timesteps", logs.get("time/total_timesteps", ""), indent=indent))

    # 학습 정보 출력
    if any(k in output_keys for k in [
        "train/approx_kl", "train/clip_fraction", "train/clip_range", "train/entropy_loss", 
        "train/explained_variance", "train/learning_rate", "train/loss", "train/n_updates",
        "train/policy_gradient_loss", "train/std", "train/value_loss"]):
        print(f"| {'train/':<25}{' '*indent} | {'':<12} |")
        if "train/approx_kl" in output_keys:
            print(fmt("approx_kl", logs.get("train/approx_kl", ""), indent=indent))
        if "train/clip_fraction" in output_keys:
            print(fmt("clip_fraction", logs.get("train/clip_fraction", ""), indent=indent))
        if "train/clip_range" in output_keys:
            print(fmt("clip_range", logs.get("train/clip_range", ""), indent=indent))
        if "train/entropy_loss" in output_keys:
            print(fmt("entropy_loss", logs.get("train/entropy_loss", ""), indent=indent))
        if "train/explained_variance" in output_keys:
            print(fmt("explained_variance", logs.get("train/explained_variance", ""), indent=indent))
        if "train/learning_rate" in output_keys:
            print(fmt("learning_rate", logs.get("train/learning_rate", ""), indent=indent))
        if "train/loss" in output_keys:
            print(fmt("loss", logs.get("train/loss", ""), indent=indent))
        if "train/n_updates" in output_keys:
            print(fmt("n_updates", logs.get("train/n_updates", ""), indent=indent))
        if "train/policy_gradient_loss" in output_keys:
            print(fmt("policy_gradient_loss", logs.get("train/policy_gradient_loss", ""), indent=indent))
        if "train/std" in output_keys:
            print(fmt("std", logs.get("train/std", ""), indent=indent))
        if "train/value_loss" in output_keys:
            print(fmt("value_loss", logs.get("train/value_loss", ""), indent=indent))

    print("-" * 55)
# print_summary(model, output_keys=['train/policy_gradient_loss', 'train/value_loss'])
# print_summary(model, output_keys=['train'])
# print_summary(model)







# (batch, node, attribute) -> (batch, node, )
# # 【CustomModel】#################################################################################
class E2GN2Layer(nn.Module):
    def __init__(self, node_attr_dim, coord_dim, msg_dim=32):
        """
            node_attr_dim is for absolute attribute.
            coord_dim is for relative attribute.
        """
        super().__init__()
        self.phi_e = nn.Sequential(nn.Linear(node_attr_dim*2 + coord_dim, msg_dim),
                                   nn.SiLU())
        self.phi_x = nn.Sequential(nn.Linear(msg_dim, 1),
                                   nn.SiLU())
        self.phi_x2 = nn.Sequential(nn.Linear(msg_dim, 1),
                                   nn.SiLU())
        self.phi_h = nn.Sequential(nn.Linear(node_attr_dim+msg_dim, node_attr_dim),
                                   nn.SiLU())
    
    def forward(self, obs):
        """
        Args:
            hi (batch, node_attr_dim)
            xi (batch, coord_dim)
            hj (batch, node_attr_dim)
            xj (batch, coord_dim)

        Returns:
            hi_new, xi_new       # (batch, node_attr_dim), (batch, coord_dim)
        """
        (hi, xi, hj, xj) = obs
        hi_unsqueeze = hi.unsqueeze(-2)     # (batch, 1, node_attr_dim)
        xi_unsqueeze = xi.unsqueeze(-2)     # (batch, 1, node_attr_dim)
        
        hi_shape = hi_unsqueeze.shape     # (batch, 1, node_attr_dim)
        hj_shape = hj.shape     # (batch, node, node_attr_dim)
        repeat_shape = list(torch.ones_like(torch.tensor(hi_shape)))
        repeat_shape[-2] = hj_shape[-2]
        
        hi_broad = hi_unsqueeze.repeat(repeat_shape)
        xi_broad = xi_unsqueeze.repeat(repeat_shape)
        
        # mij
        u_diff = xi_broad - xj      # (batch, node, coord_dim)
        u_2norm = (u_diff)**2       # (batch, node, coord_dim)
        
        mij_input_concat = torch.cat([hi_broad, hj, u_2norm], dim=-1)   # (batch, node, 2*node_attr_dim + coord_dim)
        mij = self.phi_e(mij_input_concat)  # (batch, node, msg_dim)
        
        # mi
        mi = torch.sum(mij, dim=-2)  # (batch, msg_dim)
        # xi_new
        phi_x_output = self.phi_x(mij)  # (batch, node, 1)
        phi_x2_output = self.phi_x2(mi)   # (batch, 1)
        xi_new = xi * phi_x2_output +  torch.mean(u_diff * phi_x_output, dim=-2)    # (batch, coord_dim)
        
        # hi_new
        hi_input_concat = torch.cat([hi, mi], dim=-1)   # (batch, node_attr_dim + msg_dim)
        hi_new = self.phi_h(hi_input_concat)        # (batch, node_attr_dim)
        
        return hi_new, xi_new, hj, xj    # (batch, node_attr_dim), (batch, coord_dim)

# hi= torch.rand(2,3)
# hj= torch.rand(2,4,3)

# xi = torch.rand(2,5)
# xj = torch.rand(2,4,5)

# e2gn2 = E2GN2Layer(3,5)
# e2gn2((hi, xi, hj, xj))
# h, x = e2gn2(hi, xi, hj, xj)
        

class CustomActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                 node_attr_dim=2, coord_dim=2, msg_dim=32, 
                 **kwargs):
        super().__init__()
        
        # input dimension
        self.obs_dim = observation_space.shape
        self.act_dim = action_space.shape[0]
        
        # actor
        self.actor_net = nn.Sequential(
            E2GN2Layer(node_attr_dim, coord_dim, msg_dim),
            E2GN2Layer(node_attr_dim, coord_dim, msg_dim),
        )
        self.actor_head = nn.Linear(node_attr_dim+coord_dim, self.act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(self.act_dim))  # Learnable log_std
        
        # critic
        self.critic_net = nn.Sequential(
            E2GN2Layer(node_attr_dim, coord_dim, msg_dim),
            E2GN2Layer(node_attr_dim, coord_dim, msg_dim),
        )
        self.critic_head = nn.Linear(node_attr_dim, 1)
    
    def feature_decompose(self, obs):
        hi = torch.mean(obs[..., 0:2], dim=-2)    # (batch, 2)
        xi = torch.mean(obs[..., 2:4], dim=-2)    # (batch, 2)
        hj = obs[..., 4:6]      # (batch, node, 2)
        xj = obs[..., 4:6]      # (batch, node, 2)
        return (hi, xi, hj, xj)
    
    def actor_forward(self, obs):
        tuple_input = self.feature_decompose(obs)
        hi_out, xi_out, hj, xj = self.actor_net(tuple_input)
        latent_pi_cat = torch.cat([hi_out, xi_out], dim=-1)
        
        # distribution
        mean = self.actor_head(latent_pi_cat)
        std = torch.exp(self.actor_log_std)  # Convert log_std to std
        base_dist = Independent(Normal(mean,std), reinterpreted_batch_ndims=0)     # reinterpreted_batch_ndims : 마지막 몇개 차원을 event group으로 볼것인가?
        dist = TransformedDistribution(base_dist, [SigmoidTransform()])
        return dist, base_dist
    
    def critic_forward(self, obs):
        tuple_input = self.feature_decompose(obs)
        hi_out, xi_out, hj, xj = self.critic_net(tuple_input)
        values = self.critic_head(hi_out)
        return values
    
    def forward(self, obs, deterministic=False):
        dist, base_dist = self.actor_forward(obs)
        if deterministic:
            actions = dist.mean
        else:
            actions = dist.sample()     # backprop 불가
            # actions = dist.rsample()    # backprop 가능 (reparameterization trick)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.critic_forward(obs)
        return actions, values, log_probs
    
    def evaluate_actions(self, obs, actions):
        dist, base_dist = self.actor_forward(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = base_dist.entropy().sum(dim=-1)
        values = self.critic_forward(obs)
        return values, log_probs, entropy
    
    def predict_values(self, obs):
        values = self.critic_forward(obs)
        return values

# cac = CustomActorCritic(torch.rand(4,8), torch.rand(5,4))
# cac.forward(torch.rand(5,8))
# cac.forward(torch.rand(4,5,8))
# cac.evaluate_actions(torch.rand(5,8), torch.rand(5))
# cac.evaluate_actions(torch.rand(4,5,8), torch.rand(4,5))



# ----------------------------------------------------
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
    
    # [Customizing을 위한 필수 Method 1] (obs, deterministic=False) → (actions, values, log_probs)
    def forward(self, obs, deterministic=False):
        return self.custom_model.forward(obs, deterministic)

    # (Customizing을 위한 필수 Method 2) (obs, actions) → (values, log_probs, entropy)    
    def evaluate_actions(self, obs, actions):
        return self.custom_model.evaluate_actions(obs, actions)

    # (Customizing을 위한 필수 Method 3) (obs) → (values)
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
        custom_obs = []
        cur_agent_velpos = obs[agent][:4]  # [velocity[0:2], abs_pos[2:4]]
        agent_abs = {}
        for i, agent_name in enumerate(self.agents):
            landmark_velpos = np.zeros(4)
            landmark_velpos[2:] = self.env.unwrapped.world.landmarks[i].state.p_pos
            custom_obs.append(np.concat([cur_agent_velpos, landmark_velpos]))
            
            agent_vel = self.env.unwrapped.world.agents[i].state.p_vel
            agent_pos = self.env.unwrapped.world.agents[i].state.p_pos
            agent_abs[agent_name] = np.concat([agent_vel, agent_pos])
        
        for other_agent in self.agents:
            if other_agent != agent:
                custom_obs.append( np.concat([cur_agent_velpos, agent_abs[other_agent]]) )
        return np.stack(custom_obs)
            # [cur_ag_velpos[0:4],  land1_abs_velpos[4:8], 
            #                       land2_abs_velpos[4:8], 
            #                       land3_abs_velpos[4:8],
            #                       other_ag1_velpos[4:8],
            #                       other_ag2_velpos[4:8] ]
    
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

max_cycles = 100
raw_env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=max_cycles, continuous_actions=True)
wrapper_env = CustomObservationWrapper(raw_env)
vec_env = ss.pettingzoo_env_to_vec_env_v1(wrapper_env)
env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=10, base_class='stable_baselines3')


# obs, infos = wrapper_env.reset()
# np.round(obs['agent_0'],3)
# np.round(obs['agent_1'],3)
# np.round(obs['agent_2'],3)
# raw_env.observation_space('agent_0')
# raw_env.observation_spaces
# raw_env.aec_env.world.agents
# wrapper_env.observation_space('agent_0')
# wrapper_env.observation_spaces
# wrapper_env.state()
# wrapper_env.unwrapped.world.agents[1].state.p_vel
# wrapper_env.unwrapped.world.agents[1].state.p_pos       # (parallel mode)
# wrapper_env.unwrapped.world.landmarks[0].state.p_pos    # (parallel mode)

# env.observation_space
# env.venv.vec_envs    # 병렬 vector_env 
# env.venv.vec_envs[0].par_env  # WapperEnv
# env.venv.vec_envs[0].par_env.customize_obs()

##########################################################################################

N_ITER = 500
TOTAL_TIMESTEPS = 2000        # Rollout Timstep
N_REPLAY_STEPS = 500         # Rollout 횟수
N_EPOCHS = 10              # Model Learning 횟수

checkpoint_path = '/home/kimds929/CodePractice/SimpleSpread_sb3_e2gn2'
if "ppo_decentalize_e2gn2_checkpoint.zip" in os.listdir(checkpoint_path):
    model_path = f"{checkpoint_path}/ppo_decentalize_e2gn2_checkpoint.zip"
    model = PPO.load(model_path, env=env)
    
    print('*** Load Model.')
else:
    # PPO 정책 정의
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO(
        policy=CustomizingPolicy,
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
    print_summary(model, output_keys=['train'])
    mean_reward = model.rollout_buffer.rewards.mean()
    print(f"Total Environment Reward: {mean_reward:.2f}")
    # if i % 10 == 0:
    #     model.save(f"ppo_shared_checkpoint_{i}")
    
    # (gif_save)  ########################################################################################
    if (i % 5 == 0) or (i == N_ITER-1):
        save_path = f"/home/kimds929/CodePractice/SimpleSpread_sb3_e2gn2"
        model.save(f"{save_path}/ppo_decentalize_e2gn2_checkpoint{i}")   # backup
        model.save(f"{save_path}/ppo_decentalize_e2gn2_checkpoint")   # recent
        writer = imageio.get_writer(f"{save_path}/iter_{i}.gif", fps=20, loop=0)

        states, info = wrapper_env.reset()
        env_agents = wrapper_env.possible_agents
        for step_idx in range(max_cycles):
            actions_dict = {}
            # 각 agent별로 개별적으로 action 예측
            for agent in env_agents:
                obs = states[agent]
                action, _ = model.predict(obs, deterministic=True)  # deterministic=True for evaluation
                actions_dict[agent] = action

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


################################################################################################################################
# Episode Steps -----------------------------------------------------------------


# # env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=max_cycles, continuous_actions=False)

# env_agents = wrapper_env.possible_agents

# states, infos = wrapper_env.reset()
# for step_idx in range(max_cycles):
#     actions_dict = {}

#     # 각 agent별로 개별적으로 action 예측
#     for agent in env_agents:
#         obs = states[agent]
#         action, _ = model.predict(obs, deterministic=True)  # deterministic=True for evaluation
#         actions_dict[agent] = action

#     # step 진행
#     states, rewards, terminations, truncations, infos = wrapper_env.step(actions_dict)

#     # 시각화 (3 step마다)
#     if step_idx % 3 == 0:
#         plt.imshow(wrapper_env.render())
#         plt.axis("off")
#         plt.show()
#         time.sleep(0.05)
#         clear_output(wait=True)













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
