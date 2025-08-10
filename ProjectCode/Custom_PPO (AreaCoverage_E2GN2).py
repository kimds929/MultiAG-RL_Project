#%% init
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
env_path = r'/home/kimds929/RLproject/AreaCoverage_Env_3'
sys.path.append(env_path)

import AreaCoverage
import numpy as np
import matplotlib.pyplot as plt

import time
from IPython.display import clear_output

# ####################################################################################
# env = AreaCoverage.parallel_env(grid_size=(6, 4), num_agents=3, 
#                        obstacle_spec="random", max_cycles=1000,
#                        render_mode="pyplot", animate=False, local_ratio=0.5)
# obs, infos = env.reset()

# agent_idx = env.agents[1]
# obs[agent_idx][0]
# x_diff_abs = obs[agent_idx][1]
# y_diff_abs = obs[agent_idx][2]
# x_diff_rel = obs[agent_idx][3]
# y_diff_rel = obs[agent_idx][4]

# # (direction) 0:right, 1:down, 2:left, 3:up
# print( np.allclose(x_diff_rel, x_diff_abs) ) # right
# print( np.allclose(x_diff_rel, y_diff_abs) ) # down
# print( np.allclose(x_diff_rel, -x_diff_abs) ) # left
# print( np.allclose(x_diff_rel, -y_diff_abs) ) # up

# env.aec_env.world.agents[0].state['direction']
# env.aec_env.world.agents[1].state['direction']
# env.aec_env.world.agents[2].state['direction']
# plt.imshow(env.render())



# # ####################################################################################
# grid_size = (6,6)
# num_agents = 3
# env = AreaCoverage.parallel_env(grid_size=grid_size, num_agents=num_agents,
#                                 obstacle_spec="bridge", max_cycles=1000, 
#                                 render_mode="pyplot", animate=True, local_ratio=0.7)
# obs, infos = env.reset()
# # plt.imshow(env.render())
# # (action) # 0:stop, 1:forward, 2:right, 3:left
# # (direction) 0:right, 1:down, 2:left, 3:up
# #           [right, down, left, up]

# cumulative_rewards = {agent: 0.0 for agent in env.agents}

# for i in range(50):
#     actions = {agent: env.action_space(agent).sample().item() for agent in env.agents}
#     obs, rewards, terminations, truncations, infos = env.step(actions)
    
#     for agent, r in rewards.items():
#         cumulative_rewards[agent] += r
    
#     # Animation
#     if i % 1 == 0:
#         plt.imshow(env.render())
#         plt.show()
#         time.sleep(0.02)
#         clear_output(wait=True)
# # ####################################################################################






############################################################################################################
# # 【Decentralize】#########################################################################################

import gymnasium as gym

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, TransformedDistribution, SigmoidTransform, Independent

from stable_baselines3.common.policies import ActorCriticPolicy

import supersuit as ss

from stable_baselines3 import PPO
from pettingzoo.utils.wrappers import BaseParallelWrapper

import imageio

# raw_env = AreaCoverage.parallel_env(grid_size=(6, 4), num_agents=2, 
#                        obstacle_spec="random", max_cycles=1000,
#                        render_mode="pyplot", animate=False, local_ratio=0.5)

# raw_env.observation_spaces
# dir(raw_env.observation_space('agent_0'))
# # raw_env.action_spaces
# raw_env.action_space('agent_0')
# # raw_env.possible_agents
# # raw_env.aec_env.agents
# # np.array(dir(raw_env))
# x = obs['agent_0']
# # x[0]
# ca_model = CustomActorCritic(x)
# ca_model.forward(x)
# ca_model.evaluate_actions(x, torch.IntTensor([1]))
# ca_model.predict_values(x)


# obs, infos = wrapper_env.reset()
# obs_agent_0 = torch.tensor(obs['agent_0'])

# cac = CustomActorCritic(wrapper_env.observation_spaces['agent_0'],
#                   wrapper_env.action_spaces['agent_0'],
#                   )

# cac.forward(obs_agent_0)
# cac.evaluate_actions(obs_agent_0, torch.rand(4))



# cac.forward(batch_obs_agent_0)
# cac.evaluate_actions(batch_obs_agent_0, torch.rand(3,4))


########################################################################################################################
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


class FeatureExtractor(nn.Module):
    def __init__(self, num_embeddings=10, embed_dim=4, coord_scale=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)
        self.coord_scale = coord_scale

    def forward(self, obs: torch.Tensor):
        """
        obs: (..., 5, D1, D2, ..., Dn)
             임의 개수의 leading(batch) 차원 + 채널 5 + n개의 spatial dims
        """

        # 1) 채널(=5) 축 찾기
        dims = list(obs.shape)
        nd = obs.ndim
        channel_axis = next(
            i for i, d in enumerate(dims) if d == 5 and i <= nd - 3
        )

        # 2) spatial dims 모두 합치기 → shape: (..., 5, M)
        obs_flat = obs.flatten(start_dim=channel_axis + 1)
        lead_shape = dims[:channel_axis]            # 원래의 batch dims
        M = obs_flat.size(-1)                       # map_size = ∏(spatial dims)
        C = obs_flat.size(channel_axis)             # =5

        # 3) 모든 선행 dims를 하나의 “가상 배치” L로 합치고 reshape
        L = int(torch.prod(torch.tensor(lead_shape))) if lead_shape else 1
        obs2 = obs_flat.reshape(L, C, M)            # (L, 5, M)

        # 4) 채널별로 나누기
        infos  = obs2[:, 0, :].long()               # (L, M)
        rel_x  = obs2[:, 1, :]                      # (L, M)
        rel_y  = obs2[:, 2, :]
        rot_x  = obs2[:, 3, :]
        rot_y  = obs2[:, 4, :]

        # 5) 임베딩 및 좌표 결합
        symbol_h = self.embedding_layer(infos)      # (L, M, E)
        coords   = torch.stack([rot_x, rot_y], dim=-1)  # (L, M, 2)

        # 6) 각 샘플별 self-agent 위치 찾기
        self_idx = (infos == 1).to(torch.int64).argmax(dim=1)  # (L,)

        # 7) hi, xi: fancy-indexing
        batch_idx = torch.arange(L, device=obs.device)
        hi = symbol_h[batch_idx, self_idx]                        # (L, E)
        xi = coords[batch_idx, self_idx] / self.coord_scale       # (L, 2)

        # 8) hj, xj: 자신 제외한 나머지
        arange      = torch.arange(M, device=obs.device)          # (M,)
        arange_row  = arange.unsqueeze(0).expand(L, M)            # (L, M)
        mask        = arange_row != self_idx.unsqueeze(1)         # (L, M)
        other_idxs  = arange_row[mask].view(L, M - 1)             # (L, M-1)
        E = symbol_h.size(-1)
        hj = symbol_h.gather(
            dim=1,
            index=other_idxs.unsqueeze(-1).expand(-1, -1, E)
        )                                                         # (L, M-1, E)
        xj = coords.gather(
            dim=1,
            index=other_idxs.unsqueeze(-1).expand(-1, -1, 2)
        ) / self.coord_scale                                      # (L, M-1, 2)

        # 9) 원래 batch 형태 복원 & 불필요한 1차원 제거
        if lead_shape:
            hi = hi.view(*lead_shape,  -1)      # (..., E)
            xi = xi.view(*lead_shape,  -1)      # (..., 2)
            hj = hj.view(*lead_shape,  M-1, -1) # (..., M-1, E)
            xj = xj.view(*lead_shape,  M-1, 2)  # (..., M-1, 2)
        else:
            hi = hi.squeeze(0)  # (E,)
            xi = xi.squeeze(0)  # (2,)
            hj = hj.squeeze(0)  # (M-1, E)
            xj = xj.squeeze(0)  # (M-1, 2)

        return hi, xi, hj, xj

# EGNN layer
class EGNNLayer(nn.Module):
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
        
        # xi_new
        phi_x_output = self.phi_x(mij)  # (batch, node, 1)
        xi_new = xi +  torch.mean(u_diff * phi_x_output, dim=-2)    # (batch, coord_dim)
        
        # mi
        mi = torch.sum(mij, dim=-2)  # (batch, msg_dim)
        
        # hi_new
        hi_input_concat = torch.cat([hi, mi], dim=-1)   # (batch, node_attr_dim + msg_dim)
        hi_new = self.phi_h(hi_input_concat)        # (batch, node_attr_dim)
        
        return hi_new, xi_new, hj, xj    # (batch, node_attr_dim), (batch, coord_dim)


# [ Customizing Layer ]#################################################################################
class CustomActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                 num_embeddings=10, embed_dim=4,
                 coord_dim=2, msg_dim=32, 
                 **kwargs):
        super().__init__()
        
        # input dimension
        self.obs_dim = observation_space.shape
        self.act_dim = action_space.n
        
        self.map_size = torch.prod(torch.tensor(list(self.obs_dim[1:]))).item()
        self.coord_scale = torch.sqrt(torch.prod(torch.tensor(list(self.obs_dim[1:])))).item()
        
        # actor
        self.actor_net = nn.Sequential(
            FeatureExtractor(num_embeddings, embed_dim, self.coord_scale),
            EGNNLayer(embed_dim, coord_dim, msg_dim),
            EGNNLayer(embed_dim, coord_dim, msg_dim)
        )
        self.actor_head = nn.Linear(embed_dim+coord_dim, self.act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(self.act_dim))  # Learnable log_std
        
        # critic
        self.critic_net = nn.Sequential(
            FeatureExtractor(num_embeddings, embed_dim, self.coord_scale),
            EGNNLayer(embed_dim, coord_dim, msg_dim),
            EGNNLayer(embed_dim, coord_dim, msg_dim),
        )
        self.critic_head = nn.Linear(embed_dim, 1)

    def actor_forward(self, obs):
        hi_out, xi_out, hj, xj = self.actor_net(obs)
        latent_pi_cat = torch.cat([hi_out, xi_out], dim=-1)
        
        # distribution
        logits = self.actor_head(latent_pi_cat)
        dist = torch.distributions.categorical.Categorical(logits=logits)
        return dist
    
    def critic_forward(self, obs):
        hi_out, xi_out, hj, xj = self.critic_net(obs)
        values = self.critic_head(hi_out)
        return values
    
    def forward(self, obs, deterministic=False):
        dist= self.actor_forward(obs)
        if deterministic:
            actions = dist.logits.argmax(dim=-1)
            # actions = dist.mean
        else:
            actions = dist.sample()     # backprop 불가
            # actions = dist.rsample()    # backprop 가능 (reparameterization trick)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.critic_forward(obs)
        return actions, values, log_probs
    
    def evaluate_actions(self, obs, actions):
        dist = self.actor_forward(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic_forward(obs)
        return values, log_probs, entropy
    
    def predict_values(self, obs):
        values = self.critic_forward(obs)
        return values


class CustomizingPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, 
                 embed_dim = 1, msg_dim=32,
                 **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        self.custom_model = CustomActorCritic(observation_space=observation_space,
                                            action_space=action_space, 
                                            embed_dim=embed_dim,
                                            msg_dim=msg_dim).to(self.device)
    
    def forward(self, obs, deterministic=False):
        return self.custom_model.forward(obs, deterministic)
    
    def evaluate_actions(self, obs, actions):
        return self.custom_model.evaluate_actions(obs, actions)

    def predict_values(self, obs):
        return self.custom_model.predict_values(obs)

# [ Parallel Env Wrapper ]#################################################################################

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
        for agent in self.agents:
            origin_obs_space = self.origin_observation_spaces[agent]
            self._observation_spaces[agent] = gym.spaces.Box(
                low = -np.inf,
                high = np.inf,
                shape= self.customize_obs(origin_obs_space.low).shape,
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
    def customize_obs(self, obs):
        # [Customizing] observations
        return obs
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)  # {'agent_0': obs0, 'agent_1': obs1, ...}
        # return {agent: self.customize_obs(obs[agent]) for agent in self.agents}, {}
        return obs, info

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        # [Customizing] reward setting 
        # ...
        # obs = {agent: self.customize_obs(obs[agent]) for agent in self.agents}
        return obs, rewards, terminations, truncations, infos

    # def render(self):
    #     return self.env.render()


raw_env = AreaCoverage.parallel_env(grid_size=(5,5), num_agents=3, 
                       obstacle_spec="random", max_cycles=1000,
                       render_mode="pyplot", animate=False)
wrapper_env = CustomObservationWrapper(raw_env)
vec_env = ss.pettingzoo_env_to_vec_env_v1(wrapper_env)
env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=10, base_class='stable_baselines3')



# raw_env.observation_spaces
# actions = {agent: raw_env.action_space(agent).sample().item() for agent in raw_env.agents}
# raw_env.step(actions)
# wrapper_env.step(actions)
# obs, infos = wrapper_env.reset()
# env.observation_space
# env.action_space
# env.venv.vec_envs    # 병렬 vector_env 
# env.venv.vec_envs[0].par_env  # WapperEnv
# env.venv.vec_envs[0].par_env.customize_obs()
# np.array(dir(env))
# np.array(dir(env.unwrapped))
# env.unwrapped.vec_envs

# --------------------------------------------------------------
# raw_env = AreaCoverage.parallel_env(grid_size=(6, 4), num_agents=3, 
#                        obstacle_spec="random", max_cycles=1000,
#                        render_mode="pyplot", animate=False, local_ratio=0.5)
# obs, infos = raw_env.reset()

# raw_env.scenario.observation(raw_env.world.agents[0], raw_env.world).shape



# agent_idx = raw_env.agents[1]
# obs[agent_idx][0]
# x_diff_abs = obs[agent_idx][1]
# y_diff_abs = obs[agent_idx][2]
# x_diff_rel = obs[agent_idx][3]
# y_diff_rel = obs[agent_idx][4]

# # (direction) 0:right, 1:down, 2:left, 3:up
# print( np.allclose(x_diff_rel, x_diff_abs) ) # right
# print( np.allclose(x_diff_rel, y_diff_abs) ) # down
# print( np.allclose(x_diff_rel, -x_diff_abs) ) # left
# print( np.allclose(x_diff_rel, -y_diff_abs) ) # up

# raw_env.world.agents[0].state['direction']
# raw_env.world.agents[1].state['direction']
# raw_env.world.agents[2].state['direction']
# plt.imshow(raw_env.render())


##########################################################################################



max_cycles = 300

N_ITER = 500
TOTAL_TIMESTEPS = 2000        # Rollout Timstep
N_REPLAY_STEPS = 500         # Rollout 횟수
N_EPOCHS = 10              # Model Learning 횟수


checkpoint_path = '/home/kimds929/CodePractice/AreaCoverageE2GN2'
if "ppo_decentalize_checkpoint.zip" in os.listdir(checkpoint_path):
    model_path = f"{checkpoint_path}/ppo_decentalize_checkpoint.zip"
    model = PPO.load(model_path, env=env)
    
    print('*** Load Model.')
else:
    # PPO 정책 정의
    policy_kwargs = dict(embed_dim=4, msg_dim=32)
    
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


print(f"Starting training for {N_ITER} iterations")
for i in range(N_ITER):
    print(f"\n--- Iteration {i+1}/{N_ITER} ---")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    print_summary(model, output_keys=['train'])
    mean_reward = model.rollout_buffer.rewards.mean()
    print(f"Total Environment Reward: {mean_reward:.2f}")
    
    # (gif_save)  ########################################################################################
    if (i % 5 == 0) or (i == N_ITER-1):
        save_path = f"/home/kimds929/CodePractice/AreaCoverageE2GN2"
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



# ####################################################################################
# states, infos = wrapper_env.reset()

# states.keys()
# states['agent_0'][0]
# states['agent_1'][0]
# states['agent_2'][0]
# env_agents = raw_env.agents

# # plt.imshow(env.render())
# # (action) # 0:stop, 1:forward, 2:right, 3:left
# # (direction) 0:right, 1:down, 2:left, 3:up
# #           [right, down, left, up]

# for i in range(50):
#     # Animation
#     if i % 1 == 0:
#         plt.imshow(raw_env.render())
#         plt.show()
#         time.sleep(0.01)
#         clear_output(wait=True)
    
#     actions_dict = {}
#     # 각 agent별로 개별적으로 action 예측
#     for agent in env_agents:
#         obs = states[agent]
#         action, _ = model.predict(obs, deterministic=True)  # deterministic=True for evaluation
#         actions_dict[agent] = action.item()

#     states, rewards, terminations, truncations, infos = raw_env.step(actions_dict)
#     # if terminated:
#     #     print("Goal reached!", "Reward:", reward)
#     #     break
#     # elif truncated:
#     #     print("Fail to find goal", "Reward:", reward)
#     #     break

# # np.array(dir(model))
# # model.policy
# # model.policy_class
# # np.array(dir(model.policy_class)[-50:])

# # model.policy_class.predict?