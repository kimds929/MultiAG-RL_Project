import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from mpe2 import simple_spread_v3
from stable_baselines3.common.distributions import DiagGaussianDistribution
import time

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLP 클래스
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# GNN 레이어 클래스
class GNNLayer(nn.Module):
    def __init__(self, node_dim, edge_attr_dim, hidden_dim, output_dim):
        super(GNNLayer, self).__init__()
        self.phi_e = MLP(node_dim * 2 + edge_attr_dim, hidden_dim, hidden_dim)
        self.phi_h = MLP(node_dim + hidden_dim, hidden_dim, output_dim)
        
    def forward(self, h, edge_index, edge_attr):
        num_nodes = h.size(0)
        src, dst = edge_index
        h_i = h[src]
        h_j = h[dst]
        edge_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        m_ij = self.phi_e(edge_input)
        m_i = torch.zeros(num_nodes, m_ij.size(-1), device=h.device)
        m_i.scatter_add_(0, dst.unsqueeze(-1).expand(-1, m_ij.size(-1)), m_ij)
        node_input = torch.cat([h, m_i], dim=-1)
        h_next = self.phi_h(node_input)
        return h_next

# GNN 모델
class GNN(nn.Module):
    def __init__(self, node_dim, edge_attr_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(node_dim, edge_attr_dim, hidden_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GNNLayer(hidden_dim, edge_attr_dim, hidden_dim, hidden_dim))
        self.layers.append(GNNLayer(hidden_dim, edge_attr_dim, hidden_dim, output_dim))
        
    def forward(self, h, edge_index, edge_attr):
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return h

# 커스텀 GNN 정책
class GNNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, node_dim, edge_attr_dim, hidden_dim, num_layers, **kwargs):
        super(GNNPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.node_dim = node_dim
        self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_agents = 3
        
        # 액션 네트워크 (정책)
        self.actor = GNN(node_dim, edge_attr_dim, hidden_dim, action_space.shape[0] // self.n_agents, num_layers).to(device)
        # 가치 네트워크
        self.critic = GNN(node_dim, edge_attr_dim, hidden_dim, 1, num_layers).to(device)
        
        # 로그 표준편차 초기화
        self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]).to(device), requires_grad=True)
        
    def forward(self, obs, deterministic=False):
        # obs: [batch_size, n_agents * node_dim + n_edges * edge_attr_dim]
        obs = obs.to(device)
        node_features = obs[:, :self.n_agents * self.node_dim].reshape(-1, self.n_agents, self.node_dim)
        edge_attr = obs[:, self.n_agents * self.node_dim:].reshape(-1, self.n_agents * (self.n_agents - 1), self.edge_attr_dim)
        edge_index = torch.tensor([[0,0,1,1,2,2], [1,2,0,2,0,1]], dtype=torch.long, device=device)
        
        batch_size = node_features.size(0)
        actions = []
        values = []
        log_probs = []
        
        for i in range(batch_size):
            h = node_features[i]
            e_attr = edge_attr[i]
            
            # 액션 예측
            action_mean = self.actor(h, edge_index, e_attr).view(-1)
            action_dist = DiagGaussianDistribution(action_mean.size(0))
            action_dist.proba_distribution(action_mean, self.log_std)
            action = action_dist.mode() if deterministic else action_dist.sample()
            actions.append(action)
            log_prob = action_dist.log_prob(action)
            log_probs.append(log_prob)
            
            # 가치 예측
            value = self.critic(h, edge_index, e_attr).mean()
            values.append(value)
        
        actions = torch.stack(actions)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        
        return actions, values, log_probs

    def evaluate_actions(self, obs, actions):
        obs = obs.to(device)
        node_features = obs[:, :self.n_agents * self.node_dim].reshape(-1, self.n_agents, self.node_dim)
        edge_attr = obs[:, self.n_agents * self.node_dim:].reshape(-1, self.n_agents * (self.n_agents - 1), self.edge_attr_dim)
        edge_index = torch.tensor([[0,0,1,1,2,2], [1,2,0,2,0,1]], dtype=torch.long, device=device)
        
        batch_size = node_features.size(0)
        values = []
        log_probs = []
        entropies = []
        
        for i in range(batch_size):
            h = node_features[i]
            e_attr = edge_attr[i]
            
            action_mean = self.actor(h, edge_index, e_attr).view(-1)
            action_dist = DiagGaussianDistribution(action_mean.size(0))
            action_dist.proba_distribution(action_mean, self.log_std)
            log_prob = action_dist.log_prob(actions[i])
            entropy = action_dist.entropy()
            
            value = self.critic(h, edge_index, e_attr).mean()
            
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
        
        return torch.stack(values), torch.stack(log_probs), torch.stack(entropies)

# 다중 에이전트 래퍼
class MultiAgentWrapper(gym.Env):
    def __init__(self, max_steps=100):
        super().__init__()
        self.env = simple_spread_v3.parallel_env(max_cycles=max_steps, continuous_actions=True, render_mode=None)
        self.env.reset()
        self.agents = self.env.agents[:3]
        self.n_agents = len(self.agents)
        
        single_obs_space = self.env.observation_space(self.agents[0])
        self.node_dim = single_obs_space.shape[0]
        self.edge_attr_dim = 2
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents * self.node_dim + (self.n_agents * (self.n_agents - 1)) * self.edge_attr_dim,),
            dtype=np.float32
        )
        
        single_action_space = self.env.action_space(self.agents[0])
        self.action_space = gym.spaces.Box(
            low=np.tile(single_action_space.low, self.n_agents),
            high=np.tile(single_action_space.high, self.n_agents),
            shape=(single_action_space.shape[0] * self.n_agents,),
            dtype=single_action_space.dtype
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        node_features = np.concatenate([obs[agent] for agent in self.agents])
        edge_attr = self._compute_edge_attr(obs)
        combined_obs = np.concatenate([node_features, edge_attr.flatten()])
        return combined_obs, {}

    def step(self, action):
        action_per_agent = np.split(action, self.n_agents)
        actions = {agent: action_per_agent[i] for i, agent in enumerate(self.agents)}
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        node_features = np.concatenate([obs[agent] for agent in self.agents])
        edge_attr = self._compute_edge_attr(obs)
        combined_obs = np.concatenate([node_features, edge_attr.flatten()])
        reward = sum(rewards.values())
        terminated = all(terminations.values())
        truncated = all(truncations.values())
        return combined_obs, reward, terminated, truncated, infos

    def _compute_edge_attr(self, obs):
        edge_attr = []
        for i, agent_i in enumerate(self.agents):
            pos_i = obs[agent_i][2:4]
            for j, agent_j in enumerate(self.agents):
                if i != j:
                    pos_j = obs[agent_j][2:4]
                    rel_pos = pos_j - pos_i
                    edge_attr.append(rel_pos)
        return np.array(edge_attr)

    def render(self):
        self.env.render()

# 학습 함수
def train_ppo():
    env_fn = lambda: MultiAgentWrapper(max_steps=100)
    env = make_vec_env(env_fn, n_envs=8, vec_env_cls=SubprocVecEnv)
    
    policy_kwargs = dict(
        node_dim=env.get_attr('node_dim')[0],
        edge_attr_dim=env.get_attr('edge_attr_dim')[0],
        hidden_dim=32,
        num_layers=2
    )
    
    model = PPO(
        policy=GNNPolicy,
        env=env,
        learning_rate=3e-4,
        n_steps=2000 // 4,
        batch_size=1000,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=device
    )
    
    total_timesteps = 1000000
    num_iterations = total_timesteps // 2000
    print(f"Starting training for {num_iterations} iterations")
    
    # List to store iteration times
    iteration_times = []
    
    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        start_time = time.time()  # Start timing
        model.learn(total_timesteps=2000)
        end_time = time.time()  # End timing
        iteration_time = end_time - start_time  # Calculate duration
        iteration_times.append(iteration_time)
        mean_reward = model.rollout_buffer.rewards.mean()
        print(f"Total Environment Reward: {mean_reward:.2f}")
        print(f"Iteration Time: {iteration_time:.2f} seconds")
        if i % 10 == 0:
            model.save(f"ppo_gnn_checkpoint_{i}")
    
    env.close()
    
    # Print average iteration time
    avg_iteration_time = sum(iteration_times) / len(iteration_times)
    print(f"\nAverage Iteration Time: {avg_iteration_time:.2f} seconds")

if __name__ == "__main__":
    train_ppo()
