import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from mpe2 import simple_spread_v3


# 다중 에이전트 래퍼 (공유 정책용)
class MultiAgentWrapper(gym.Env):
    def __init__(self, max_steps=100):
        super().__init__()
        self.env = simple_spread_v3.parallel_env(max_cycles=max_steps, continuous_actions=True, render_mode=None)
        self.env.reset()
        self.agents = self.env.agents
        self.n_agents = len(self.agents)

        # 관찰 공간은 그대로 Box
        single_obs_space = self.env.observation_space(self.agents[0])
        self.observation_space = gym.spaces.Box(
            low=np.tile(single_obs_space.low, self.n_agents),
            high=np.tile(single_obs_space.high, self.n_agents),
            shape=(single_obs_space.shape[0] * self.n_agents,),
            dtype=single_obs_space.dtype
        )
    
        # 액션 공간: Discrete vs Continuous 구분
        single_action_space = self.env.action_space(self.agents[0])
        if isinstance(single_action_space, gym.spaces.Discrete):
            # Discrete 액션을 가지는 경우
            self.discrete = True
            # 에이전트마다 single_action_space.n 개의 선택지가 있으므로
            # MultiDiscrete([n, n, ..., n])
            self.action_space = gym.spaces.MultiDiscrete(
                [single_action_space.n] * self.n_agents
            )
        else:
            # Continuous(Box) 액션을 가지는 경우
            self.discrete = False
            self.action_space = gym.spaces.Box(
                low=np.tile(single_action_space.low, self.n_agents),
                high=np.tile(single_action_space.high, self.n_agents),
                shape=(single_action_space.shape[0] * self.n_agents,),
                dtype=single_action_space.dtype
            )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        combined_obs = np.concatenate([obs[agent] for agent in self.agents])
        return combined_obs, {}

    def step(self, action):
        if self.discrete:
            # action: shape (n_agents,), 각 원소가 int
            actions = {
                agent: int(action[i])
                for i, agent in enumerate(self.agents)
            }
        else:
            # action: 연속 값 벡터
            action_per_agent = np.split(action, self.n_agents)
            actions = {
                agent: action_per_agent[i]
                for i, agent in enumerate(self.agents)
            }

        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        combined_obs = np.concatenate([obs[agent] for agent in self.agents])

        # 공유 보상: 필요에 따라 수정 가능 (여기서는 agent_0 기준)
        reward = rewards[self.agents[0]]
        terminated = all(terminations.values())
        truncated = all(truncations.values())
        # print(f'reward:{rewards}')
        return combined_obs, reward, terminated, truncated, infos

    def render(self):
        self.env.render()

# 학습 함수
def train_ppo():
    # 환경 생성
    env_fn = lambda: MultiAgentWrapper(max_steps=100)
    env = make_vec_env(env_fn, n_envs=4, vec_env_cls=SubprocVecEnv)

    # PPO 정책 정의
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO(
        policy="MlpPolicy",
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
    )

    # 학습 루프
    total_timesteps = 1000000
    num_iterations = total_timesteps // 2000  # 500
    print(f"Starting training for {num_iterations} iterations")
    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        model.learn(total_timesteps=2000)
        mean_reward = model.rollout_buffer.rewards.mean()
        print(f"Total Environment Reward: {mean_reward:.2f}")
        if i % 10 == 0:
            model.save(f"ppo_shared_checkpoint_{i}")

    env.close()

if __name__ == "__main__":
    train_ppo()