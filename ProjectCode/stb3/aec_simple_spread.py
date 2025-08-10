import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from mpe2 import simple_spread_v3

# === MultiAgentWrapper_AEC ===
class MultiAgentWrapper_AEC(gym.Env):
    def __init__(self, max_steps=25):
        super().__init__()
        # AEC env 사용
        self.env = simple_spread_v3.env(max_cycles=max_steps, continuous_actions=True)
        self.env.reset()
        self.agents = self.env.agents
        self.n_agents = len(self.agents)

        # Agent 별 관측, 액션 space
        self.obs_spaces = [self.env.observation_space(agent) for agent in self.agents]
        self.action_spaces = [self.env.action_space(agent) for agent in self.agents]

        # Gym Env 처럼 단일 agent 기준으로 설정 (shared policy 학습 가능)
        self.observation_space = self.obs_spaces[0]
        self.action_space = self.action_spaces[0]

        # Internal state
        self.current_agent_idx = 0

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        self.current_agent_idx = 0
        current_agent = self.agents[self.current_agent_idx]

        # obs 가져오기
        obs, _, _, _, _ = self.env.last()

        return obs, {}


    def step(self, action):
        current_agent = self.agents[self.current_agent_idx]
        # 현재 agent 상태
        obs, reward, termination, truncation, info = self.env.last()

        # action 적용
        self.env.step(action)

        # 다음 agent 로 이동
        self.current_agent_idx += 1
        done = False

        if self.current_agent_idx >= self.n_agents:
            # 모든 agent step 후 → 다음 cycle
            self.current_agent_idx = 0
            done = all(self.env.terminations.values()) or all(self.env.truncations.values())

        next_agent = self.agents[self.current_agent_idx]
        next_obs = self.env.observe(next_agent)

        return next_obs, reward, done, False, info

    def render(self):
        self.env.render()

# === Shared PPO 학습 함수 ===
def train_shared_ppo():
    # Shared PPO 모델 하나만 사용
    env_fn = lambda: MultiAgentWrapper_AEC(max_steps=25)
    env = make_vec_env(env_fn, n_envs=4, vec_env_cls=SubprocVecEnv)

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

    total_timesteps = 1_000_000
    num_iterations = total_timesteps // 2000

    print(f"Starting shared PPO training for {num_iterations} iterations")
    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        model.learn(total_timesteps=2000)

        # Logging
        mean_reward = model.rollout_buffer.rewards.mean()
        print(f"Shared PPO - Mean Reward: {mean_reward:.2f}")

        if i % 10 == 0:
            model.save(f"./checkpoints_ippo/ppo_shared_checkpoint_{i}")

    env.close()

# === Main 실행 ===
if __name__ == "__main__":
    train_shared_ppo()
