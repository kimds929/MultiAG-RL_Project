import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from mpe2 import simple_spread_v3

class MultiAgentWrapper(gym.Env):
    def __init__(self, max_steps=25):
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
        # reward = rewards[self.agents[0]]
        reward = rewards[self.agents[0]]+rewards[self.agents[1]]+rewards[self.agents[2]]
        terminated = all(terminations.values())
        truncated = all(truncations.values())
        # print(f'reward:{rewards}')
        return combined_obs, reward, terminated, truncated, infos

    def render(self):
        self.env.render()

# 체크포인트 평가 및 그래프 그리기
def plot_reward_curve():
    # 환경 생성
    env = MultiAgentWrapper(max_steps=25)

    # 체크포인트 인덱스 (0, 10, 20, ..., 490)
    checkpoint_indices = list(range(0, 491, 10))
    env_steps = []
    mean_rewards = []
    std_rewards = []

    # 각 체크포인트 평가
    total_timesteps_per_iter = 2000  # 학습 스크립트에서 한 iteration당 2000 timesteps
    for idx in checkpoint_indices:
        # 체크포인트 로드
        model_path = f"./checkpoints/ppo_shared_checkpoint_{idx}.zip"
        model = PPO.load(model_path, env=env)

        # 환경 스텝 수 계산
        steps = idx * total_timesteps_per_iter
        env_steps.append(steps)

        # 정책 평가
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=10,  # 10번의 에피소드로 평가
            deterministic=True
        )

        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)
        print(f"Checkpoint {idx}: Steps={steps}, Mean Reward={mean_reward:.2f}, Std Reward={std_reward:.2f}")

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(env_steps, mean_rewards, label="Mean Reward", color="blue")
    plt.fill_between(
        env_steps,
        np.array(mean_rewards) - np.array(std_rewards),
        np.array(mean_rewards) + np.array(std_rewards),
        color="blue",
        alpha=0.2,
        label="±1 Std"
    )
    plt.xlabel("Environment Steps")
    plt.ylabel("Mean Reward")
    plt.title("PPO Training: Reward vs Environment Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig("ppo_reward_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_reward_curve()