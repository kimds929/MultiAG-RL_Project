# import gymnasium as gym
# import numpy as np
# from stable_baselines3 import PPO
# import time
# from mpe2 import simple_spread_v3

# class MultiAgentWrapper(gym.Env):
#     def __init__(self, max_steps=25, render_mode=None):
#         super().__init__()
#         self.env = simple_spread_v3.parallel_env(max_cycles=max_steps, continuous_actions=True, render_mode=render_mode)
#         # Use possible_agents for PettingZoo compatibility
#         self.agents = self.env.possible_agents
#         self.n_agents = len(self.agents)
#         print(f"Agents: {self.agents}, n_agents: {self.n_agents}")

#         # Test reset to inspect initial output
#         test_obs = self.env.reset()
#         print(f"Initial env reset output: {test_obs}")

#         # Observation space
#         single_obs_space = self.env.observation_space(self.agents[0])
#         self.observation_space = gym.spaces.Box(
#             low=np.tile(single_obs_space.low, self.n_agents),
#             high=np.tile(single_obs_space.high, self.n_agents),
#             shape=(single_obs_space.shape[0] * self.n_agents,),
#             dtype=single_obs_space.dtype
#         )

#         # Action space
#         single_action_space = self.env.action_space(self.agents[0])
#         self.discrete = False  # continuous_actions=True
#         self.action_space = gym.spaces.Box(
#             low=np.tile(single_action_space.low, self.n_agents),
#             high=np.tile(single_action_space.high, self.n_agents),
#             shape=(single_action_space.shape[0] * self.n_agents,),
#             dtype=single_action_space.dtype
#         )

#     def reset(self, seed=None, options=None):
#         try:
#             result = self.env.reset(seed=seed)
#             print(f"Reset result: {result}")  # Debugging
#             # Handle tuple or dict output
#             if isinstance(result, tuple):
#                 obs = result[0] if len(result) > 0 else result
#                 info = result[1] if len(result) > 1 else {}
#             else:
#                 obs = result
#                 info = {}
#             # Ensure obs is a dictionary
#             if not isinstance(obs, dict):
#                 raise ValueError(f"Expected obs to be a dict, got {type(obs)}: {obs}")
#             combined_obs = np.concatenate([obs[agent] for agent in self.agents])
#             print(f"Combined obs shape: {combined_obs.shape}")  # Debugging
#             return combined_obs, info
#         except Exception as e:
#             print(f"Reset failed: {e}")
#             raise

#     def step(self, action):
#         action_per_agent = np.split(action, self.n_agents)
#         actions = {agent: action_per_agent[i] for i, agent in enumerate(self.agents)}

#         obs, rewards, terminations, truncations, infos = self.env.step(actions)
#         combined_obs = np.concatenate([obs[agent] for agent in self.agents])
#         reward = rewards[self.agents[0]]  # agent_0 기준 보상
#         terminated = all(terminations.values())
#         truncated = all(truncations.values())
#         return combined_obs, reward, terminated, truncated, infos

#     def render(self):
#         self.env.render()

#     def close(self):
#         self.env.close()

# # 특정 체크포인트로 렌더링
# def render_checkpoint(checkpoint_idx=490, n_episodes=1):
#     # 환경 생성 (render_mode="human")
#     env = MultiAgentWrapper(max_steps=25, render_mode="human")

#     # 체크포인트 로드
#     model_path = f"ppo_shared_checkpoint_{checkpoint_idx}.zip"
#     try:
#         # Load model without wrapping the environment
#         model = PPO.load(model_path, env=env)
#         print(f"Loaded model from {model_path}")
#     except Exception as e:
#         print(f"Failed to load {model_path}: {e}")
#         env.close()
#         return

#     # 에피소드 실행
#     for episode in range(n_episodes):
#         obs, _ = env.reset()
#         done = False
#         total_reward = 0
#         step = 0

#         print(f"\nEpisode {episode + 1}")
#         while not done:
#             # 모델 예측
#             action, _ = model.predict(obs, deterministic=True)
#             # 환경 스텝
#             obs, reward, terminated, truncated, infos = env.step(action)
#             total_reward += reward
#             done = terminated or truncated
#             step += 1

#             # 렌더링
#             env.render()
#             # 렌더링 속도 조절
#             time.sleep(0.05)  # 50ms 대기

#             # 디버깅 출력
#             print(f"Step {step}: Reward={reward:.2f}, Total Reward={total_reward:.2f}")

#         print(f"Episode {episode + 1} finished: Total Reward={total_reward:.2f}")

#     # 환경 종료
#     env.close()

# if __name__ == "__main__":
#     render_checkpoint(checkpoint_idx=490, n_episodes=1)

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import time
import imageio
import cv2
from mpe2 import simple_spread_v3

class MultiAgentWrapper(gym.Env):
    def __init__(self, max_steps=100, render_mode=None):
        super().__init__()
        self.env = simple_spread_v3.parallel_env(max_cycles=max_steps, continuous_actions=True, render_mode=render_mode)
        self.agents = self.env.possible_agents
        self.n_agents = len(self.agents)
        print(f"Agents: {self.agents}, n_agents: {self.n_agents}")

        # Test reset
        test_obs = self.env.reset()
        print(f"Initial env reset output: {test_obs}")

        # Observation space
        single_obs_space = self.env.observation_space(self.agents[0])
        self.observation_space = gym.spaces.Box(
            low=np.tile(single_obs_space.low, self.n_agents),
            high=np.tile(single_obs_space.high, self.n_agents),
            shape=(single_obs_space.shape[0] * self.n_agents,),
            dtype=single_obs_space.dtype
        )

        # Action space
        single_action_space = self.env.action_space(self.agents[0])
        self.discrete = False
        self.action_space = gym.spaces.Box(
            low=np.tile(single_action_space.low, self.n_agents),
            high=np.tile(single_action_space.high, self.n_agents),
            shape=(single_action_space.shape[0] * self.n_agents,),
            dtype=single_action_space.dtype
        )

    def reset(self, seed=None, options=None):
        try:
            result = self.env.reset(seed=seed)
            print(f"Reset result: {result}")
            if isinstance(result, tuple):
                obs = result[0] if len(result) > 0 else result
                info = result[1] if len(result) > 1 else {}
            else:
                obs = result
                info = {}
            if not isinstance(obs, dict):
                raise ValueError(f"Expected obs to be a dict, got {type(obs)}: {obs}")
            combined_obs = np.concatenate([obs[agent] for agent in self.agents])
            print(f"Combined obs shape: {combined_obs.shape}")
            return combined_obs, info
        except Exception as e:
            print(f"Reset failed: {e}")
            raise

    def step(self, action):
        action_per_agent = np.split(action, self.n_agents)
        actions = {agent: action_per_agent[i] for i, agent in enumerate(self.agents)}

        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        combined_obs = np.concatenate([obs[agent] for agent in self.agents])
        reward = rewards[self.agents[0]]+rewards[self.agents[1]]+rewards[self.agents[2]]
        terminated = all(terminations.values())
        truncated = all(truncations.values())
        return combined_obs, reward, terminated, truncated, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

# 특정 체크포인트로 렌더링하고 GIF 또는 MP4 저장
def render_checkpoint(checkpoint_idx=490, n_episodes=5, save_format="gif", output_path=None):
    # 환경 생성 (render_mode="rgb_array" for frame capture)
    env = MultiAgentWrapper(max_steps=100, render_mode="rgb_array")

    # 체크포인트 로드
    model_path = f"./checkpoints/ppo_shared_checkpoint_{checkpoint_idx}.zip"
    try:
        model = PPO.load(model_path, env=env)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        env.close()
        return

    # 출력 파일 경로 설정
    if output_path is None:
        output_path = f"checkpoint_{checkpoint_idx}.{save_format}"
    
    # GIF 또는 MP4 저장 초기화
    if save_format == "gif":
        writer = imageio.get_writer(output_path, fps=20, loop=0)
    elif save_format == "mp4":
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,  # FPS
            (600, 600)  # 해상도 (simple_spread_v3 기본값, 필요 시 조정)
        )
    else:
        raise ValueError("save_format must be 'gif' or 'mp4'")

    # 에피소드 실행
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        frames = []  # GIF용 프레임 저장

        print(f"\nEpisode {episode + 1}")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, infos = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

            # 프레임 캡처
            frame = env.render()
            if frame is None:
                print("Warning: render() returned None")
                continue

            # 저장 형식에 따라 프레임 처리
            if save_format == "gif":
                frames.append(frame)
            elif save_format == "mp4":
                # RGB -> BGR 변환
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            # 디버깅 출력
            print(f"Step {step}: Reward={reward:.2f}, Total Reward={total_reward:.2f}")

        # GIF 저장: 에피소드 끝난 후
        if save_format == "gif":
            for frame in frames:
                writer.append_data(frame)

        print(f"Episode {episode + 1} finished: Total Reward={total_reward:.2f}")

    # 리소스 해제
    if save_format == "gif":
        writer.close()
    elif save_format == "mp4":
        writer.release()
    env.close()
    print(f"Saved rendering to {output_path}")

if __name__ == "__main__":
    idx=490
    
    # GIF 저장
    render_checkpoint(checkpoint_idx=idx, n_episodes=5, save_format="gif", output_path=f"checkpoint_{idx}.gif")
    
    # MP4 저장 
    # render_checkpoint(checkpoint_idx=idx, n_episodes=5, save_format="mp4", output_path=f"checkpoint_{idx}.mp4")