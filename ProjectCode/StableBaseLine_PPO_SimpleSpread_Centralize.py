import gymnasium as gym

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from mpe2 import simple_spread_v3



# (Util Funciton) ##############################################################################################################
# Allclose_either
def allclose_either(a, b, atol=1e-8, rtol=1e-5):
    abs_diff = np.abs(a - b)
    cond1 = abs_diff <= atol        # 절대 오차 조건
    cond2 = abs_diff <= rtol * np.abs(b)  # 상대 오차 조건
    return np.all(cond1 | cond2)    # 둘 중 하나라도 True면 통과

# Check Finish
def check_finish(env, rtol=1e-2, atol=1e-8):
    agent_pos = [env.unwrapped.world.agents[ai].state.p_pos for ai in range(env.num_agents)]
    landmarks_pos = [env.unwrapped.world.landmarks[ai].state.p_pos for ai in range(env.num_agents)]
    match = np.array([[allclose_either(ap, lp, rtol=rtol, atol=atol) for lp in landmarks_pos] for ap in agent_pos])
    return np.sum(match,axis=0).all() and np.sum(match,axis=1).all()
################################################################################################################################


# 다중 에이전트 래퍼 (공유 정책용)
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
        reward = rewards[self.agents[0]]
        terminated = all(terminations.values())
        truncated = all(truncations.values())
        # print(f'reward:{rewards}')
        return combined_obs, reward, terminated, truncated, infos

    def render(self):
        self.env.render()
################################################################################################################################
n_envs = 4
env_fns = [lambda: MultiAgentWrapper(max_steps=50) for _ in range(n_envs)]
env = DummyVecEnv(env_fns)








################################################################################################################################
# # 【 PythonFile Env 】# ▶ Jupyter 셀에서 벡터 환경을 만들 때는 SubprocVecEnv 대신 DummyVecEnv 쓰기
# env_fn = lambda: MultiAgentWrapper(max_steps=25)
# env = make_vec_env(env_fn, n_envs=4, vec_env_cls=SubprocVecEnv)

# 【 Jupyter Env 】# ▶ Jupyter 셀에서 벡터 환경을 만들 때는 SubprocVecEnv 대신 DummyVecEnv 쓰기
n_envs = 4
env_fns = [lambda: MultiAgentWrapper(max_steps=50) for _ in range(n_envs)]
env = DummyVecEnv(env_fns)

# ----------------------------------------------------------------------------------------

env_fns
# (TRAINING-ITERATION)
# iter : N_ITER
#   ㄴ loop : num_loop = (TOTAL_TIMESTEPS-1)//N_REPLAY_STEPS+1     # 전체 loop 횟수
#       ㄴ rolout : N_REPLAY_STEPS (loop내 rollout 횟수)
#       ㄴ learning : N_EPOCHS (loop내 model backprop 횟수)
N_ITER = 100
TOTAL_TIMESTEPS = 500 * 50        # Rollout Timstep
N_REPLAY_STEPS = 500         # Rollout 횟수
N_EPOCHS = 10               # Model Learning 횟수

env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=25, continuous_actions=True)

# PPO 정책 정의
policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=N_REPLAY_STEPS,
    batch_size=1000,
    n_epochs=N_EPOCHS,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    policy_kwargs=policy_kwargs,
)

# 학습 루프
print(f"Starting training for {N_ITER} iterations")
for i in range(N_ITER):
    print(f"\n--- Iteration {i+1}/{N_ITER} ---")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    mean_reward = model.rollout_buffer.rewards.mean()
    print(f"Total Environment Reward: {mean_reward:.2f}")
    if i % 10 == 0:
        model.save(f"ppo_shared_checkpoint_{i}")
    env.close()




################################################################################################################################
# Episode Steps -----------------------------------------------------------------
max_cycles = 50

env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=max_cycles, continuous_actions=True)
env_agents = env.possible_agents

states, info = env.reset()
for step_idx in range(max_cycles):
    cur_actions = {}
    cur_done = {agent_name:False for agent_name in env_agents}

    state = np.stack([v for v in states.values()]).reshape(-1)
    actions, _ = model.predict(state)
    actions_dict = {agent:action for agent, action in zip(env_agents, np.split(actions, 3))}
    # for agent_idx, agent in enumerate(env_agents):
    #     # state = np.stack([v[:14] for v in states.values()]).reshape(-1)
    #     state = states[agent][:14]
    #     state_tensor = torch.FloatTensor(state).to(device)
    #     action, log_prob, entropy = policy_network.forward(state_tensor, deterministic=True)
    #     # action = policy_network.explore_action(state_tensor)
    #     cur_actions[agent] = action.item()
    
    # step       
    next_states, reward, terminations, truncation, info = env.step(actions_dict)
    
    if (step_idx==max_cycles-1) or check_finish(env) or np.array(list(truncation.values())).all():
        break
    else:
        states = next_states

    # Animation
    if step_idx % 3 == 0:
        plt.imshow(env.render())
        plt.show()
        time.sleep(0.05)
        clear_output(wait=True)
env.close()



















































# # 【Decentralize】#################################################################################
from pettingzoo.utils.wrappers import BaseParallelWrapper
import supersuit as ss

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
        return obs[:4]
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)  # {'agent_0': obs0, 'agent_1': obs1, ...}
        return {agent: self.customize_obs(obs[agent]) for agent in self.agents}, {}

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        # [Customizing] reward setting 
        # ...
        obs = {agent: self.customize_obs(obs[agent]) for agent in self.agents}
        return obs, rewards, terminations, truncations, infos

    # def render(self):
    #     return self.env.render()


env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=25)
env = CustomObservationWrapper(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=10, base_class='stable_baselines3')

##########################################################################################

N_ITER = 100
TOTAL_TIMESTEPS = 500 * 10        # Rollout Timstep
N_REPLAY_STEPS = 500         # Rollout 횟수
N_EPOCHS = 3              # Model Learning 횟수
max_cycles = 25


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

# 학습 루프
print(f"Starting training for {N_ITER} iterations")
for i in range(N_ITER):
    print(f"\n--- Iteration {i+1}/{N_ITER} ---")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    mean_reward = model.rollout_buffer.rewards.mean()
    print(f"Total Environment Reward: {mean_reward:.2f}")
    # if i % 10 == 0:
    #     model.save(f"ppo_shared_checkpoint_{i}")
    env.close()


env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=25)
env = CustomObservationWrapper(env)
env.reset()
env.observation_spaces



################################################################################################################################
# Episode Steps -----------------------------------------------------------------
max_cycles = 50

env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=max_cycles, continuous_actions=False)
env_agents = env.possible_agents

states, info = env.reset()
for step_idx in range(max_cycles):
    actions_dict = {}

    # 각 agent별로 개별적으로 action 예측
    for agent in env_agents:
        obs = states[agent]
        action, _ = model.predict(obs[:4], deterministic=True)  # deterministic=True for evaluation
        actions_dict[agent] = action.item()

    # step 진행
    states, rewards, terminations, truncations, infos = env.step(actions_dict)

    # 시각화 (3 step마다)
    if step_idx % 3 == 0:
        plt.imshow(env.render())
        plt.axis("off")
        plt.show()
        time.sleep(0.05)
        clear_output(wait=True)

env.close()

