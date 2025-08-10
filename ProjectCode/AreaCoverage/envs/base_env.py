import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector
from gymnasium import spaces
from AreaCoverage.core.utils import render_grid
import cv2
import imageio
import os

class AreaCoverageBaseEnv(AECEnv):
    metadata = {
        "render_modes": ["pyplot", "cv2"],
        "name": "area_coverage_v0",
        "is_parallelizable": True,
    }

    def __init__(self, scenario, world, render_mode="cv2", animate=True,
                 max_cycles=1000, local_ratio=0.5,
                 ):
        super().__init__()
        self.scenario = scenario
        self.world = world
        self.render_mode = render_mode
        self.animate = animate
        self.max_cycles = max_cycles
        self.local_ratio = local_ratio

        self.render_frames = []

        self.agents = [agent.name for agent in world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {name: i for i, name in enumerate(self.agents)}
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.steps = 0
        self.current_actions = [None] * len(self.agents)

        self.reset()

    def _accumulate_rewards(self):
        self._cumulative_rewards = {a: self._cumulative_rewards[a] + self.rewards[a] for a in self.agents}

    def _setup_spaces(self):
        self.action_spaces = {}
        self.observation_spaces = {}

        for agent in self.world.agents:
            self.action_spaces[agent.name] = spaces.Discrete(4)  # stop, forward, right, left
            obs_shape = self.scenario.observation(agent, self.world).shape
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
            )

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()

        self.scenario.reset_world(self.world, self.np_random)
        self.world.all_covered_reward_given = False

        self.render_frames = []

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.steps = 0
        self.current_actions = [None] * len(self.agents)

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._setup_spaces()
        self.obs = {agent: self.observe(agent) for agent in self.agents}

        return self.obs, self.infos

    def observe(self, agent):
        return self.scenario.observation(self.world.agents[self._index_map[agent]], self.world)

    def state(self):
        return [self.observe(agent) for agent in self.agents]
        # return {agent: self.observe(agent) for agent in self.agents}
    
    def step(self, action):
        agent = self.agent_selection
        idx = self._index_map[agent]
        next_idx = (idx + 1) % self.world.num_agents
        self.agent_selection = self._agent_selector.next()
        self.current_actions[idx] = action
        self.agents = [agent for agent in self.agents if not (self.terminations[agent] or self.truncations[agent])]
        if next_idx == 0:
            for i, a in enumerate(self.agents):
                self.world.agents[i].action.act = self.current_actions[i]

            self.world.step()
            self.steps += 1

            for i, agent in enumerate(self.agents):
                reward = self.scenario.reward(self.world.agents[i], self.world)
                self.rewards[agent] = reward

            # covered_cells = (self.world.coverage == 1).sum()
            # if covered_cells >= self.world.todo:
            if self.world.all_covered_reward_given:
                self.terminations = {a: True for a in self.agents}
                self.close_called = True

            if self.steps >= self.max_cycles:
                self.truncations = {a: True for a in self.agents}
                self.close_called = True
        else:
            self._clear_rewards()

        self._accumulate_rewards()    
        # if (self.world.all_covered_reward_given):
        #     print(self.steps,", ".join(f"{agent.replace('_',' ')}: {self._cumulative_rewards[agent]:.3f}" for agent in self.agents))        
        return (self.obs, self.rewards, self.terminations, self.truncations, self.infos)

    def render(self):
        frame, close = render_grid(self.steps,
            grid_size_x=self.world.grid_size_x,
            grid_size_y=self.world.grid_size_y,
            walls=self.world.walls,
            coverage=self.world.coverage,
            coverage_owner=self.world.coverage_owner,
            agent_positions=[a.state["p_pos"] for a in self.world.agents],
            agent_directions=[a.state["direction"] for a in self.world.agents],
            mode=self.render_mode,
            animate=self.animate,
            info=self.world.shared_grid,
        )
        self.render_frames.append(frame.copy())
        self.close_called = close
        return frame

    def close(self):
        if hasattr(self, "render_frames") and self.render_frames:
            gif_path = f"{self.world.grid_size_y}x{self.world.grid_size_x}_N{self.world.num_agents}_{self.steps}.gif"
            os.makedirs("gifs", exist_ok=True)
            gif_path = os.path.join("gifs", gif_path)
            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in self.render_frames]
            imageio.mimsave(gif_path, frames_rgb, duration=0.005, fps=40)
            print(f"🎞️ GIF saved to {gif_path}")

        # 창 닫기
        if getattr(self, "close_called", False):
            cv2.destroyAllWindows()
