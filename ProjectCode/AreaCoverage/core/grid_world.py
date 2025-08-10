import numpy as np
from collections import Counter
from AreaCoverage.core.utils import pattern_walls, sample_random_walls
from AreaCoverage.core.agent import GridAgent


class GridWorld:
    def __init__(self, grid_size=(5, 5), num_agents=3, obstacle_spec="random", rng=None):
        self.grid_size_y, self.grid_size_x = grid_size
        self.num_agents = num_agents
        self.rng = rng if rng is not None else np.random.RandomState()
        self.dir_dict = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

        self.agents = []
        self.agent_names = [f"agent_{i}" for i in range(num_agents)]

        self.shared_grid = np.full((self.grid_size_y, self.grid_size_x), 9, dtype=np.int32)
        
        self.wall_bumped = {a.name: False for a in self.agents}
        self. conflicted_agents = set()
        self.lazy_agents = {a.name: False for a in self.agents}
        self.newly_covered = {a.name: False for a in self.agents}

        self.obstacle_spec = obstacle_spec
        self.reset(self.rng) 

    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def _init_obstacles(self, spec):
        self.walls = np.zeros((self.grid_size_y, self.grid_size_x), dtype=bool)
        if spec == "random":
            num = int(0.25 * self.grid_size_y * self.grid_size_x)
            self.walls = sample_random_walls(self.rng, self.grid_size_y, self.grid_size_x, num)
        else:
            self.walls = pattern_walls(self.obstacle_spec, self.grid_size_y, self.grid_size_x)

    def reset(self, rng=None):
        self.rng = rng or self.rng
        self._init_obstacles(self.obstacle_spec)
        self._init_agents()
        self._init_coverage()
        self.todo = np.sum(~self.walls)
        self.all_covered_reward_given = False
        self.shared_grid = np.full((self.grid_size_y, self.grid_size_x), 9, dtype=np.int32)

    def _init_agents(self):
        self.agents = []
        occupied = self.walls.copy()
        for i, name in enumerate(self.agent_names):
            while True:
                y = self.rng.randint(self.grid_size_y)
                x = self.rng.randint(self.grid_size_x)
                if not occupied[y, x]:
                    break
            occupied[y, x] = True
            agent = GridAgent(name, init_pos=(y, x), init_dir=self.rng.randint(4), grid_shape=(self.grid_size_y, self.grid_size_x))
            self.agents.append(agent)

    def _init_coverage(self):
        self.coverage = np.zeros((self.grid_size_y, self.grid_size_x), dtype=int)
        self.coverage_owner = -np.ones((self.grid_size_y, self.grid_size_x), dtype=int)
        for i, agent in enumerate(self.agents):
            y, x = agent.state["p_pos"]
            self.coverage[y, x] = 1
            self.coverage_owner[y, x] = i

    def step(self):
        """Main public interface."""
        self.previous_positions = {a.name: a.state["p_pos"] for a in self.agents}
        for agent in self.scripted_agents:
            agent.action.act = agent.compute_action(obs=None, world=self)
        proposed = self.compute_proposed_positions()
        resolved = self.resolve_conflicts(proposed)
        self.apply_positions(resolved)

    def compute_proposed_positions(self) -> dict:
        self.wall_bumped = {agent.name: False for agent in self.agents}
        proposed = {}
        for agent in self.agents:
            act = agent.action.act
            y, x = agent.state["p_pos"]
            d = agent.state["direction"]

            if act == 1:  # move forward
                dy, dx = self.dir_dict[d]
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_size_y and 0 <= nx < self.grid_size_x and not self.walls[ny, nx]:
                    proposed[agent.name] = (ny, nx)
                else:
                    self.wall_bumped[agent.name] = True
                    proposed[agent.name] = (y, x)
            elif act == 2:  # turn right
                agent.state["direction"] = (d + 1) % 4
                proposed[agent.name] = (y, x)
            elif act == 3:  # turn left
                agent.state["direction"] = (d - 1) % 4
                proposed[agent.name] = (y, x)
            else:  # stop
                proposed[agent.name] = (y, x)
        return proposed

    # resolve_conflicts의 대상 -> penalty를 부여할 수 있는 대상
    def resolve_conflicts(self, proposed: dict) -> dict:
        self.conflicted_agents = set()
        old_positions = {a.name: a.state["p_pos"] for a in self.agents}
        counts = Counter(proposed.values())

        # 1. 같은 칸 충돌
        for agent in self.agents:
            tgt = proposed[agent.name]
            if counts[tgt] > 1:
                self.conflicted_agents.add(agent.name)
                proposed[agent.name] = old_positions[agent.name]

        # 2. swap conflict
        for a1 in self.agents:
            for a2 in self.agents:
                if a1 == a2:
                    continue
                if proposed[a1.name] == old_positions[a2.name] and proposed[a2.name] == old_positions[a1.name]:
                    self.conflicted_agents.add(a1.name)
                    self.conflicted_agents.add(a2.name)
                    proposed[a1.name] = old_positions[a1.name]
                    proposed[a2.name] = old_positions[a2.name]

        return proposed

    def apply_positions(self, resolved: dict):
        self.newly_covered = {agent.name: False for agent in self.agents}
        self.lazy_agents = {agent.name: False for agent in self.agents}

        for i, agent in enumerate(self.agents):
            old_pos = self.previous_positions[agent.name]
            new_pos = resolved[agent.name]
            agent.state["p_pos"] = new_pos

            if new_pos == old_pos:
                self.lazy_agents[agent.name] = True

            if self.coverage[new_pos] == 0:
                self.coverage[new_pos] = 1
                self.coverage_owner[new_pos] = i
                self.newly_covered[agent.name] = True

    @property
    def entities(self):
        return self.agents
