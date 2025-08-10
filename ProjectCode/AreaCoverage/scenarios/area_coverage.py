from AreaCoverage.core.grid_world import GridWorld
import numpy as np

class Scenario:
    def __init__(self):
        pass

    def make_world(self, num_agents=3, grid_size=(5, 5), obstacle_spec="random"):
        return GridWorld(grid_size=grid_size,
                         num_agents=num_agents,
                         obstacle_spec=obstacle_spec)

    def reset_world(self, world, np_random=None):
        # world.reset(rng=np_random)
        world.reset()

    def reward(self, agent, world):
        name = agent.name
        reward = -0.05  # ⏱️ 시간 패널티 기본값

        lazy_map = getattr(world, "lazy_agents", {})
        bump_map = getattr(world, "wall_bumped", {})
        # # 🧱 벽과 충돌 (움직이려 했지만 못감 + 벽)
        if lazy_map.get(name, False) and bump_map.get(name, False):
             reward += -1.0

        # 🤜 agent 간 충돌
        elif name in world.conflicted_agents:
            reward += -2.0

        # ✅ 새 땅 커버
        elif world.newly_covered.get(name, False):
            reward += +5.0

        # 🔁 이미 방문한 셀 이동
        elif not world.newly_covered.get(name, False):
            reward += -0.1

        # 🎯 완전 커버 보상 (한 번만)
        if np.all((world.coverage == 1) | (world.walls)):
            reward += 100.0
            world.all_covered_reward_given = True

        return reward


    def update_shared_grid(self, agent, world):
        directional_offsets = {
            0: [(0,  0), (0,  1), (-1,  1), (1,  1)],   # →
            1: [(0,  0), (1,  0), (1,  1), (1, -1)],    # ↓
            2: [(0,  0), (0, -1), (1, -1), (-1, -1)],   # ←
            3: [(0,  0), (-1, 0), (-1, -1), (-1, 1)]    # ↑
        }

        ay, ax = agent.state["p_pos"]
        d = agent.state["direction"]
        grid_y, grid_x = world.grid_size_y, world.grid_size_x

        for dy, dx in directional_offsets[d]:
            ny, nx = ay + dy, ax + dx
            if 0 <= ny < grid_y and 0 <= nx < grid_x:
                # self.compute_cell_state(agent, world, ny, nx)
                if world.walls[ny, nx] == 1:
                    world.shared_grid[ny, nx] = 6
                elif world.coverage[ny, nx] == 1:
                    world.shared_grid[ny, nx] = 7
                elif world.shared_grid[ny, nx] != 6:  # 이미 벽으로 기록된 건 그대로 두고
                    world.shared_grid[ny, nx] = 8


    # def compute_cell_state(self, agent, world, y, x):
    #     ay, ax = agent.state["p_pos"]
    #     ad = agent.state["direction"]

    #     if (y, x) == (ay, ax): return 1                     # 1: 해당 칸이 해당 agent

    #     for other in world.agents:
    #         if other.name == agent.name: continue
    #         oy, ox = other.state["p_pos"]
    #         if (y, x) == (oy, ox): 
    #             rel = (other.state["direction"] - ad) % 4
    #             if rel == 0: return 2                       # 2: 해당 칸에 같은 방향을 보고있는 다른 agent
    #             elif rel == 1: return 3                     # 3: 해당 칸에 해당 agent 기준 오른쪽 방향을 보고있는 다른 agent
    #             elif rel == 3: return 4                     # 4: 해당 칸에 해당 agent 기준 왼쪽 방향을 보고있는 다른 agent
    #             elif rel == 2: return 5                     # 5: 해당 칸에 해당 agent 기준 뒷쪽 방향을 보고있는 다른 agent

    #     if world.walls[y, x] == 1: return 6                 # 6: 해당 칸이 벽
    #     if world.coverage[y, x] == 1: return 7              # 7: 해당 칸이 covered
    #     if world.shared_grid[y, x] != 9: return 8           # 8: 해당 칸을 본적은 있지만 uncovered
    #     return 9                                            # 9: 어떠한 agent도 아직 해당 칸을 본적이 없음음


    def observation(self, agent, world):
        ay, ax = agent.state["p_pos"]
        d = agent.state["direction"]
        grid_y, grid_x = world.grid_size_y, world.grid_size_x

        feat = np.zeros((grid_y, grid_x, 5), dtype=np.float32)
        self.update_shared_grid(agent, world)

        for i in range(grid_y):
            for j in range(grid_x):
                state = world.shared_grid[i, j]

                # 현재 칸이 agent 자신의 위치인지?
                if (i, j) == (ay, ax):
                    # state = 11 + int(agent.name[-1])
                    state = 1

                # 다른 agent가 이 칸에 있다면 방향에 따라 덮어씀
                for other in world.agents:
                    if other.name == agent.name:
                        continue
                    oy, ox = other.state["p_pos"]
                    if (i, j) == (oy, ox):
                        rel = (other.state["direction"] - d) % 4
                        if rel == 0:   state = 2
                        elif rel == 1: state = 3
                        elif rel == 3: state = 4
                        elif rel == 2: state = 5

                feat[i, j, 0] = state

                # 절대 거리
                dx = j - ax
                dy = ay - i
                feat[i, j, 1] = dx
                feat[i, j, 2] = dy

                # 방향 기준 상대 거리
                if d == 0:   dx_local, dy_local = dx, dy       # →
                elif d == 1: dx_local, dy_local = -dy, dx      # ↓
                elif d == 2: dx_local, dy_local = -dx, -dy     # ←
                elif d == 3: dx_local, dy_local = dy, -dx      # ↑

                feat[i, j, 3] = dx_local
                feat[i, j, 4] = dy_local

        return feat.transpose(2, 0, 1)