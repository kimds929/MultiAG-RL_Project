import numpy as np

class Action:  # action of the agent
    def __init__(self):
        self.act = None

class GridAgent:
    def __init__(self, name, init_pos, init_dir=0, policy=None, grid_shape=(5,5)):
        self.name = name
        self.state = {
            "p_pos": init_pos,       # (y, x)
            "direction": init_dir    # 0:right, 1:down, 2:left, 3:up
        }
        self.action = Action()
        self.action_callback = None
        self.policy = policy
        self.wall_memory = np.zeros(grid_shape, dtype=np.float32)
    
    def action_callback(self, obs, world):
        if self.policy is not None:
            return self.policy(obs, world)
        return 0  # default: stop

# class TurnRightPolicy:
#     def __call__(self, obs, world):
#         return 2  # turn right

# agent = GridAgent(name="agent_0", init_pos=(0,0), policy=TurnRightPolicy())