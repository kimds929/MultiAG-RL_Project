import numpy as np
from gymnasium.utils import EzPickle

from AreaCoverage.scenarios.area_coverage import Scenario
from AreaCoverage.envs.base_env import AreaCoverageBaseEnv


class raw_env(AreaCoverageBaseEnv, EzPickle):
    def __init__(self, num_agents=3, grid_size=(5, 5),
                 obstacle_spec="random", local_ratio=0.5,
                 max_cycles=1000, render_mode=None, animate=True):
        EzPickle.__init__(self, num_agents=num_agents, grid_size=grid_size,
                          obstacle_spec=obstacle_spec, local_ratio=local_ratio,
                          max_cycles=max_cycles, render_mode=render_mode, animate=animate)

        assert 0.0 <= local_ratio <= 1.0, "local_ratio must be between 0 and 1"

        scenario = Scenario()
        world = scenario.make_world(num_agents=num_agents,
                                    grid_size=grid_size,
                                    obstacle_spec=obstacle_spec,)

        AreaCoverageBaseEnv.__init__(self, scenario=scenario, world=world,
                                     render_mode=render_mode, max_cycles=max_cycles,
                                     local_ratio=local_ratio,)

        self.metadata["name"] = "area_coverage_v0"
