from pettingzoo.utils.conversions import parallel_wrapper_fn
from AreaCoverage.envs.raw_env import raw_env


def env(**kwargs):
    return raw_env(**kwargs)

parallel_env = parallel_wrapper_fn(raw_env)
