from typing import Optional, Union, Callable

from ray.rllib.algorithms import ppo, AlgorithmConfig
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_SAMPLED
from ray.rllib.utils.typing import ResultDict, PartialAlgorithmConfigDict, EnvType
from ray.tune import result
from ray.tune.logger import Logger


class TimedPPO(ppo.PPO):
    """
    This class behave like PPO, but add timesteps_this_iter to results.
    Without doing so, trainer cannot load checkpoints correctly
    (e.g., it will set the time steps counter to zero, consequently restarting schedulers.)
    """

    def __init__(self,
                 config: Optional[Union[PartialAlgorithmConfigDict, AlgorithmConfig]] = None,
                 env: Optional[Union[str, EnvType]] = None,
                 logger_creator: Optional[Callable[[], Logger]] = None,
                 **kwargs):
        super().__init__(config, env, logger_creator, **kwargs)
        self._num_agent_steps_sampled = 0
        self._loaded = False

    def step(self) -> ResultDict:
        if self._timesteps_total and not self._loaded:
            # in case of recovering from checkpoint
            self._num_agent_steps_sampled = self._counters[NUM_AGENT_STEPS_SAMPLED] = self._timesteps_total
            self.workers.sync_weights(global_vars={'timestep': self._timesteps_total})
        self._loaded = True
        _result = super().step()
        num_agent_steps_sampled = _result[NUM_AGENT_STEPS_SAMPLED]
        _result[result.TIMESTEPS_THIS_ITER] = num_agent_steps_sampled - self._num_agent_steps_sampled
        self._num_agent_steps_sampled = num_agent_steps_sampled
        return _result