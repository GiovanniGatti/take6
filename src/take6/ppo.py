import os
import pathlib
import pickle
from collections import defaultdict
from typing import Optional, Union, Callable

import trueskill
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

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        saved = super().save_checkpoint(checkpoint_dir)
        if hasattr(self, 'ratings'):
            ratings_file = os.path.join(checkpoint_dir, 'ratings')
            pickle.dump(dict(self.ratings), open(ratings_file, 'wb'))
        return saved

    def load_checkpoint(self, checkpoint_path: str) -> None:
        super().load_checkpoint(checkpoint_path)
        parent = pathlib.Path(checkpoint_path).parent
        ratings_file = pathlib.Path(parent, 'ratings')
        original = pickle.load(open(ratings_file, 'rb'))
        trueskill.setup(mu=25., sigma=25. / 3, beta=20.8, tau=25. / 100, draw_probability=0.18)
        self.ratings = defaultdict(lambda: trueskill.Rating(), original)
