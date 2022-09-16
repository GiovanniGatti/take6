import abc
import random
from typing import List, Union, Optional, Dict, Tuple

import gym
import numpy as np
from ray import rllib
from ray.rllib import SampleBatch
from ray.rllib.evaluation import Episode
from ray.rllib.models import ModelV2, ActionDistribution
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, ModelGradients, ModelWeights, TensorStructType


class _NotLearnablePolicy(rllib.Policy, abc.ABC):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        super().__init__(observation_space, action_space, config)

    @abc.abstractmethod
    def actions(self, obs_batch: Union[List[TensorStructType], TensorStructType]) -> np.ndarray:
        pass

    def compute_log_likelihoods(self,
                                actions: Union[List[TensorType], TensorType],
                                obs_batch: Union[List[TensorType], TensorType],
                                state_batches: Optional[List[TensorType]] = None,
                                prev_action_batch: Optional[Union[List[TensorType], TensorType]] = None,
                                prev_reward_batch: Optional[Union[List[TensorType], TensorType]] = None,
                                actions_normalized: bool = True) -> TensorType:
        pass

    def compute_actions(self,
                        obs_batch: Union[List[TensorStructType], TensorStructType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
                        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List[Episode]] = None,
                        explore: Optional[bool] = None,
                        timestep: Optional[int] = None,
                        **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        actions = self.actions(obs_batch)
        return actions, [], {}

    def loss(
            self, model: ModelV2, dist_class: ActionDistribution, train_batch: SampleBatch
    ) -> Union[TensorType, List[TensorType]]:
        pass

    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
        pass

    def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
        pass

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
        pass

    def compute_gradients(self, postprocessed_batch: SampleBatch) -> Tuple[ModelGradients, Dict[str, TensorType]]:
        return [], {}

    def apply_gradients(self, gradients: ModelGradients) -> None:
        pass

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        return {}

    def get_weights(self) -> ModelWeights:
        return {}

    def set_weights(self, weights: ModelWeights) -> None:
        pass

    def export_model(self, export_dir: str, onnx: Optional[int] = None) -> None:
        pass

    def export_checkpoint(self, export_dir: str) -> None:
        pass

    def import_model_from_h5(self, import_file: str) -> None:
        pass


class RandomPolicy(_NotLearnablePolicy):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        super().__init__(observation_space, action_space, config)
        self._rand = random.Random()

    def actions(self, obs_batch: Union[List[TensorStructType], TensorStructType]) -> np.ndarray:
        mask = obs_batch[:, :10]
        weights = np.zeros_like(mask)
        weights[mask == 0] = -np.inf
        probs = np.exp(weights) / np.sum(np.exp(weights), axis=1).reshape(weights.shape[0], 1)
        actions = []
        for _p in probs:
            actions.append(self._rand.choices(range(10), weights=_p)[0])
        return np.array(actions)
