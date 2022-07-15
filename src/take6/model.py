from typing import Dict, List, Tuple

import gym
import numpy as np
from gym import spaces
from ray.rllib.models.tf import tf_modelv2, fcnet
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.typing import ModelConfigDict

_, tf, _ = try_import_tf()


class Take6Model(tf_modelv2.TFModelV2):

    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str):
        original_space = obs_space.original_space if hasattr(obs_space, 'original_space') else obs_space
        real_obs_space = original_space['real_obs']

        flat_space_shape = 0
        assert isinstance(real_obs_space, spaces.Tuple)
        for s in real_obs_space.spaces:
            assert isinstance(s, spaces.Box)
            assert np.alltrue(s.low == 0.) and np.alltrue(s.high == 1.)
            flat_space_shape += s.shape[0]

        flat_space = spaces.Box(low=0, high=1, shape=(flat_space_shape,))

        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self._action_embed_model = fcnet.FullyConnectedNetwork(
            flat_space, action_space, num_outputs, model_config, name)

    def forward(
            self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:
        action_mask = input_dict['obs']['action_mask']
        action_embed, _ = self._action_embed_model({
            'obs': tf.concat(input_dict['obs']['real_obs'], axis=1)
        })
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_embed + inf_mask, state

    def value_function(self) -> TensorType:
        return self._action_embed_model.value_function()

    def import_from_h5(self, h5_file: str) -> None:
        self._action_embed_model.import_from_h5(h5_file)
