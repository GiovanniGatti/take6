import abc
from typing import Dict, Any, Tuple, List

import gym
import numpy as np
from gym import spaces
from ray.rllib import env
from ray.rllib.utils.typing import MultiAgentDict


class Table:

    def __init__(self, num_players: int):
        self._num_players = num_players
        self._rows = np.zeros((4, 5), dtype=int)

        self._cattle_heads = np.ones(104, dtype=int)
        self._cattle_heads[4::5] += 1  # multiples of 5  : 2 cattle heads
        self._cattle_heads[9::10] += 1  # multiples of 10 : 3 cattle heads
        self._cattle_heads[10::11] += 4  # multiples of 11 : 5 cattle heads
        self._cattle_heads[54] += 1  # multiples of 55 : 7 cattle heads

        self._stack = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])

        self._turn = 0

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def rows(self) -> np.ndarray:
        return self._rows

    @property
    def turn(self) -> int:
        return self._turn

    @classmethod
    def enc_space(cls) -> spaces.Space:
        return spaces.Box(low=0, high=1, shape=(4 * 104,))

    def encode(self) -> np.ndarray:
        enc = np.zeros((4, 104), dtype=np.float32)
        flat = self._rows.flatten()
        mask = flat > 0
        enc[self._stack[mask], flat[mask] - 1] = 1
        return enc.flatten()

    def play(self, cards: np.ndarray) -> np.ndarray:
        scores = np.zeros(self._num_players, dtype=int)

        id_card = np.concatenate((np.arange(self._num_players).reshape((self._num_players, 1)),
                                  cards.reshape((self._num_players, 1))),
                                 axis=-1)

        for i, card in id_card[np.argsort(id_card[:, 1])]:
            highest_indexes = np.argmax(self._rows, axis=1)
            highest_cards = np.max(self._rows, axis=1)

            if card > np.min(highest_cards):
                valid_idx = np.where(card - highest_cards > 0)[0]
                target_stack = valid_idx[highest_cards[valid_idx].argmax()]
                idx = highest_indexes[target_stack] + 1
                if idx < 5:
                    self._rows[target_stack, idx] = card
                else:
                    scores[i] = -np.sum(self._cattle_heads[self._rows[target_stack] - 1])
                    self._rows[target_stack, :] = 0
                    self._rows[target_stack, 0] = card
            else:
                stack_scores = -np.sum(self._cattle_heads[self._rows - 1] * (self._rows > 0), axis=1)
                idx = np.argmax(stack_scores)

                ties = stack_scores == stack_scores[idx]
                if np.sum(ties) > 1:  # tie-break
                    highest = np.max(highest_cards[ties])
                    idx = np.nonzero(highest_cards == highest)[0]

                scores[i] = stack_scores[idx]
                self._rows[idx, :] = 0
                self._rows[idx, 0] = card

        self._turn += 1

        return scores

    def reset(self, cards: np.ndarray) -> None:
        assert cards.shape == (4,)
        self._rows[:] = 0
        self._rows[:, 0] = cards
        self._turn = 0

    @classmethod
    def from_nparray(cls, rows: np.ndarray, num_players: int) -> 'Table':
        _table = Table(num_players)
        assert rows.shape == _table._rows.shape
        _table._rows = rows
        return _table


class Hand:

    def __init__(self, cards: np.ndarray):
        self._cards = np.sort(cards)

    @property
    def cards(self) -> np.ndarray:
        return self._cards

    @classmethod
    def enc_space(cls) -> spaces.Space:
        return spaces.Box(low=0, high=1, shape=(104,))

    @classmethod
    def mask_enc_space(cls) -> spaces.Space:
        return spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def encode(self) -> np.ndarray:
        enc = np.zeros(104, dtype=np.float32)
        idx = self._cards[self._cards > 0] - 1
        enc[idx] = 1
        return enc

    def select(self, i: int) -> int:
        card = self._cards[i]
        self._cards[i] = 0
        self._cards = np.sort(self._cards)
        return card

    def mask(self) -> np.ndarray:
        return (self._cards > 0).astype(np.float32)


class Deck:

    def __init__(self):
        self._deck = np.arange(1, 105)
        self._rand = np.random.default_rng()

    def distribute(self, num_players: int) -> Tuple[np.ndarray, List[Hand]]:
        self._rand.shuffle(self._deck)
        stacks, deck = self._deck[:4], self._deck[4:]
        hands = np.sort(self._rand.choice(deck, size=(num_players, 10), replace=False), axis=1)
        return stacks, [Hand(hands[i]) for i in range(num_players)]


class Scoreboard:

    def __init__(self, num_players: int):
        self._num_players = num_players
        self._scores = np.zeros(self._num_players, dtype=np.float32)
        self._min_score = -66.

    @property
    def scores(self) -> np.ndarray:
        return self._scores

    def enc_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=1, shape=(self._num_players,))

    def encode(self) -> np.ndarray:
        return np.minimum(1., self._scores / self._min_score)

    def __add__(self, round_scores: np.ndarray) -> 'Scoreboard':
        self._scores += round_scores
        return self

    def reset(self) -> None:
        self._scores = np.zeros_like(self._scores)


class GroupedActionMultiEnv(env.MultiAgentEnv, abc.ABC):

    def __init__(self, num_players: int):
        super().__init__()
        self.num_players = num_players
        self._agent_ids = set(range(num_players))

    def observation_space_sample(self, agent_ids: List[Any] = None) -> MultiAgentDict:
        if agent_ids is None:
            agent_ids = list(range(self.num_players))
        obs = {agent_id: self.observation_space.sample() for agent_id in agent_ids}
        return obs

    def action_space_sample(self, agent_ids: List[Any] = None) -> MultiAgentDict:
        if agent_ids is None:
            agent_ids = list(range(self.num_players))
        actions = {agent_id: self.action_space.sample() for agent_id in agent_ids}
        return actions

    def action_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
            return False
        return all(self.action_space.contains(val) for val in x.values())

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
            return False
        return all(self.observation_space.contains(val) for val in x.values())


class Take6(GroupedActionMultiEnv):

    def __init__(self, table: Table, deck: Deck, scoreboard: Scoreboard) -> None:
        self._table = table
        self._deck = deck
        self._hands: List[Hand] = []
        self._history = np.zeros(104, dtype=int)

        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Dict(
            {'action_mask': Hand.mask_enc_space(),
             'real_obs': spaces.Tuple((Hand.enc_space(), Table.enc_space()))})

        self._scoreboard = scoreboard
        super().__init__(table.num_players)

    def step(
            self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        played_cards = np.array([self._hands[i].select(a) for i, a in action_dict.items()])
        assert np.alltrue(played_cards >= 1)
        assert np.alltrue(played_cards <= 104)

        self._history[played_cards - 1] += 1
        assert np.alltrue(self._history < 2), 'Cards are being repeated {}'.format(self._history)

        round_scores = self._table.play(played_cards)
        _done = self._table.turn >= 10

        self._scoreboard += round_scores

        table_enc = self._table.encode()
        obs = {i: {'action_mask': self._hands[i].mask(), 'real_obs': (self._hands[i].encode(), table_enc)}
               for i in range(self._table.num_players)}
        rwd = {i: round_scores[i] for i in range(self._table.num_players)}
        done = {i: _done for i in range(self._table.num_players)}
        done['__all__'] = _done
        return obs, rwd, done, {i: {'score': self._scoreboard.scores[i]} for i in range(self._table.num_players)}

    def reset(self) -> MultiAgentDict:
        stacks, self._hands = self._deck.distribute(self._table.num_players)
        self._history[:] = 0
        self._history[stacks - 1] += 1
        self._table.reset(stacks)
        table_enc = self._table.encode()
        self._scoreboard.reset()
        return {i: {'action_mask': self._hands[i].mask(),
                    'real_obs': (self._hands[i].encode(), table_enc)} for i in range(self._table.num_players)}

    def render(self, mode=None) -> None:
        pass


class ClassificationRwd(GroupedActionMultiEnv):

    def __init__(self, _env: env.MultiAgentEnv, scoreboard: Scoreboard, num_players: int):
        super().__init__(num_players)
        self._env = _env
        self._num_players = num_players
        self._scoreboard = scoreboard
        self.observation_space = _env.observation_space
        self.action_space = _env.action_space

    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs, _, done, info = self._env.step(action_dict)

        if done[0]:
            scores = self._scoreboard.scores
            classification = np.zeros(scores.shape[0], dtype=np.int)
            classification[np.argsort(scores)] = np.arange(4)[::-1]
            s, c = np.unique(scores, return_counts=True)
            for _s in s[c > 1]:  # handle ties
                classification[scores == _s] = np.min(classification[scores == _s])
            rwd = {i: -classification[i] for i in range(self._num_players)}
        else:
            rwd = {i: 0 for i in range(self._num_players)}

        return obs, rwd, done, info

    def reset(self) -> MultiAgentDict:
        return self._env.reset()

    def render(self, mode=None) -> None:
        pass


class ProportionalRwd(GroupedActionMultiEnv):

    def __init__(self, _env: env.MultiAgentEnv, scoreboard: Scoreboard, num_players: int):
        super().__init__(num_players)
        self._env = _env
        self._num_players = num_players
        self._scoreboard = scoreboard
        self.observation_space = _env.observation_space
        self.action_space = _env.action_space

    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs, rwd, done, info = self._env.step(action_dict)

        if done[0]:
            scores = self._scoreboard.scores
            new_rwd = {i: -scores[i] / np.sum(scores) for i in range(self._num_players)}
        else:
            new_rwd = {i: 0. for i in range(self._num_players)}

        return obs, new_rwd, done, info

    def reset(self) -> MultiAgentDict:
        return self._env.reset()

    def render(self, mode=None) -> None:
        pass


class ScoreWrapper(GroupedActionMultiEnv):

    def __init__(self, _env: GroupedActionMultiEnv, scoreboard: Scoreboard):
        assert isinstance(_env.observation_space, spaces.Dict), 'Original environment must have a Dict obs. space'
        assert 'real_obs' in _env.observation_space.keys(), 'Original environment must have an real_obs space'
        super().__init__(_env.num_players)

        self._env = _env
        self._scoreboard = scoreboard
        self.action_space = _env.action_space
        self.observation_space = _env.observation_space

        self.observation_space['real_obs'] = spaces.Tuple((*self.observation_space['real_obs'], scoreboard.enc_space()))

    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs, rwd, done, info = self._env.step(action_dict)
        return self.observation(obs), rwd, done, info

    def reset(self) -> MultiAgentDict:
        return self.observation(self._env.reset())

    def observation(self, obs: MultiAgentDict) -> MultiAgentDict:
        enc_score = self._scoreboard.encode()
        for _obs in obs.values():
            _obs['real_obs'] = (*_obs['real_obs'], enc_score)
        return obs

    def render(self, mode=None) -> None:
        pass


# noinspection PyMissingConstructor
class PlayedCardsWrapper(GroupedActionMultiEnv):

    def __init__(self, _env: GroupedActionMultiEnv, table: Table):
        assert isinstance(_env.observation_space, spaces.Dict), 'Original environment must have a Dict obs. space'
        assert 'real_obs' in _env.observation_space.keys(), 'Original environment must have an real_obs space'
        super().__init__(_env.num_players)

        self.observation_space = _env.observation_space
        self.action_space = _env.action_space
        self._env = _env
        self._table = table

        self.observation_space['real_obs'] = spaces.Tuple((*self.observation_space['real_obs'],
                                                           spaces.Box(low=0, high=1, shape=(104,))))

        self._history = np.zeros(104, dtype=np.float32)

    def step(
            self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs, rwd, done, info = self._env.step(action_dict)
        self._history[self._table.rows.flatten() - 1] = 1
        for _obs in obs.values():
            _obs['real_obs'] = (*_obs['real_obs'], self._history)
        return obs, rwd, done, info

    def reset(self) -> MultiAgentDict:
        obs = self._env.reset()
        self._history[:] = 0
        self._history[self._table.rows.flatten() - 1] = 1
        for _obs in obs.values():
            _obs['real_obs'] = (*_obs['real_obs'], self._history)
        return obs

    def render(self, mode=None) -> None:
        pass


def take6(config: Dict[str, Any]) -> gym.Env:
    num_players = config['num-players']
    scoreboard = Scoreboard(num_players)
    table = Table(num_players)

    _env = Take6(table, Deck(), scoreboard)

    rwd_fn = config['rwd-fn']
    if rwd_fn == 'proportional-score':
        _env = ProportionalRwd(_env, scoreboard, num_players)
    elif rwd_fn == 'classification':
        _env = ClassificationRwd(_env, scoreboard, num_players)

    if config['with-history']:
        _env = PlayedCardsWrapper(_env, table)

    if config['with-scores']:
        _env = ScoreWrapper(_env, scoreboard)

    return _env
