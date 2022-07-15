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
                if np.sum(ties) > 1:  # tie break
                    lowest = np.min(highest_cards[ties])
                    idx = np.nonzero(highest_cards == lowest)[0]

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
        return self._cards > 0


class Deck:

    def __init__(self):
        self._deck = np.arange(1, 105)
        self._rand = np.random.default_rng()

    def distribute(self, num_players: int) -> Tuple[np.ndarray, List[Hand]]:
        self._rand.shuffle(self._deck)
        stacks, deck = self._deck[:4], self._deck[4:]
        hands = np.sort(self._rand.choice(deck, size=(num_players, 10), replace=False), axis=1)
        return stacks, [Hand(hands[i]) for i in range(num_players)]


class Take6(env.MultiAgentEnv):

    def __init__(self, table: Table, deck: Deck) -> None:
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Dict(
            {'action_mask': spaces.MultiBinary(10),
             'real_obs': spaces.Tuple((Hand.enc_space(), Table.enc_space()))})

        self._table = table
        self._deck = deck
        self._hands: List[Hand] = []
        self._history = np.zeros(104, dtype=int)

    def step(
            self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        played_cards = np.array([self._hands[i].select(a) for i, a in action_dict.items()])
        assert np.alltrue(played_cards >= 1)
        assert np.alltrue(played_cards <= 104)

        self._history[played_cards - 1] += 1
        assert np.alltrue(self._history < 2), 'Cards are being repeated {}'.format(self._history)

        round_scores = self._table.play(played_cards)
        _done = self._table.turn >= 10

        table_enc = self._table.encode()
        obs = {i: {'action_mask': self._hands[i].mask(), 'real_obs': (self._hands[i].encode(), table_enc)}
               for i in range(self._table.num_players)}
        rwd = {i: round_scores[i] for i in range(self._table.num_players)}
        done = {i: _done for i in range(self._table.num_players)}
        done['__all__'] = _done
        return obs, rwd, done, {}

    def reset(self) -> MultiAgentDict:
        stacks, self._hands = self._deck.distribute(self._table.num_players)
        self._history[:] = 0
        self._history[stacks - 1] += 1
        self._table.reset(stacks)
        table_enc = self._table.encode()
        return {i: {'action_mask': self._hands[i].mask(), 'real_obs': (self._hands[i].encode(), table_enc)}
                for i in range(self._table.num_players)}

    def render(self, mode=None) -> None:
        pass


def take6(config: Dict[str, Any]) -> gym.Env:
    return Take6(Table(config['num-players']), Deck())
