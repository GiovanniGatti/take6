import random
import unittest
from typing import List

import numpy as np
import pytest

from take6 import env


class TestTable:

    @pytest.mark.parametrize(
        'table, played_card, expected, expected_score',
        [
            # place at highest feasible stack
            ([[2, 0, 0, 0, 0],
              [7, 0, 0, 0, 0],
              [8, 0, 0, 0, 0],
              [15, 0, 0, 0, 0]],
             14,
             [[2, 0, 0, 0, 0],
              [7, 0, 0, 0, 0],
              [8, 14, 0, 0, 0],
              [15, 0, 0, 0, 0]],
             0),

            # take all points and clean stack
            ([[2, 0, 0, 0, 0],
              [7, 0, 0, 0, 0],
              [8, 9, 10, 11, 12],
              [15, 0, 0, 0, 0]],
             14,
             [[2, 0, 0, 0, 0],
              [7, 0, 0, 0, 0],
              [14, 0, 0, 0, 0],
              [15, 0, 0, 0, 0]],
             -11),

            # clean smallest
            ([[2, 0, 0, 0, 0],
              [7, 0, 0, 0, 0],
              [8, 0, 0, 0, 0],
              [15, 0, 0, 0, 0]],
             1,
             [[1, 0, 0, 0, 0],
              [7, 0, 0, 0, 0],
              [8, 0, 0, 0, 0],
              [15, 0, 0, 0, 0]],
             -1),

            # clean smallest 2 (situation might not be possible)
            ([[2, 19, 0, 0, 0],
              [7, 18, 0, 0, 0],
              [8, 20, 0, 0, 0],
              [15, 11, 0, 0, 0]],
             1,
             [[2, 19, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [8, 20, 0, 0, 0],
              [15, 11, 0, 0, 0]],
             -2),

            # clean smallest number of points
            ([[2, 5, 7, 10, 0],
              [7, 11, 0, 0, 0],
              [8, 22, 0, 0, 0],
              [15, 16, 0, 0, 0]],
             1,
             [[2, 5, 7, 10, 0],
              [7, 11, 0, 0, 0],
              [8, 22, 0, 0, 0],
              [1, 0, 0, 0, 0]],
             -3),

            # place in further position
            ([[2, 5, 7, 10, 0],
              [7, 11, 17, 21, 0],
              [8, 22, 35, 0, 0],
              [15, 16, 75, 90, 0]],
             80,
             [[2, 5, 7, 10, 0],
              [7, 11, 17, 21, 0],
              [8, 22, 35, 80, 0],
              [15, 16, 75, 90, 0]],
             0),

        ])
    def test_card_placement(
            self, table: List[List[int]], played_card: int, expected: List[List[int]], expected_score: int
    ) -> None:
        t = env.Table.from_nparray(np.array(table), num_players=1)
        score = t.play(np.array([played_card]))
        assert np.array_equal(t.rows, np.array(expected))
        assert score == expected_score

    @pytest.mark.parametrize(
        'table, played_cards, expected, expected_score',
        [
            # nobody takes
            ([[10, 0, 0, 0, 0],
              [25, 27, 0, 0, 0],
              [55, 61, 66, 68, 0],
              [60, 73, 82, 0, 0]],
             [29, 83, 69, 26],
             [[10, 26, 0, 0, 0],
              [25, 27, 29, 0, 0],
              [55, 61, 66, 68, 69],
              [60, 73, 82, 83, 0]],
             [0, 0, 0, 0]),

            # only one player takes
            ([[10, 0, 0, 0, 0],
              [25, 27, 0, 0, 0],
              [55, 61, 66, 68, 0],
              [60, 73, 82, 0, 0]],
             [29, 83, 70, 69],
             [[10, 0, 0, 0, 0],
              [25, 27, 29, 0, 0],
              [70, 0, 0, 0, 0],
              [60, 73, 82, 83, 0]],
             [0, 0, -15, 0]),

            # every player plays on the same stack
            ([[10, 0, 0, 0, 0],
              [25, 27, 0, 0, 0],
              [55, 61, 66, 68, 0],
              [60, 73, 82, 0, 0]],
             [28, 31, 29, 30],
             [[10, 0, 0, 0, 0],
              [31, 0, 0, 0, 0],
              [55, 61, 66, 68, 0],
              [60, 73, 82, 0, 0]],
             [0, -8, 0, 0]),

            # clean before others
            ([[10, 0, 0, 0, 0],
              [25, 27, 0, 0, 0],
              [55, 61, 66, 68, 0],
              [60, 73, 82, 0, 0]],
             [9, 69, 70, 11],
             [[9, 11, 0, 0, 0],
              [25, 27, 0, 0, 0],
              [70, 0, 0, 0, 0],
              [60, 73, 82, 0, 0]],
             [-3, 0, -15, 0]),
        ])
    def test_play_ordering(
            self, table: List[List[int]], played_cards: List[int], expected: List[List[int]], expected_score: List[int]
    ) -> None:
        t = env.Table.from_nparray(np.array(table), num_players=4)
        score = t.play(np.array(played_cards))
        assert np.array_equal(t.rows, np.array(expected))
        assert np.array_equal(score, expected_score)

    def test_reset(self) -> None:
        t = env.Table(num_players=3)
        t.reset(np.array([10, 7, 55, 12]))
        t.play(np.array([13, 80, 99]))
        assert t.turn == 1

        t.reset(np.array([13, 9, 22, 55]))
        assert np.array_equal(t.rows, np.array([[13, 0, 0, 0, 0],
                                                [9, 0, 0, 0, 0],
                                                [22, 0, 0, 0, 0],
                                                [55, 0, 0, 0, 0]]))
        assert t.turn == 0

    def test_encode(self) -> None:
        t = env.Table.from_nparray(np.array([[13, 15, 0, 0, 0],
                                             [9, 0, 0, 0, 0],
                                             [22, 83, 101, 0, 0],
                                             [55, 75, 104, 0, 0]]), num_players=random.randint(2, 11))

        enc = t.encode()

        expected = np.zeros((4, 104), dtype=np.float32)
        stack = np.array([0, 0, 1, 2, 2, 2, 3, 3, 3])
        cards = np.array([13, 15, 9, 22, 83, 101, 55, 75, 104])
        expected[stack, cards - 1] = 1

        assert env.Table.enc_space().contains(enc)
        assert np.array_equal(enc, expected.flatten())


class TestDeck:

    def test_distribute_all_cards(self) -> None:
        deck = env.Deck()

        stacks, hands = deck.distribute(num_players=10)

        assert stacks.shape == (4,)
        assert len(hands) == 10
        assert hands[random.randint(0, 10)].cards.shape == (10,)
        all_cards = np.array([hands[i].cards for i in range(10)])
        assert np.alltrue(np.sort(np.concatenate((stacks, all_cards.flatten()))) == np.arange(1, 105))

    def test_cards_are_not_duplicated(self) -> None:
        deck = env.Deck()

        stacks, hands = deck.distribute(num_players=3)

        assert stacks.shape == (4,)
        assert len(hands) == 3
        assert hands[random.randint(0, 3)].cards.shape == (10,)
        all_cards = np.array([hands[i].cards for i in range(3)])
        assert np.alltrue(np.unique(np.concatenate((stacks, all_cards.flatten())), return_counts=True)[1] == 1)


class TestHand:

    def test_initialized_cards_are_sorted(self) -> None:
        cards = np.arange(1, 11)
        np.random.shuffle(cards)

        hand = env.Hand(cards)

        assert np.array_equal(hand.cards, np.arange(1, 11))
        assert np.alltrue(hand.mask() == np.ones(10))

    def test_cards_are_sorted_after_selection(self) -> None:
        hand = env.Hand(cards=np.arange(55, 66))

        card = hand.select(3)

        assert card == 58
        assert np.array_equal(hand.cards, np.array([0, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65]))
        assert np.alltrue(hand.mask() == [False, True, True, True, True, True, True, True, True, True, True])

    def test_encode(self) -> None:
        h = env.Hand(cards=np.array([0, 0, 1, 3, 5, 7, 55, 83, 103, 104]))

        enc = h.encode()

        expected = np.zeros(104, dtype=np.float32)
        cards = np.array([1, 3, 5, 7, 55, 83, 103, 104])
        expected[cards - 1] = 1

        assert np.array_equal(enc, expected)
        assert env.Hand.enc_space().contains(enc)


if __name__ == '__main__':
    unittest.main()
