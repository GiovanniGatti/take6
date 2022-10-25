import argparse
import os
import pickle
import sys
import readline
from typing import List

import inquirer
import numpy as np
from ray import tune
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOTF1Policy, PPO
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from take6 import env, model

tf1, tf, tfv = try_import_tf()


def print_gr(skk: str) -> None: print('\033[92m {}\033[00m'.format(skk))


def read_input_cards(input_text: str, num_expected_cards: int) -> List[int]:
    while True:
        inp = input(input_text)
        try:
            inp = [int(i) for i in filter(lambda s: s != '', inp.strip().split(' '))]
        except ValueError:
            print('Expecting {} cards separated by spaces, ex.: 10 14 55 12 16 103 etc.'.format(num_expected_cards))
            continue
        if len(inp) != num_expected_cards:
            print('Expecting {} cards, counted {}'.format(num_expected_cards, len(inp)))
            continue
        if set(filter(lambda x: x < 1 or x > 104, inp)):
            print('Invalid cards, expected between 1 and 104')
            continue
        if len(set(inp)) != num_expected_cards:
            print('Repeated cards')
            continue
        break
    return inp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debugging', action='store_true', help='Run locally with simplified settings')

    namespace = parser.parse_args(sys.argv[1:])

    questions = [
        inquirer.List('size',
                      message='Select #players and rules',
                      choices=['2 (expert)', '4 (standard)', '5 (standard)', ],
                      ),
    ]
    answers = inquirer.prompt(questions)

    project_root = os.path.dirname(os.path.dirname(__file__))
    trained_anns_path = os.path.join(project_root, 'trained-anns')

    if answers['size'] == '2 (expert)':
        num_players = 2
        expert = True
        run_id = 'take6-train_1666602477_8fffa276'
        model_path = os.path.join(trained_anns_path, '2-players-expert')
    elif answers['size'] == '4 (standard)':
        num_players = 4
        expert = False
        run_id = 'take6-train_1666596986_3fa6f5cc'
        model_path = os.path.join(trained_anns_path, '4-players-standard')
    else:
        num_players = 5
        expert = False
        run_id = 'take6-train_1666339821_634f7946'
        model_path = os.path.join(trained_anns_path, '5-players-standard')

    original_config = {
        'env': 'take6',
        'env_config': {
            'num-players': num_players,
            'expert': expert,
            'rwd-fn': 'proportional-rwd',
            'with-scores': True,
            'with-history': True,
        },
        'rollout_fragment_length': 10,
        'framework': 'tf',

        'model': {
            'fcnet_hiddens': [256, 256, 256, ],
            'custom_model': 'take6',
            'fcnet_activation': 'relu',
        },
        '_disable_preprocessor_api': True,
    }

    trainer_config = Algorithm.merge_trainer_configs(PPO.get_default_config(), original_config)
    ModelCatalog.register_custom_model('take6', model.Take6Model)
    tune.register_env('take6', env.take6)

    e = env.take6(trainer_config['env_config'])
    policy = PPOTF1Policy(e.observation_space, e.action_space, trainer_config)
    policy.set_state(pickle.load(open(model_path, 'rb')))

    print('')
    print('--- --- --- --- ---')
    print('')

    while True:
        hands = {}
        for i in range(num_players - 1):
            inp = read_input_cards('Hand for player #{}: '.format(i + 1), num_expected_cards=10)
            hands[i] = env.Hand(np.array(inp))

        print('')
        inp = read_input_cards('Table: ', num_expected_cards=4)
        t = np.zeros((4, 5), dtype=np.int)
        t[:, 0] = inp
        table = env.Table.from_nparray(t)

        history = np.zeros(104, dtype=np.float32)
        history[np.array(inp) - 1] = 1

        scoreboard = env.Scoreboard()
        scoreboard.reset(num_players=num_players)

        indexes = np.arange(env.Scoreboard.enc_space().shape[0])

        while table.turn < 10:
            selected_cards = []
            enc_score = scoreboard.encode()
            enc_table = table.encode()
            for i in range(num_players - 1):
                enc = np.concatenate((np.expand_dims(enc_score[i], axis=0), enc_score[indexes != i]))
                raw_obs = np.concatenate((
                    hands[i].encode(), enc_table, history, enc, np.array([expert])))
                obs = {'action_mask': np.array(hands[i].mask(), dtype=np.float32).reshape(1, -1),
                       'real_obs': raw_obs.reshape(1, -1)}
                out, _ = policy.model({'obs': obs})
                j = policy.get_session().run(tf.squeeze(tf.random.categorical(out, 1)))
                played_card = hands[i].select(j)
                selected_cards += [played_card, ]

            print('')
            input('Press Enter to continue...')
            print('')
            print_gr('Play cards: {}'.format(selected_cards))

            inp = read_input_cards('Opponent\'s card: ', num_expected_cards=1)

            played_cards = np.concatenate((selected_cards, inp))
            current_scores = table.play(played_cards)
            history[played_cards - 1] = 1

            print('')
            print('What I see:')
            for row in table.rows:
                print(str('{0:3d} {1:3d} {2:3d} {3:3d} {4:3d}'.format(*row)).replace(' 0', ' -'))

            scoreboard += current_scores

        print('')
        print('Machine scores were {}'.format(-scoreboard.scores[:num_players - 1]))
        print('Player scores was {}'.format(-scoreboard.scores[num_players - 1]))
        inp = input('\033[92m Another round (Y/n)?\033[00m ')
        if inp == 'n' or inp == 'N':
            sys.exit(0)
        print('')
        print('')
