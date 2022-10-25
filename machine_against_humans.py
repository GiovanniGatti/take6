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
        inp = read_input_cards('Hand: ', num_expected_cards=10)
        my_hand = env.Hand(np.array(inp))

        inp = read_input_cards('Table: ', num_expected_cards=4)
        t = np.zeros((4, 5), dtype=np.int)
        t[:, 0] = inp
        table = env.Table.from_nparray(t)

        history = np.zeros(104, dtype=np.float32)
        history[np.array(inp) - 1] = 1

        scoreboard = env.Scoreboard()
        scoreboard.reset(num_players=num_players)

        while table.turn < 10:
            raw_obs = np.concatenate((
                my_hand.encode(), table.encode(), history, scoreboard.encode(), np.array([expert])))
            obs = {'action_mask': np.array(my_hand.mask(), dtype=np.float32).reshape(1, -1),
                   'real_obs': raw_obs.reshape(1, -1)}
            out, _ = policy.model({'obs': obs})
            a0 = out - tf.reduce_max(out, axis=1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, axis=1, keepdims=True)
            p0 = ea0 / z0
            probs = policy.get_session().run(p0).flatten()
            i = np.random.choice(probs.shape[0], p=probs)
            played_card = my_hand.select(i)

            if namespace.debugging:
                print('')
                print(f'Current scores: {-scoreboard.scores}')
                print(f'v = {policy.get_session().run(policy.model.value_function()).flatten()}')
                print('')
                input("Press Enter to continue...")
                print('')

            print_gr('Play card: {}'.format(played_card))

            if namespace.debugging:
                print('')
                print(f'probs = {np.round(probs, 2)}')
                print('')

            inp = read_input_cards('Opponents\' cards (clockwise): ', num_expected_cards=1)

            played_cards = np.concatenate((np.array([played_card]), inp))
            current_scores = table.play(played_cards)
            history[played_cards - 1] = 1

            print('')
            print('What I see:')
            for row in table.rows:
                print(str('{0:3d} {1:3d} {2:3d} {3:3d} {4:3d}'.format(*row)).replace(' 0', ' -'))

            scoreboard += current_scores

        print('')
        print(f'Final scores are {-scoreboard.scores if namespace.debugging else -scoreboard.scores[0]}')
        inp = input('\033[92m Another round (Y/n)?\033[00m ')
        if inp == 'n' or inp == 'N':
            sys.exit(0)
        print('')
        print('')
