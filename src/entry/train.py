import argparse
import json
import math
import multiprocessing
import pathlib
import random
import sys
import tempfile
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import ray
import trueskill
from azureml import core
from ray import tune, rllib
from ray.rllib.algorithms import algorithm, callbacks
from ray.rllib.evaluation import worker_set
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import Dict
from ray.tune.analysis import experiment_analysis
from ray.tune.utils import log
from statsmodels.stats import weightstats

from take6 import env, model, policy, ppo
from take6.aztools import checkpoint

_, tf, _ = try_import_tf()


def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
        return True
    elif 'FALSE'.startswith(ua):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f'Expecting either true or false, but found {arg}. If you want None, simply don\'t specify the parameter.')


def policy_mapping_fn(agent_id, episode, worker, **kwargs) -> str:
    if agent_id == 0:
        return 'learner'
    policies_to_sample = list(sorted(filter(lambda k: k not in ('learner', 'random'), worker.policy_map.keys()),
                                     key=lambda k: int(k.replace('opponent_v', '')), reverse=True))
    policies_to_sample = policies_to_sample[:POLICY_BUFFER_SIZE]
    return random.choice(policies_to_sample)


class TrackingCallback(callbacks.DefaultCallbacks):

    def on_train_result(self, result: Dict[Any, Any], **kwargs) -> None:
        _run = core.Run.get_context()
        # custom metrics
        _run.log(name='training/win_rate', value=result['win_rate'])
        _run.log(name='training/score_rate', value=result['score_rate'])
        _run.log(name='training/relative_score', value=result['relative_score'])
        _run.log(name='training/weighted_classification', value=result['weighted_classification'])
        _run.log(name='training/policy_reward_mean', value=result['policy_reward_mean']['learner'])
        _run.log(name='training/league_size', value=result['league_size'])

        # learning stats
        learner = result['info']['learner']['learner']['learner_stats']
        _run.log(name='learner_stats/entropy', value=learner['entropy'])
        _run.log(name='learner_stats/entropy_coeff', value=learner['entropy_coeff'])
        _run.log(name='learner_stats/kl', value=learner['kl'])
        _run.log(name='learner_stats/cur_kl_coeff', value=learner['cur_kl_coeff'])
        _run.log(name='learner_stats/policy_loss', value=learner['policy_loss'])
        _run.log(name='learner_stats/vf_loss', value=learner['vf_loss'])
        _run.log(name='learner_stats/vf_explained_var', value=learner['vf_explained_var'])
        _run.log(name='learner_stats/total_loss', value=learner['total_loss'])
        _run.log(name='learner_stats/curr_lr', value=learner['cur_lr'])

        # perf
        perf = result['perf']
        sampler_perf = result['sampler_perf']
        _run.log(name='perf/mean_env_wait_ms', value=sampler_perf['mean_env_wait_ms'])
        _run.log(name='perf/mean_raw_obs_processing_ms', value=sampler_perf['mean_raw_obs_processing_ms'])
        _run.log(name='perf/mean_inference_ms', value=sampler_perf['mean_inference_ms'])
        _run.log(name='perf/mean_action_processing_ms', value=sampler_perf['mean_action_processing_ms'])
        _run.log(name='perf/cpu_util_percent', value=perf['cpu_util_percent'])
        _run.log(name='perf/ram_util_percent', value=perf['ram_util_percent'])
        if 'gpu_util_percent0' in perf:
            _run.log(name='perf/gpu_util_percent0', value=perf['gpu_util_percent0'])
            _run.log(name='perf/vram_util_percent0', value=perf['vram_util_percent0'])

        timers = result['timers']
        _run.log(name='timers/training_iteration_time_ms', value=timers['training_iteration_time_ms'])
        _run.log(name='timers/learn_time_ms', value=timers['learn_time_ms'])
        _run.log(name='timers/learn_throughput', value=timers['learn_throughput'])
        if 'synch_weights_time_ms' in timers:
            _run.log(name='timers/synch_weights_time_ms', value=timers['synch_weights_time_ms'])

        # progress
        _run.log(name='progress/timesteps_total', value=result['timesteps_total'])
        _run.log(name='progress/time_this_iter_s', value=result['time_this_iter_s'])
        _run.log(name='progress/time_total_s', value=result['time_total_s'])
        _run.log(name='progress/episodes_total', value=result["episodes_total"])

        if 'evaluation' in result and 'custom_metrics' in result['evaluation']:
            _run.log(name='eval/policy_reward_mean', value=result['evaluation']['policy_reward_mean']['learner'])
            _run.log(name='eval/score_rate', value=result['evaluation']['custom_metrics']['eval_score_rate'])
            _run.log(name='eval/relative_score', value=result['evaluation']['custom_metrics']['eval_relative_score'])
            _run.log(name='eval/weighted_classification',
                     value=result['evaluation']['custom_metrics']['eval_weighted_classification'])
            _run.log(name='eval/win_rate', value=result['evaluation']['custom_metrics']['eval_win_rate'])
            _run.log(name='eval/ties_rate', value=result['evaluation']['custom_metrics']['eval_ties'])
            _run.log(name='eval/relative_score_to_most_recent_policy',
                     value=result['evaluation']['custom_metrics']['relative_score_to_most_recent_policy'])

            ratings = result['evaluation']['custom_metrics']['trueskill']

            print('=== training iteration {} ==='.format(result['training_iteration']))
            print(json.dumps({k: 'Rating(mu={:.3f}, sigma={:.3f}) / mmr={:.1f}'
                             .format(r.mu, r.sigma, r.mu - 3 * r.sigma) for k, r in ratings.items()},
                             indent=2, sort_keys=True))

            _learner = ratings['learner']
            _random = ratings['random']
            _opponent_mmr = _random.mu - 3 * _random.sigma
            _learner_mmr = _learner.mu - 3 * _learner.sigma
            _run.log(name='eval/mu', value=_learner.mu)
            _run.log(name='eval/sigma', value=_learner.sigma)
            _run.log(name='eval/mmr', value=_learner_mmr)
            _run.log(name='eval/quality', value=trueskill.quality_1vs1(_learner, _random))
            _run.log(name='eval/random_mu', value=_random.mu)
            _run.log(name='eval/random_sigma', value=_random.sigma)
            _run.log(name='eval/random_mmr', value=_opponent_mmr)
            _run.log(name='eval/relative_mmr', value=_learner_mmr - _opponent_mmr)


class CustomizedMetricsCallback(callbacks.DefaultCallbacks):

    def on_episode_end(self, *,
                       worker: rllib.RolloutWorker,
                       base_env: rllib.BaseEnv,
                       policies: Dict[rllib.utils.typing.PolicyID, rllib.Policy],
                       episode: rllib.evaluation.episode.MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs: Dict[Any, Any]) -> None:
        super().on_episode_end(
            worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs)
        scores = []
        agents = []
        ids = []
        for _id, t in episode.agent_rewards.keys():
            ids.append(_id)
            agents.append(t)
            scores.append(episode.last_info_for(_id)['score'])

        # This is a misuse of hist_data, but I could not find another way to send data to tournament method
        episode.hist_data['true_score'] = scores

        if 'learner' in agents:
            num_players = len(agents)
            classification = np.zeros(num_players, dtype=np.int)
            classification[np.argsort(scores)[::-1]] = np.arange(num_players)[::-1]
            s, c = np.unique(scores, return_counts=True)
            for _s in s[c > 1]:  # handle ties
                classification[scores == _s] = np.min(classification[scores == _s])

            idx = agents.index('learner')
            episode.custom_metrics['win'] = 1 if np.argmin(scores) == idx else 0
            episode.custom_metrics['score'] = scores[idx]
            episode.custom_metrics['relative_score'] = scores[idx] / np.sum(scores)
            episode.custom_metrics['weighted_classification'] = -np.arange(len(scores))[classification[idx]]

    def on_train_result(self, result, **kwargs):
        trainer = kwargs['trainer']
        result['win_rate'] = result['custom_metrics']['win_mean']
        result['score_rate'] = result['custom_metrics']['score_mean']
        result['relative_score'] = result['custom_metrics']['relative_score_mean']
        result['weighted_classification'] = result['custom_metrics']['weighted_classification_mean']

        current_policies = list(trainer.workers.local_worker().policy_map.keys())
        result['league_size'] = len(current_policies) - 2


def tournament(_algorithm: algorithm.Algorithm, eval_workers: worker_set.WorkerSet) -> Dict[Any, Any]:
    if not hasattr(_algorithm, 'ratings'):
        # see https://trueskill.info/help.html
        trueskill.setup(mu=25., sigma=25. / 3, beta=20.8, tau=25. / 100, draw_probability=.5)
        _algorithm.ratings = defaultdict(lambda: trueskill.Rating())

    assert _algorithm.config['evaluation_duration_unit'] == 'episodes'

    worker_m, workers_mmr = eval_workers.remote_workers()[0], eval_workers.remote_workers()[1:]

    def policy_mapping_fn(agent_id, episode, worker, **kwargs) -> str:
        if agent_id == 0:
            return 'learner'
        return next(iter(sorted(filter(lambda k: k not in ('learner', 'random'), worker.policy_map.keys()),
                                key=lambda k: int(k.replace('opponent_v', '')), reverse=True)))

    ray.get([worker_m.apply.remote(lambda w: w.set_policy_mapping_fn(policy_mapping_fn))])

    # metrics becomes more and more noisy as the number of policies in the pool grows
    # for this reason, we are limiting the number of policies available for evaluation
    # at each training iteration
    selected_policies = list(filter(lambda k: k not in ('learner', 'random'),
                                    _algorithm.workers.local_worker().policy_map.keys()))
    selected_policies = list(np.random.choice(selected_policies, size=min(8, len(selected_policies)), replace=False))
    selected_policies.extend(('learner', 'random'))

    def mmr_policy_mapping_fn(agent_id, episode, worker, **kwargs) -> str:
        return random.choice(selected_policies)

    for w in workers_mmr:
        ray.get([w.apply.remote(lambda _w: _w.set_policy_mapping_fn(mmr_policy_mapping_fn))])

    evaluation_duration = _algorithm.config['evaluation_duration']
    total_ep = 0
    while True:
        batch = eval_workers.foreach_worker(lambda w: w.sample())
        total_ep += batch[0].env_steps() / 10
        if total_ep >= evaluation_duration:
            break

    #
    # Deciding whether to save the current learner's policy in pool of tournament policies
    #
    episodes, _ = collect_episodes(remote_workers=[worker_m])
    result = summarize_episodes(episodes, keep_custom_metrics=True)

    env_config = _algorithm.config['env_config']
    num_players: Optional[int] = env_config['num-players']
    # Expected score in a balanced game
    if num_players:
        target_score = 0.95 * (1 / num_players)
    else:
        target_score = 0.95 * np.mean(1 / np.arange(11)[2:])

    current_policies = list(_algorithm.workers.local_worker().policy_map.keys())
    tournament_policies = [p for p in current_policies if p not in ('learner', 'random')]

    # taking a conservative measure of performance
    relative_scores = weightstats.DescrStatsW(result['custom_metrics']['relative_score'])
    relative_score = max(relative_scores.tconfint_mean(alpha=0.02))

    if namespace.with_self_play and relative_score < target_score:
        latest_opponent = max(int(_id.replace('opponent_v', '')) for _id in tournament_policies) \
            if tournament_policies else 0
        new_pol_id = f'opponent_v{latest_opponent + 1}'
        new_policy = _algorithm.add_policy(
            policy_id=new_pol_id,
            policy_cls=type(_algorithm.get_policy('learner')),
        )
        main_state = _algorithm.get_policy('learner').get_state()
        new_policy.set_state(main_state)

        _algorithm.workers.sync_weights()

    #
    # Computing MMR
    #
    episodes, _ = collect_episodes(remote_workers=workers_mmr)
    ties = 0
    for episode in episodes:
        scores = []
        players = []

        for idx, t in episode.agent_rewards.keys():
            scores.append(episode.hist_data['true_score'][idx])
            players.append(t)

        suffixed_players = list(players)
        u_players, counts = np.unique(players, return_counts=True)
        for prefix, n in zip(u_players[counts > 1], counts[counts > 1]):
            # need to suffix duplicates
            idx = 0
            for i in range(n - 1):
                idx = players.index(prefix, idx)
                player_id = prefix + '_{}'.format(i)
                suffixed_players[idx] = player_id
                idx += 1

        ratings = [(_algorithm.ratings[p],) for p in suffixed_players]

        num_players = len(players)
        classification = np.zeros(num_players, dtype=int)
        classification[np.argsort(scores)[::-1]] = np.arange(num_players, dtype=int)[::-1]
        s, c = np.unique(scores, return_counts=True)
        ties += np.sum(c > 1)
        for _s in s[c > 1]:  # handle ties
            classification[scores == _s] = np.min(classification[scores == _s])
        new_ratings = trueskill.rate(ratings, ranks=list(classification))

        assert np.unique(suffixed_players).shape[0] == num_players, \
            'Repeated versions of a policy are playing within the same match'

        for i, p in enumerate(suffixed_players):
            (_algorithm.ratings[p],) = new_ratings[i]

    metrics = summarize_episodes(episodes, keep_custom_metrics=True)

    custom_metrics = metrics['custom_metrics']
    custom_metrics['trueskill'] = _algorithm.ratings
    custom_metrics['eval_win_rate'] = np.mean(custom_metrics['win'])
    custom_metrics['eval_score_rate'] = np.mean(custom_metrics['score'])
    custom_metrics['eval_relative_score'] = np.mean(custom_metrics['relative_score'])
    custom_metrics['eval_weighted_classification'] = np.mean(custom_metrics['weighted_classification'])
    custom_metrics['eval_ties'] = ties / len(episodes)

    custom_metrics['relative_score_to_most_recent_policy'] = relative_score

    return metrics


def main(_namespace: argparse.Namespace, _tmp_dir: str) -> experiment_analysis.ExperimentAnalysis:
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    num_cpus = multiprocessing.cpu_count()

    # 1 worker dedicated to learner evaluation and
    # 1 produces that produces num_envs_per_worker episodes for mmr computation
    num_eval_workers = 2

    if _namespace.debugging:
        num_workers = 1
        num_envs_per_worker = 1
        sgd_minibatch_size = 10
        num_sgd_iter = 3
        framework = 'tf2'
        evaluation_duration = 10
        local_dir = str(pathlib.Path(_tmp_dir, 'ray-results'))
    else:
        num_workers = num_cpus - num_eval_workers - 1
        num_envs_per_worker = int(math.ceil(_namespace.batch_size / (num_workers * 10)))
        sgd_minibatch_size = _namespace.minibatch_size
        num_sgd_iter = _namespace.num_sgd_iter
        evaluation_duration = 3000  # approximate number of tournament matches
        framework = 'tf'
        local_dir = './logs/ray-results'

    train_batch_size = num_workers * num_envs_per_worker * 10

    _checkpoint = checkpoint.recover_from_preemption(local_dir)
    if _checkpoint is None and _namespace.checkpoint is not None:
        checkpoint.retrieve_metrics(_namespace.run_id, _namespace.checkpoint)
        _checkpoint = checkpoint.load_checkpoint_from(_namespace.run_id, _namespace.checkpoint)
        print('Using checkpoint file \'{}\' from run-id \'{}\''.format(_checkpoint, _namespace.run_id))
    elif _checkpoint is not None:
        print('Using checkpoint file \'{}\' before preemption'.format(_checkpoint))

    ModelCatalog.register_custom_model('take6', model.Take6Model)
    tune.register_env('take6', env.take6)

    entropy_coeff_schedule = None
    if len(_namespace.entropy_coeff) == 1:
        entropy_coeff = _namespace.entropy_coeff[0]
    elif len(_namespace.entropy_coeff) == 2:
        entropy_coeff = max(_namespace.entropy_coeff)
        if _namespace.num_players:
            decay_timesteps = _namespace.num_players * train_batch_size * _namespace.entropy_coeff_decay
        else:
            decay_timesteps = (sum(range(2, 11)) / 9) * train_batch_size * _namespace.entropy_coeff_decay
        entropy_coeff_schedule = [(0, max(_namespace.entropy_coeff)), (decay_timesteps, min(_namespace.entropy_coeff))]
    else:
        raise ValueError('Expected 1 (constant) or 2 (initial and final) values for'
                         ' the entropy coefficient hyperparamter, but found {}'.format(_namespace.entropy_coeff))

    return tune.run(
        run_or_experiment=ppo.TimedPPO,
        config={
            # Training settings
            'env': 'take6',
            'env_config': {
                'num-players': _namespace.num_players,
                'expert': _namespace.expert_mode,
                'rwd-fn': _namespace.rwd_fn,
                'with-scores': _namespace.with_scores,
                'with-history': _namespace.with_history,
            },
            'num_workers': num_workers,
            'num_cpus_per_worker': 1,
            'num_envs_per_worker': num_envs_per_worker,
            'rollout_fragment_length': 10,
            'framework': framework,

            'model': {
                'fcnet_hiddens': [256, 256, 256],
                'custom_model': 'take6',
                'fcnet_activation': 'relu',
            },

            'lr': _namespace.lr,
            "entropy_coeff": entropy_coeff,
            'entropy_coeff_schedule': entropy_coeff_schedule,

            # Continuing Task settings
            'gamma': _namespace.gamma,
            'lambda': vars(_namespace)['lambda'],

            'multiagent': {
                'policies': {
                    # Our main policy, we'd like to optimize.
                    'learner': PolicySpec(None, None, None, None),
                    # mirrored policy for training
                    'opponent_v0': PolicySpec(None, None, None, {}),
                    # ref policy
                    'random': PolicySpec(policy.RandomPolicy, None, None, {})
                },
                'policy_mapping_fn': policy_mapping_fn,
                'policies_to_train': ['learner'],
            },

            'num_cpus_for_driver': 1,
            'tf_session_args': {
                'intra_op_parallelism_threads': 0,
                'inter_op_parallelism_threads': 0,
                'log_device_placement': False,
                'device_count': {'CPU': 1},
                'gpu_options': {'allow_growth': True},
                'allow_soft_placement': True,
            },
            'local_tf_session_args': {
                'intra_op_parallelism_threads': 0,
                'inter_op_parallelism_threads': 0,
            },

            'num_gpus': num_gpus,

            # PPO specific
            'train_batch_size': train_batch_size,
            'sgd_minibatch_size': sgd_minibatch_size,
            'num_sgd_iter': num_sgd_iter,
            'vf_clip_param': .2,
            'clip_param': .1,
            'kl_target': 1e-2,

            # Policy evaluation config
            'evaluation_interval': 1,
            'custom_eval_function': tournament,
            'evaluation_num_workers': num_eval_workers,
            'evaluation_duration': evaluation_duration,
            'evaluation_duration_unit': 'episodes',

            '_disable_preprocessor_api': True,

            'callbacks': callbacks.MultiCallbacks([CustomizedMetricsCallback, TrackingCallback]),
        },
        restore=_checkpoint,
        checkpoint_freq=1,
        stop=lambda t, r: (r['info']['learner']['learner']['learner_stats']['entropy'] <= _namespace.stop or
                           r['training_iteration'] >= _namespace.max_iterations),
        checkpoint_at_end=True,
        raise_on_failed_trial=False,
        max_failures=50,
        verbose=log.Verbosity.V1_EXPERIMENT,
        local_dir=local_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # checkpoint
    parser.add_argument('--run-id', type=str, help='The run-id to start the training from')
    parser.add_argument('--checkpoint', type=int, help='The checkpoint number to start the training from')

    parser.add_argument('--num-players', type=int, default=None, help='The number of players in the training env. '
                                                                      'By default, the number of players is arbitrary, '
                                                                      'ranging from 2 to 10 players.')
    parser.add_argument('--expert-mode', type=t_or_f, default=None,
                        help='Whether env follows expert rules mode or not. '
                             'By default, rules selected are arbitrarily.')

    # hyper-parameters
    parser.add_argument('--minibatch-size', type=int, default=1024, help='The sgd minibatch size')
    parser.add_argument('--batch-size', type=int, default=307_200, help='The sgd minibatch size')
    parser.add_argument('--num-sgd-iter', type=int, default=20, help='The number of sgd iterations per training step')
    parser.add_argument('--entropy-coeff', type=float, nargs='*', default=[1e-3, 3e-2],
                        help='The weight to the entropy coefficient in the loss function')
    parser.add_argument('--entropy-coeff-decay', type=int, default=350,
                        help='The number of training iterations to decay the entropy coefficient')
    parser.add_argument('--gamma', type=float, default=1., help='The discount rate')
    parser.add_argument('--lambda', type=float, default=.1, help='The eligibility trace')
    parser.add_argument('--lr', type=float, default=7e-5, help='The learning rate')
    parser.add_argument('--rwd-fn', type=str, default='proportional-score',
                        choices=['raw-score', 'proportional-score', 'classification'],
                        help='The reward signal to use')

    parser.add_argument('--with-scores', dest='with_scores', action='store_true',
                        help='Append current scores to agent\'s observations')
    parser.add_argument('--without-scores', dest='with_scores', action='store_false',
                        help='Remove scores from agent\'s observations')
    parser.set_defaults(with_scores=True)

    parser.add_argument('--with-history', dest='with_history', action='store_true',
                        help='Append history of played cards tp agent\'s observations')
    parser.add_argument('--without-history', dest='with_history', action='store_false',
                        help='Remove history of played cards from agent\'s observations')
    parser.set_defaults(with_history=True)

    parser.add_argument('--with-self-play', dest='with_self_play', action='store_true', help='Train by self-play')
    parser.add_argument('--without-self-play', dest='with_self_play', action='store_false',
                        help='Use only random initialized policies to train the agent')
    parser.set_defaults(with_self_play=True)

    parser.add_argument('--policy-buffer-size', type=int, default=5,
                        help='The number of historical policies to use during self-play')

    # miscellaneous
    parser.add_argument('--stop', type=float, default=.18, help='The policy entropy value which training stops')
    parser.add_argument('--max-iterations', type=int, default=600, help='The maximum number of training iterations')

    # debugging
    parser.add_argument('--debugging', action='store_true', help='Run locally with simplified settings')

    namespace = parser.parse_args(sys.argv[1:])

    if namespace.run_id or namespace.checkpoint:
        if not (namespace.run_id and namespace.checkpoint):
            raise RuntimeError('arguments --run-id and --checkpoint-id both are required for loading checkpoints')

    run = core.Run.get_context()
    run.set_tags(vars(namespace))
    if len(sys.argv) == 1:
        run.tag('baseline', 'True')

    ray.init(local_mode=namespace.debugging)

    POLICY_BUFFER_SIZE = namespace.policy_buffer_size

    tmp_dir = tempfile.mkdtemp()
    main(namespace, tmp_dir)

    sys.exit(0)
