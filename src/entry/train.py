import argparse
import math
import multiprocessing
import pathlib
import sys
import tempfile
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import ray
import trueskill
from azureml import core
from ray import tune, rllib
from ray.rllib import agents
from ray.rllib.agents import callbacks
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms import algorithm
from ray.rllib.evaluation import worker_set
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import Dict
from ray.tune.analysis import experiment_analysis

from take6 import env, model
from take6.aztools import checkpoint

_, tf, _ = try_import_tf()


def win_probability(player_1, player_2):
    delta_mu = player_1.mu - player_2.mu
    sum_sigma = sum(r.sigma ** 2 for r in [player_1, player_2])
    ts = trueskill.global_env()
    denom = math.sqrt(2 * (ts.beta * ts.beta) + sum_sigma)
    return ts.cdf(delta_mu / denom)


class TrackingCallback(callbacks.DefaultCallbacks):

    def on_train_result(self, *,
                        _algorithm: Optional['Algorithm'] = None,
                        result: Dict[Any, Any],
                        trainer: agents.Trainer,
                        **kwargs: Dict[Any, Any]) -> None:
        _run = core.Run.get_context()
        # custom metrics
        _run.log(name='training/win_rate', value=result['win_rate'])
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
            _run.log(name='eval/win_rate', value=result['evaluation']['custom_metrics']['eval_win_rate'])

            ratings = result['evaluation']['custom_metrics']['trueskill']
            _learner = ratings['learner']
            _opponent_v0 = ratings['opponent_v0']
            _run.log(name='eval/mu', value=_learner.mu)
            _run.log(name='eval/sigma', value=_learner.sigma)
            _run.log(name='eval/quality', value=trueskill.quality_1vs1(_learner, _opponent_v0))
            _run.log(name='eval/win_probability', value=win_probability(_learner, _opponent_v0))
            _run.log(name='eval/opponent_v0_mu', value=_opponent_v0.mu)
            _run.log(name='eval/opponent_v0_sigma', value=_opponent_v0.sigma)


class SelfPlayCallback(callbacks.DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self.current_opponent = 3
        self._training_step = 0

    def on_episode_end(self, *,
                       worker: rllib.RolloutWorker,
                       base_env: rllib.BaseEnv,
                       policies: Dict[rllib.utils.typing.PolicyID, rllib.Policy],
                       episode: rllib.evaluation.episode.MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs: Dict[Any, Any]) -> None:
        super().on_episode_end(
            worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs)
        learner_score = episode.agent_rewards[0, 'learner']
        for (_, t), score in episode.agent_rewards.items():
            if t != 'learner':
                if score >= learner_score:
                    episode.custom_metrics['win'] = 0
                    return
        episode.custom_metrics['win'] = 1

    def on_train_result(self, result, **kwargs):
        trainer = kwargs['trainer']
        win_rate = result['custom_metrics']['win_mean']
        result['win_rate'] = win_rate

        if self._training_step > 0 and self._training_step % 25 == 0 and False:
            new_pol_id = f'opponent_v{self.current_opponent}'

            def policy_mapping_fn(agent_id, episode, worker, **kwargs) -> str:
                if agent_id == 0:
                    # I haven't found anyway to know which policies were
                    # already selected between two calls of this function.
                    # The workaround I found was to store this temporary
                    # information in the user_data section of the episode.
                    episode.user_data['selected_policies_id'] = []
                    return 'learner'

                already_selected_ids = np.array(episode.user_data['selected_policies_id'], dtype=int)

                weights = np.arange(self.current_opponent + 1)[::-1] / 2
                weights[already_selected_ids] = np.inf
                weights = np.exp(-weights) / np.sum(np.exp(-weights))
                _id = np.random.choice(list(range(self.current_opponent + 1)), p=weights)

                episode.user_data['selected_policies_id'].append(_id)

                return 'opponent_v{}'.format(_id)

            new_policy = trainer.add_policy(
                policy_id=new_pol_id,
                policy_cls=type(trainer.get_policy('learner')),
                policy_mapping_fn=policy_mapping_fn,
            )

            main_state = trainer.get_policy('learner').get_state()
            new_policy.set_state(main_state)
            trainer.workers.sync_weights()

            self.current_opponent += 1

        result['league_size'] = self.current_opponent
        self._training_step += 1


class Evaluation:

    def __init__(self):
        self._ratings = defaultdict(lambda: trueskill.Rating())

    def __call__(self, _algorithm: algorithm.Algorithm, eval_workers: worker_set.WorkerSet):
        def eval_policy_mapping_fn(agent_id, episode, worker, **kwargs) -> str:
            if agent_id == 0:
                episode.user_data['selected_policies_id'] = []
                return 'learner'
            already_selected_ids = np.array(episode.user_data['selected_policies_id'], dtype=int)
            num_policies = len(worker.policy_map) - 1
            choices = np.arange(num_policies)
            select = np.ones_like(choices, dtype=bool)
            select[already_selected_ids] = False
            choices = choices[select]
            _id = np.random.choice(choices)
            episode.user_data['selected_policies_id'].append(_id)
            return 'opponent_v{}'.format(_id)

        eval_workers.foreach_worker(lambda w: w.set_policy_mapping_fn(eval_policy_mapping_fn))

        assert _algorithm.config['evaluation_duration_unit'] == 'episodes'
        num_episodes = _algorithm.config['evaluation_duration']
        num_workers = len(eval_workers.remote_workers())

        try:
            for i in range(max(1, round(num_episodes / num_workers))):
                ray.get([w.sample.remote() for w in eval_workers.remote_workers()])
        except AssertionError or StopIteration:
            _, new_workers = eval_workers.recreate_failed_workers(_algorithm.workers.local_worker())
            eval_workers.reset(new_workers)
            return self(_algorithm, eval_workers)  # see https://github.com/ray-project/ray/issues/15297

        episodes, _ = collect_episodes(remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)

        for episode in episodes:
            ratings = []
            scores = []
            players = []

            for (_, t), score in episode.agent_rewards.items():
                ratings.append((self._ratings[t],))
                scores.append(score)
                players.append(t)

            classification = np.zeros(len(ratings))
            classification[np.argsort(scores)] = np.arange(4)[::-1]
            s, c = np.unique(scores, return_counts=True)
            for _s in s[c > 1]:  # handle ties
                classification[scores == _s] = np.min(classification[scores == _s])
            new_ratings = trueskill.rate(ratings, ranks=list(classification))

            episode.custom_metrics['win'] = classification[0] == 0

            assert np.unique(players).shape[0] == 4, 'Repeated versions of a policy are playing within the same match'

            for i, p in enumerate(players):
                (self._ratings[p],) = new_ratings[i]

        metrics = summarize_episodes(episodes, keep_custom_metrics=True)

        custom_metrics = metrics['custom_metrics']
        custom_metrics['trueskill'] = self._ratings
        custom_metrics['eval_win_rate'] = np.mean(custom_metrics['win'])

        return metrics


def main(_namespace: argparse.Namespace, _tmp_dir: str) -> experiment_analysis.ExperimentAnalysis:
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    num_cpus = multiprocessing.cpu_count()
    eval_workers = 2

    if _namespace.debugging:
        num_workers = 1
        num_envs_per_worker = 1
        sgd_minibatch_size = 10
        num_sgd_iter = 2
        eval_workers = 1
        framework = 'tf2'
        local_dir = str(pathlib.Path(_tmp_dir, 'ray-results'))
    else:
        num_workers = num_cpus - eval_workers - 1
        num_envs_per_worker = int(math.ceil(_namespace.batch_size / (num_workers * 10)))
        sgd_minibatch_size = _namespace.minibatch_size
        num_sgd_iter = _namespace.num_sgd_iter
        framework = 'tf'
        local_dir = './logs/ray-results'

    train_batch_size = num_workers * num_envs_per_worker * 10

    _checkpoint = checkpoint.recover_from_preemption(local_dir)
    if _checkpoint is None and _namespace.checkpoint is not None:
        _checkpoint = checkpoint.load_checkpoint_from(_namespace.run_id, _namespace.checkpoint)
        print('Using checkpoint file \'{}\' from run-id \'{}\''.format(_checkpoint, _namespace.run_id))
    elif _checkpoint is not None:
        print('Using checkpoint file \'{}\' before preemption'.format(_checkpoint))

    ModelCatalog.register_custom_model('take6', model.Take6Model)
    tune.register_env('take6', env.take6)

    eval_method = Evaluation()

    def warmup_agent_mapping(agent_id, **kwargs):
        return 'learner' if agent_id == 0 else \
            'opponent_v{}'.format(agent_id - 1)

    return tune.run(
        run_or_experiment=PPOTrainer,
        config={
            # Training settings
            'env': 'take6',
            'env_config': {'num-players': 4},
            'num_workers': num_workers,
            'num_cpus_per_worker': 1,
            'num_envs_per_worker': num_envs_per_worker,
            'rollout_fragment_length': 10,
            'framework': framework,

            'model': {
                'fcnet_hiddens': [512, 512],
                'custom_model': 'take6',
                'fcnet_activation': 'relu',
            },

            'lr': _namespace.lr,
            "entropy_coeff": _namespace.entropy_coeff,
            'vf_loss_coeff': _namespace.vf_loss_coeff,

            # Continuing Task settings
            'gamma': _namespace.gamma,
            'lambda': vars(_namespace)['lambda'],

            'multiagent': {
                'policies': {
                    # Our main policy, we'd like to optimize.
                    'learner': PolicySpec(None, None, None, None),
                    # An initial random opponents to play against.
                    'opponent_v0': PolicySpec(None, None, None, {}),
                    'opponent_v1': PolicySpec(None, None, None, {}),
                    'opponent_v2': PolicySpec(None, None, None, {}),
                },
                'policy_mapping_fn': warmup_agent_mapping,
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
            'vf_clip_param': 10.,

            # Policy evaluation config
            'evaluation_interval': 1,
            'custom_eval_function': lambda alg, workers: eval_method(alg, workers),
            'evaluation_num_workers': eval_workers,
            'evaluation_duration': 2 if _namespace.debugging else 1024,
            'evaluation_duration_unit': 'episodes',

            'callbacks': callbacks.MultiCallbacks([SelfPlayCallback, TrackingCallback]),
        },
        restore=_checkpoint,
        checkpoint_freq=5,
        stop=lambda t, r: (r['info']['learner']['learner']['learner_stats']['entropy'] <= _namespace.stop or
                           r['training_iteration'] >= _namespace.max_iterations),
        checkpoint_at_end=True,
        raise_on_failed_trial=False,
        local_dir=local_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # checkpoint
    parser.add_argument('--run-id', type=str, help='The run-id to start the training from')
    parser.add_argument('--checkpoint', type=int, help='The checkpoint number to start the training from')

    # hyper-parameters
    parser.add_argument('--minibatch-size', type=int, default=2048, help='The sgd minibatch size')
    parser.add_argument('--batch-size', type=int, default=102_400, help='The sgd minibatch size')
    parser.add_argument('--num-sgd-iter', type=int, default=32, help='The number of sgd iterations per training step')
    parser.add_argument('--entropy-coeff', type=float, default=1.5e-3,
                        help='The weight to the entropy coefficient in the loss function')
    parser.add_argument('--gamma', type=float, default=1., help='The discount rate')
    parser.add_argument('--lambda', type=float, default=.9, help='The eligibility trace')
    parser.add_argument('--vf-loss-coeff', type=float, default=1.,
                        help='The value loss coefficient (optimize it if actor and critic share layers)')
    parser.add_argument('--lr', type=float, default=6e-6, help='The learning rate')

    # miscellaneous
    parser.add_argument('--stop', type=float, default=.05, help='The policy entropy value which training stops')
    parser.add_argument('--max-iterations', type=int, default=170, help='The maximum number of training iterations')

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

    tmp_dir = tempfile.mkdtemp()
    analysis = main(namespace, tmp_dir)

    sys.exit(0)
