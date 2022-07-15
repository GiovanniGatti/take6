import argparse
import math
import multiprocessing
import pathlib
import sys
import tempfile
from typing import Optional, Any

import numpy as np
import ray
from azureml import core
from ray import tune, rllib
from ray.rllib import agents
from ray.rllib.agents import callbacks
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import Dict
from ray.tune.analysis import experiment_analysis

from take6 import env, model
from take6.aztools import checkpoint

_, tf, _ = try_import_tf()


class TrackingCallback(callbacks.DefaultCallbacks):

    def __init__(self) -> None:
        super().__init__()
        self._run = core.Run.get_context()

    def on_episode_start(self, *,
                         worker: rllib.RolloutWorker,
                         base_env: rllib.BaseEnv,
                         policies: Dict[rllib.utils.typing.PolicyID, rllib.Policy],
                         episode: rllib.evaluation.episode.MultiAgentEpisode,
                         env_index: Optional[int] = None,
                         **kwargs: Dict[Any, Any]) -> None:
        super().on_episode_start(
            worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs)

    def on_train_result(self, *, trainer: agents.Trainer, result: Dict[Any, Any], **kwargs: Dict[Any, Any]) -> None:
        # custom metrics
        self._run.log(name='policy_reward_mean', value=result['policy_reward_mean']['learner'])
        self._run.log(name='win_rate', value=result['win_rate'])
        self._run.log(name='league_size', value=result['league_size'])

        # learning stats
        learner = result['info']['learner']['learner']['learner_stats']
        self._run.log(name='learner_stats/entropy', value=learner['entropy'])
        self._run.log(name='learner_stats/entropy_coeff', value=learner['entropy_coeff'])
        self._run.log(name='learner_stats/kl', value=learner['kl'])
        self._run.log(name='learner_stats/cur_kl_coeff', value=learner['cur_kl_coeff'])
        self._run.log(name='learner_stats/policy_loss', value=learner['policy_loss'])
        self._run.log(name='learner_stats/vf_loss', value=learner['vf_loss'])
        self._run.log(name='learner_stats/vf_explained_var', value=learner['vf_explained_var'])
        self._run.log(name='learner_stats/total_loss', value=learner['total_loss'])
        self._run.log(name='learner_stats/curr_lr', value=learner['cur_lr'])

        # perf
        perf = result['perf']
        sampler_perf = result['sampler_perf']
        self._run.log(name='perf/mean_env_wait_ms', value=sampler_perf['mean_env_wait_ms'])
        self._run.log(name='perf/mean_raw_obs_processing_ms', value=sampler_perf['mean_raw_obs_processing_ms'])
        self._run.log(name='perf/mean_inference_ms', value=sampler_perf['mean_inference_ms'])
        self._run.log(name='perf/mean_action_processing_ms', value=sampler_perf['mean_action_processing_ms'])
        self._run.log(name='perf/cpu_util_percent', value=perf['cpu_util_percent'])
        self._run.log(name='perf/ram_util_percent', value=perf['ram_util_percent'])
        if 'gpu_util_percent0' in perf:
            self._run.log(name='perf/gpu_util_percent0', value=perf['gpu_util_percent0'])
            self._run.log(name='perf/vram_util_percent0', value=perf['vram_util_percent0'])

        timers = result['timers']
        self._run.log(name='timers/sample_time_ms', value=timers['sample_time_ms'])
        self._run.log(name='timers/sample_throughput', value=timers['sample_throughput'])
        self._run.log(name='timers/learn_time_ms', value=timers['learn_time_ms'])
        self._run.log(name='timers/learn_throughput', value=timers['learn_throughput'])
        self._run.log(name='timers/update_time_ms', value=timers['update_time_ms'])
        if 'load_throughput' in timers:
            self._run.log(name='timers/load_throughput', value=timers['load_throughput'])
            self._run.log(name='timers/load_time_ms', value=timers['load_time_ms'])

        # progress
        self._run.log(name='progress/timesteps_total', value=result['timesteps_total'])
        self._run.log(name='progress/time_this_iter_s', value=result['time_this_iter_s'])
        self._run.log(name='progress/time_total_s', value=result['time_total_s'])
        self._run.log(name='progress/episodes_total', value=result["episodes_total"])


class SelfPlayCallback(callbacks.DefaultCallbacks):
    def __init__(self, win_rate_threshold: float = .95):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..
        self._win_rate_threshold = win_rate_threshold
        self.current_opponent = 0

    def on_train_result(self, result, **kwargs):
        trainer = kwargs['trainer']
        main_rew = result['hist_stats'].pop('policy_learner_reward')
        opponent_rew = list(result['hist_stats'].values())[0]
        assert len(main_rew) == len(opponent_rew)
        won = 0
        for r_main, r_opponent in zip(main_rew, opponent_rew):
            if r_main > r_opponent:
                won += 1
        win_rate = won / len(main_rew)
        result['win_rate'] = win_rate
        if win_rate > self._win_rate_threshold:
            self.current_opponent += 1
            new_pol_id = f'opponent_v{self.current_opponent}'

            def policy_mapping_fn(agent_id, episode, worker, **kwargs) -> str:
                if agent_id == 0:
                    return 'learner'

                weights = np.arange(self.current_opponent)[::-1]
                weights = np.exp(-weights) / np.sum(np.exp(-weights))
                return 'opponent_v{}'.format(
                    np.random.choice(list(range(1, self.current_opponent + 1)), p=weights))

            new_policy = trainer.add_policy(
                policy_id=new_pol_id,
                policy_cls=type(trainer.get_policy('learner')),
                policy_mapping_fn=policy_mapping_fn,
            )

            main_state = trainer.get_policy('learner').get_state()
            new_policy.set_state(main_state)
            trainer.workers.sync_weights()

        result['league_size'] = self.current_opponent + 2


def main(_namespace: argparse.Namespace, _tmp_dir: str) -> experiment_analysis.ExperimentAnalysis:
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    num_cpus = multiprocessing.cpu_count()

    if _namespace.debugging:
        num_workers = 1
        num_envs_per_worker = 1
        sgd_minibatch_size = 10
        num_sgd_iter = 2
        framework = 'tf2'
        local_dir = str(pathlib.Path(_tmp_dir, 'ray-results'))
    else:
        num_workers = num_cpus - 1
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

    def policy_mapping_fn(agent_id, episode, worker, **kwargs) -> str:
        return 'learner' if agent_id == 0 else 'opponent'

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
                    'learner': PolicySpec(None, None, None, {}),
                    # An initial random opponent to play against.
                    'opponent': PolicySpec(None, None, None, {}),
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
            'vf_clip_param': 10.,

            # Policy evaluation config
            'evaluation_interval': 0,

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
    parser.add_argument('--minibatch-size', type=int, default=512, help='The sgd minibatch size')
    parser.add_argument('--batch-size', type=int, default=5120, help='The sgd minibatch size')
    parser.add_argument('--num-sgd-iter', type=int, default=8, help='The number of sgd iterations per training step')
    parser.add_argument('--entropy-coeff', type=float, default=1e-4,
                        help='The weight to the entropy coefficient in the loss function')
    parser.add_argument('--gamma', type=float, default=.9995, help='The discount rate')
    parser.add_argument('--lambda', type=float, default=.98, help='The eligibility trace')
    parser.add_argument('--vf-loss-coeff', type=float, default=1.,
                        help='The value loss coefficient (optimize it if actor and critic share layers)')
    parser.add_argument('--lr', type=float, default=7e-4, help='The learning rate')

    # miscellaneous
    parser.add_argument('--stop', type=float, default=1.5, help='The policy entropy value which training stops')
    parser.add_argument('--max-iterations', type=int, default=200, help='The maximum number of training iterations')

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
    while True:
        # Re-launch trial when 'Assert agent_key not in self.agent_collectors' is bug fixed
        # We can safely remove it when https://github.com/ray-project/ray/issues/15297 is closed.
        relaunch = False
        incomplete_trials = []
        for trial in analysis.trials:
            if trial.status == experiment_analysis.Trial.ERROR:
                if 'assert agent_key not in self.agent_collectors' in trial.error_msg:
                    relaunch = True
                else:
                    incomplete_trials.append(trial)

        if incomplete_trials:
            raise tune.TuneError("Trials did not complete", incomplete_trials)

        if relaunch:
            analysis = main(namespace, tmp_dir)
            continue
        break

    sys.exit(0)
