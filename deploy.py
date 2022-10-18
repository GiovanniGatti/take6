import argparse
import logging
import re
import sys

from azureml import core
from azureml.core import experiment, environment, authentication, runconfig

from take6.aztools import checkconf, hyperdriveconf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--deployment-config', type=argparse.FileType('r'), default='./deployment-conf.yaml',
                        help='Azure configuration file with workspace name, subscription id and others')
    parser.add_argument('--cluster', type=str, help='The cluster to use (from deployment config file)')

    subparsers = parser.add_subparsers(dest='command', description='', help='')

    run_parser = subparsers.add_parser('run', help='Runs target script on an AzureML Compute Instance')
    run_parser.add_argument('-s', '--entry-script', type=str, required=True, help='Training script that launches entry')
    run_parser.add_argument('--num-submits', type=int, default=1, help='How many copies of this job to launch')
    run_parser.add_argument('--script-params', nargs=argparse.REMAINDER, help='Target script parameters')

    hyperdrive_parser = subparsers.add_parser('hyperdrive', help='Runs target script with HyperDriving')
    hyperdrive_parser.add_argument('-s', '--entry-script', type=str, help='Training script that launches entry')
    hyperdrive_parser.add_argument('--hyperdrive-config', type=argparse.FileType('r'), required=True,
                                   help='Hyperdrive configuration file')
    hyperdrive_parser.add_argument('--script-params', nargs=argparse.REMAINDER, help='Target script parameters')

    namespace = parser.parse_args(sys.argv[1:])

    # ML Azure authentication warning messages are logged at root level
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    try:
        deployment_conf = checkconf.load_from_yaml(namespace.deployment_config)
    except IOError as e:
        print(e, file=sys.stderr)
        print('Unable to open YAML config file', file=sys.stderr)
        sys.exit(1)
    except checkconf.ConfigurationError as e:
        print(*e.errors, sep='\n', file=sys.stderr)
        print('Invalid YAML config file', file=sys.stderr)
        sys.exit(1)

    azure = deployment_conf['azure']

    ws = core.Workspace.get(name=azure['name'],
                            subscription_id=azure['subscription-id'],
                            resource_group=azure['resource-group'],
                            auth=authentication.InteractiveLoginAuthentication())

    clusters = deployment_conf['clusters']
    clusters = {next(iter(c.keys())): c[next(iter(c.keys()))] for c in clusters}
    default_cluster = [f.replace('*', '') for f in filter(lambda x: x.endswith('*'), clusters.keys())]
    known_clusters = [c.replace('*', '') for c in clusters.keys()]
    clusters = {k.replace('*', ''): v for k, v in clusters.items()}
    if len(default_cluster) > 1:
        print('Illegal configuration file. Only a single default cluster is allowed, but found {}'
              .format(default_cluster))
        sys.exit(1)

    if len(default_cluster) == 0:
        print('Illegal configuration file. Expected at least a single default cluster is allowed, but none found')
        sys.exit(1)

    default_cluster = default_cluster[0]
    if namespace.cluster is not None and namespace.cluster not in known_clusters:
        print('cluster \'{}\' is not specified in yaml configuration. Select one of {}'
              .format(namespace.cluster, known_clusters))
        sys.exit(1)

    cluster = clusters[default_cluster] if namespace.cluster is None else clusters[namespace.cluster]
    head = cluster['name']

    if head not in ws.compute_targets:
        print('Unknown compute for head {}'.format(head))
        sys.exit(1)

    head_compute_target = ws.compute_targets[head]

    if namespace.command == 'run':
        regex = re.compile(r'(?P<expname>\w+).py$')
        experiment_name = regex.search(namespace.entry_script).group('expname')
        experiment_name = 'take6-' + experiment_name
    else:
        experiment_name = 'take6-hyperdrive'
    exp = experiment.Experiment(workspace=ws, name=experiment_name)

    registry: str = azure['registry']

    head_env = environment.Environment(name='take6-env')
    head_env.environment_variables['AZUREML_COMPUTE_USE_COMMON_RUNTIME'] = False
    head_env.docker.base_image = '{}.azurecr.io/take6{}:latest'.format(
        registry, '-gpu' if head.find('gpu') >= 0 else '')
    head_env.python.user_managed_dependencies = True
    shm_size = cluster['shm-size']

    run_config = core.ScriptRunConfig(
        source_directory='./src',
        script=namespace.entry_script,
        arguments=namespace.script_params,
        compute_target=head_compute_target,
        environment=head_env,
        docker_runtime_config=runconfig.DockerConfiguration(use_docker=True, shm_size=shm_size))

    if namespace.command == 'run':
        submitted_runs = []
        # noinspection PyBroadException
        # we want to stop on any error
        try:
            for i in range(namespace.num_submits):
                submitted_runs.append(exp.submit(run_config))
        except:
            print('Submission failed, cancelling jobs...')
            for run in submitted_runs:
                run.cancel()
            print('Submitted jobs cancelled successfully...')
            sys.exit(1)

        for i, run in enumerate(submitted_runs):
            print('Run {}, {} -> {}'.format(run.number, run.id, run.get_portal_url()))
        sys.exit(0)

    hd_config = hyperdriveconf.load_from_yaml(namespace.hyperdrive_config, run_config=run_config)
    run = exp.submit(hd_config)
    print('Run {}, {} -> {}'.format(run.number, run.id, run.get_portal_url()))
    sys.exit(0)
