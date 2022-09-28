import json
import math
import pathlib
import re
import tempfile
from typing import Optional, Dict, Any

from azureml import core
from azureml._common.async_utils import task_queue

from take6.aztools import workspace


def load_checkpoint_from(run_id: str, checkpoint: int, target_dir: str = tempfile.mkdtemp()) -> str:
    ws = workspace.get_workspace()
    run = core.Run.get(ws, run_id)
    checkpoint_id = 'checkpoint-{}'.format(checkpoint)
    target_checkpoint = next(filter(lambda filename: filename.endswith(checkpoint_id), run.get_file_names()))
    parent = pathlib.Path(target_checkpoint).parent
    task_queue.DEFAULT_FLUSH_TIMEOUT_SECONDS = 300
    run.download_files(prefix=str(parent), output_directory=target_dir)
    return str(pathlib.Path(target_dir, parent, checkpoint_id))


def retrieve_metrics(run_id: str) -> None:
    ws = workspace.get_workspace()
    run = core.Run.get(ws, run_id)
    current_run = core.Run.get_context()

    # from https://stackoverflow.com/questions/24483182/python-split-list-into-n-chunks
    def chunks(l, n):
        """Yield n number of sequential chunks from l."""
        d, r = divmod(len(l), n)
        for i in range(n):
            si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
            yield l[si:si + (d + 1 if i < r else d)]

    metrics = run.get_metrics()
    for k, v in metrics.items():
        for chunk in chunks(v, max(1, math.ceil(len(v) / 250))):  # see https://aka.ms/azure-machine-learning-limits
            current_run.log_list(name=k, value=chunk)


def get_latest_checkpoint(run_id: str) -> Optional[int]:
    if 'OfflineRun' in run_id:
        return None
    ws = workspace.get_workspace()
    run = core.Run.get(ws, run_id)
    ids = list(map(lambda _id: int(_id),
                   map(lambda m: m.group('id'),
                       filter(lambda i: i is not None,
                              [re.search('checkpoint-(?P<id>[0-9]+)$', file) for file in run.get_file_names()]))))
    if len(ids) == 0:
        return None
    return max(ids)


def get_params(run_id: str) -> Dict[Any, Any]:
    ws = workspace.get_workspace()
    run = core.Run.get(ws, run_id)
    params_file = next(filter(lambda _f: _f.endswith('params.json'), run.get_file_names()))
    tmp_dir = tempfile.mkdtemp()
    run.download_file(params_file, output_file_path=str(tmp_dir))
    params_path = pathlib.Path(tmp_dir, 'params.json')
    with open(params_path, 'r') as f:
        params = json.load(f)
    return params


def recover_from_preemption(local_dir: str) -> Optional[str]:
    _local_dir = pathlib.Path(local_dir)
    local_checkpoints = sorted(_local_dir.glob('**/checkpoint_*[0-9]'), reverse=True)
    if len(local_checkpoints) > 0:
        checkpoint_dir = pathlib.Path(next(iter(local_checkpoints)))
        checkpoint_file = next(checkpoint_dir.glob('checkpoint-*[0-9]'))
        return str(checkpoint_file)

    current_run_id = core.Run.get_context().id
    checkpoint_id = get_latest_checkpoint(current_run_id)
    if checkpoint_id is not None:
        return load_checkpoint_from(current_run_id, checkpoint_id)

    # TODO: remove it when debugging is performed
    print('Warning: no checkpoint found at {}'.format(local_dir))
    if _local_dir.exists():
        print('Contents: {}'.format(list(_local_dir.iterdir())))
    else:
        print('{} does not exist'.format(_local_dir))
    print('No checkpoint found for run_id {}'.format(current_run_id))
    return None
