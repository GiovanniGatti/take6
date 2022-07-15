import itertools
from io import TextIOWrapper
from typing import Optional, Any, List, Dict, Union

import yaml


class ConfigurationError(RuntimeError):

    def __init__(self, errors) -> None:
        super().__init__()
        self.errors = errors


class CheckConf:

    def __init__(
            self, entry_key: Optional[str] = None, entry_type: Optional[Any] = None, children: List['CheckConf'] = ()):
        self._entry_key = entry_key
        self._entry_type = entry_type
        self._children = children

    def validate(self, conf: Union[Dict[str, Any], List[Any]], errors: List[str]) -> None:
        if self._entry_key is None and self._entry_type is None:
            # entry node
            for child in self._children:
                child.validate(conf, errors)
            return

        if self._entry_key is not None:
            if self._entry_key not in conf.keys():
                errors.append('missing parameter \'{}\' at config {}'.format(self._entry_key, conf))
                return

            child_config = conf[self._entry_key]
        else:
            if len(conf.keys()) != 1:
                errors.append('Unexpected multiple entries for free input \'{}\''.format(next(iter(conf.keys()))))
            child_config = next(iter(conf.values()))

        if not isinstance(child_config, self._entry_type):
            errors.append('wrong typing for parameter \'{}\', expected type \'{}\' at at config {}'
                          .format(self._entry_key, self._entry_type, conf))

        if self._entry_type == list:
            assert len(self._children) == 1, 'Same typing must apply to all elements in the list'
            assert isinstance(child_config, list), 'Racing condition'
            for child, _child_config in zip(itertools.repeat(self._children[0]), child_config):
                child.validate(_child_config, errors)
            return

        for child in self._children:
            child.validate(child_config, errors)


def check_az_conf() -> CheckConf:
    return CheckConf(children=[
        CheckConf('azure', dict, children=[CheckConf('name', str),
                                           CheckConf('subscription-id', str),
                                           CheckConf('resource-group', str),
                                           CheckConf('vnet-name', str),
                                           CheckConf('registry', str)]),
        CheckConf('clusters', list, children=[CheckConf(None, dict, children=[CheckConf('name', str),
                                                                              CheckConf('shm-size', str)])])])


def load_from_yaml(path: TextIOWrapper) -> Dict[str, Any]:
    try:
        deployment_conf = yaml.safe_load(path)
    except IOError as e:
        raise e

    check_conf = check_az_conf()
    errors: List[str] = []
    check_conf.validate(deployment_conf, errors)

    if errors:
        raise ConfigurationError(errors)

    return deployment_conf
