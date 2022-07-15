import abc
from io import TextIOWrapper
from typing import Dict, Any, List, Optional

import yaml
from azureml import core
from azureml.train import hyperdrive


class _Node(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def compile(self, extra_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        pass


class _ArgNode(_Node):

    def __init__(self, name: str, arg: Any):
        self._name = name
        self._arg = arg

    @property
    def name(self) -> str:
        return self._name

    def compile(self, extra_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        if extra_kwargs:
            raise ValueError('Unexpected extra parameters to simple argument node')
        return self._arg


class _TypedArgNode(_Node):

    def __init__(self, name: str, _type: str, args: Any):
        self._name: str = name
        self._type: str = _type
        self._args: Any = args

    @property
    def name(self) -> str:
        return self._name

    def compile(self, extra_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        fn = getattr(hyperdrive, self._type)
        args = self._args if isinstance(self._args, list) else [self._args]

        if extra_kwargs:
            return fn(*args, **extra_kwargs)

        return fn(*args)


class _KwargsNode(_Node):

    def __init__(self, name: str, children: List[_Node]):
        self._name = name
        self._children = children

    @property
    def name(self) -> str:
        return self._name

    def compile(self, extra_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        kwargs = {c.name: c.compile() for c in self._children}
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs


class _TypedNode(_Node):

    def __init__(self, name: str, _type: str, children: List[_Node]):
        self._name: str = name
        self._type: str = _type
        self._children: List[_Node] = children

    @property
    def name(self) -> str:
        return self._name

    def compile(self, extra_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        fn = getattr(hyperdrive, self._type)
        kwargs = {c.name: c.compile() for c in self._children}
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return fn(**kwargs)


class _NoArgNode(_Node):

    def __init__(self, name: str, _type: str):
        self._name: str = name
        self._type: str = _type

    @property
    def name(self) -> str:
        return self._name

    def compile(self, extra_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        if extra_kwargs:
            raise ValueError('Unexpected parameters to node that requires no parameters')
        fn = getattr(hyperdrive, self._type)
        return fn()


def _parse(node: Dict[str, Any]) -> List[_Node]:
    ast = []
    for k, v in node.items():
        if isinstance(v, dict):
            if 'type' in v:
                if 'args' not in v:
                    ast.append(_NoArgNode(k, v['type']))
                elif isinstance(v['args'], dict):
                    children = _parse(v['args'])
                    ast.append(_TypedNode(k, v['type'], children))
                else:
                    ast.append(_TypedArgNode(k, v['type'], v['args']))
            else:
                children = _parse(v)
                ast.append(_KwargsNode(k, children))
        else:
            ast.append(_ArgNode(k, v))
    return ast


def load_from_yaml(path: TextIOWrapper, run_config: core.ScriptRunConfig) -> hyperdrive.HyperDriveRunConfig:
    try:
        conf = yaml.safe_load(path)
    except IOError as e:
        raise e

    ast = _parse(conf)

    return next(iter(ast)).compile({'run_config': run_config})
