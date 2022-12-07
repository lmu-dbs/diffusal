from typing import Type

from .random import RandomSampling
from .diff_based.diffusal import DiffusAL
from .diff_based.diffusal_ablations import DiffusALNoDiv, DiffusALNoImp, DiffusALNoUnc
from .entropy import EntropySampling
from .degree import DegreeSampling
from .age import AGE
from .featprop import FeatProp
from .grain import GRAIN
from .lscale import LSCALE
from .coreset import Coreset


def get_strategy_class(name: str) -> Type:
    if name == 'random':
        return RandomSampling
    elif name == 'entropy':
        return EntropySampling
    elif name == 'degree':
        return DegreeSampling
    elif name == 'age':
        return AGE
    elif name == 'featprop':
        return FeatProp
    elif name == 'grain':
        return GRAIN
    elif name == 'diffusal':
        return DiffusAL
    elif name == 'diff_nodiv':
        return DiffusALNoDiv
    elif name == 'diff_noimp':
        return DiffusALNoImp
    elif name == 'diff_nounc':
        return DiffusALNoUnc
    elif name == 'lscale':
        return LSCALE
    else:
        raise ValueError(f'Unrecognized strategy: {name}')
