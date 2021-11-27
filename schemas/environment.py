import datetime
from pathlib import Path
from typing import Literal

import torch
from pydantic import validator
from pydantic.types import PositiveInt, DirectoryPath

from consts import DATETIME_STRING_FORMAT
from schemas.utils import ImmutableArgs


class EnvironmentArgs(ImmutableArgs):

    #: On which device to train.
    device: Literal['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'] = 'cpu'

    #: Output path for the experiment - a sub-directory named with the date & time will be created within.
    path: DirectoryPath = './experiments'

    #: How many iterations between each training log.
    log_interval: PositiveInt = 100

    @validator('path', always=True)
    def create_out_dir(cls, v: Path):
        datetime_string = datetime.datetime.now().strftime(DATETIME_STRING_FORMAT)
        out_dir = v / datetime_string
        out_dir.mkdir(exist_ok=True)  # exist_ok because this validator is being called multiple times (I think)
        return out_dir

    @validator('device')
    def validate_device_exists(cls, v):
        if v.startswith('cuda:'):
            assert torch.cuda.is_available(), f"CUDA is not available, so can't use device {v}"
            assert int(v[-1]) < torch.cuda.device_count(), f"GPU index {v[-1]} is higher than the number of GPUs."
        return v
