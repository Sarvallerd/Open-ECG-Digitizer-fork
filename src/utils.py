from yacs.config import CfgNode as CN
from typing import Any
from typing import Dict, Tuple
from copy import deepcopy
from torch.utils.data import DataLoader
from torch import nn
import importlib


def get_data_loaders(
    dataset_config: Dict[Any, Any], dataloader_config: Dict[Any, Any]
) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    dataset_model = import_class_from_path(dataset_config["class_path"])

    dataset_kwargs = dataset_config["KWARGS"]

    train_dataloader = DataLoader(
        dataset_model(**(dataset_kwargs | dataset_config["TRAIN"]["KWARGS"])), **dataloader_config
    )
    val_dataloader = DataLoader(
        dataset_model(**(dataset_kwargs | dataset_config["VAL"]["KWARGS"])), **dataloader_config
    )
    test_dataloader = DataLoader(
        dataset_model(**(dataset_kwargs | dataset_config["TEST"]["KWARGS"])), **dataloader_config
    )
    return train_dataloader, val_dataloader, test_dataloader


def import_class_from_path(path: str) -> Any:
    module = importlib.import_module(".".join(path.split(".")[:-1]))
    return getattr(module, path.split(".")[-1])


def load_model(config: CN) -> nn.Module:
    model_class = import_class_from_path(config.MODEL.class_path)
    return model_class(**config.MODEL.KWARGS)  # type: ignore


def merge_ray_config_with_config(config: CN, ray_config: CN) -> CN:
    config = deepcopy(config)
    ray_config = deepcopy(ray_config)
    model_kwargs = deepcopy(config)
    for k, v in ray_config.items():
        if k in model_kwargs:
            config.MODEL.KWARGS[k] = v
    return config
