import torch
from torch import nn
import os


def make_fc_layers(cfg, in_channels=8):
    layers = []
    for v in cfg[:-1]:
        layers += [nn.Linear(in_channels, v), nn.ReLU(inplace=True)]
        in_channels = v
    layers += [nn.Linear(in_channels, cfg[-1])]
    return nn.Sequential(*layers)


def adjust_lr(optimizer, lr):
    group_num = len(optimizer.param_groups)
    for i in range(group_num):
        optimizer.param_groups[i]["lr"] = lr


def save_model(dict_, path, model_name="default.pth"):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(dict_, os.path.join(path, model_name))
