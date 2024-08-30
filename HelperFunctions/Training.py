import torch
from torch import nn
import MiscFunctions
#also yes i know the epoch placement is terrible I will change it later

#I'll add the needed modules and processes later
def trainModel(model: nn.Module,
               train_data: torch.Tensor,
               loss_fn: torch.Module,
               optim,
               epoch: int):
    pass


def testModel(model: nn.module,
              test_data: torch.Tensor,
              loss_fn: nn.Module,
              optim,
              epoch: int): #unsure about which type loss and optim are to so I'll change the input type later
    pass