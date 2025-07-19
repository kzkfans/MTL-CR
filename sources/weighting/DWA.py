import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .abstract_weighting import AbsWeighting

class DWA(AbsWeighting):
    r"""Dynamic Weight Average (DWA).
    
    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_. 

    Args:
        T (float, default=2.0): The softmax temperature.

    """
    def __init__(self, task_num, device=None):
        super(DWA, self).__init__()
        self.T = None
        self.task_num = task_num  # 任务数量
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_param(self):
        self.T=2.0
    def backward(self, losses, **kwargs):
        if self.epoch > 1:
            w_i = torch.Tensor(self.train_loss_buffer[:,self.epoch-1]/self.train_loss_buffer[:,self.epoch-2]).to(self.device)
            batch_weight = self.task_num*F.softmax(w_i/self.T, dim=-1)
        else:
            batch_weight = torch.ones_like(losses).to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        # loss.backward()
        weights = batch_weight.detach().cpu().numpy()
        # print('loss', loss)
        # print(weights)
        return loss, weights