import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_kd(all_out, teacher_all_out, outputs, labels, alpha, temperature):
    """
    loss function for Knowledge Distillation (KD)
    """

    T = temperature

    loss_CE = F.cross_entropy(outputs, labels)
    # D_KL = nn.KLDivLoss()(F.log_softmax(all_out/T, dim=1), F.softmax(teacher_all_out/T, dim=1)) * (T * T)

    # PyTorch Attention !
    # reduction = 'mean' doesn’t return the true kl divergence value, please use reduction = 'batchmean' 
    # which aligns with KL math definition. In the next major release, 'mean' will be changed to be the same as ‘batchmean’.
    D_KL = F.kl_div(F.log_softmax(all_out / T, dim=1), F.softmax(teacher_all_out / T, dim=1), reduction='batchmean') * (T * T)
    KD_loss =  (1. - alpha) * loss_CE + alpha * D_KL

    return KD_loss


def loss_kd_only(all_out, teacher_all_out, temperature):

    T = temperature

    # D_KL = nn.KLDivLoss()(F.log_softmax(all_out/T, dim=1), F.softmax(teacher_all_out/T, dim=1)) * (T * T)
    D_KL = F.kl_div(F.log_softmax(all_out / T, dim=1), F.softmax(teacher_all_out / T, dim=1), reduction='batchmean') * (T * T)

    return D_KL
