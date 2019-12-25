import torch
from torch import nn


def l2_loss(sim, labels):
    return nn.MSELoss()(sim, labels)


def bce_loss(sim, labels):
    loss = nn.BCEWithLogitsLoss()(sim, labels)
    return loss


def generic_ranking_loss(sim, labels, p_index, n_index):
    batch_size = len(labels)
    p_num = len(p_index.view(-1))
    num_neg = int(batch_size / p_num) - 1

    p_sim = sim[p_index].view(-1).expand(num_neg, p_num).reshape(-1)
    n_sim = sim[n_index].view(-1)

    return p_sim, n_sim


def bpr_loss(sim, labels, p_index, n_index, bpr_margin=0.0):
    p_sim, n_sim = generic_ranking_loss(sim, labels, p_index, n_index)
    loss_vec = torch.log(nn.Sigmoid()(p_sim - n_sim - bpr_margin))
    loss = -torch.mean(loss_vec)
    return loss


def margin_loss(sim, labels, p_index, n_index, margin=0.5):
    p_sim, n_sim = generic_ranking_loss(sim, labels, p_index, n_index)
    loss_vec = torch.relu(n_sim + margin - p_sim)
    loss = torch.mean(loss_vec)
    return loss

 

