import torch
from torch import nn


def l2_loss(sim, labels):
    return nn.MSELoss()(sim, labels)


def bce_loss(sim, labels, target=1.0):
    t1, t2 = 1 - target, target
    if target != 1.0:
        loss_array_1 = - t2 * torch.log(sim) - t1 * torch.log(1-sim)
        loss_array_2 = - t2 * torch.log(1-sim) - t1 * torch.log(sim)
        labels = labels.float()
        loss_vec = labels * loss_array_1 + (1 - labels) * loss_array_2

        loss = torch.mean(loss_vec)

        # loss = torch.mean(loss, 0)
    else:
        loss = nn.BCELoss()(sim, labels)
    return loss


def bpr_loss(sim, labels, p_index, n_index, bpr_margin=0.0):
    num = labels.shape[0]
    p_num = int(labels.sum(0).detach().cpu().numpy())
    num_neg = int(num / p_num) - 1

    p_sim = sim[p_index].reshape(p_num, 1).expand(p_num, num_neg).reshape(-1)
    n_sim = sim[n_index]
    loss_vec = torch.log(nn.Sigmoid()(p_sim - n_sim - bpr_margin))

    loss = -torch.mean(loss_vec)
    return loss


def margin_loss(sim, labels, p_index, n_index, margin=0.5):
    num = labels.shape[0]
    p_num = int(labels.sum(0).detach().cpu().numpy())
    num_neg = int(num / p_num) - 1

    p_sim = sim[p_index].reshape(p_num, 1).expand(p_num, num_neg).reshape(-1)
    n_sim = sim[n_index]

    loss_vec = torch.relu(n_sim + margin - p_sim)

    loss = torch.mean(loss_vec)
    return loss
 

