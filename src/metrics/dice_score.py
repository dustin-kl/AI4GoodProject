import torch
from torch import Tensor
import torch as nn

import numpy as np


def dice_coeff(
    input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f"Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})"
        )

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(
    input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
):
    # Average of Dice coefficient for all classes
    weights = torch.Tensor([0.35, 70, 5.9]).cuda()
    # weights = torch.Tensor([1, 1, 1]).cuda()
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += weights[channel] * dice_coeff(
            input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon
        )

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def generalized_dice_loss(input, target, weights, epsilon=1e-6):

    loss = 0
    for idx in range(input.shape[0]):
        nominator = 0
        denominator = 0
        for channel in range(input.shape[1]):
            input_sample = input[idx, channel, ...]
            target_sample = target[idx, channel, ...]

            inter = torch.dot(input_sample.reshape(-1), target_sample.reshape(-1))
            sets_sum = torch.sum(input_sample) + torch.sum(target_sample)

            nominator += 2 * weights[channel] * inter + epsilon
            denominator += weights[channel] * sets_sum + epsilon
        loss += 1 - (nominator / denominator)

    return loss / input.shape[0]

def calc_dsc_sets(truth, pred, c=1):
    # Obtain sets with associated class
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    # Calculate Dice
    if (pd.sum() + gt.sum()) != 0:
        dice = 2*np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum())
    else : dice = 0.0
    # Return computed Dice
    return dice

def dice(y_hat, y):
    max_idx = torch.argmax(y_hat, 1, keepdim=True)
    one_hot = torch.FloatTensor(y_hat.shape).to(torch.device(y.device))
    one_hot.zero_()
    one_hot.scatter_(1, max_idx, 1)

    scores = []
    for c in range(3):
        scores.append(calc_dsc_sets(y[:, c].cpu().numpy(), one_hot[:, c].cpu().numpy()))
    scores = np.array(scores)
    scores = torch.from_numpy(scores).to(torch.device(y.device))

    return torch.mean(scores)
    


# dice_loss(F.softmax(y_hat, dim=1).float(),
#              y.float(), # permute(0, 3, 1, 2)
#              multiclass=True) # weights = ([0.1, 2, 1])

# def focal_loss(input, target, gamma):

#     log_p = F.log_softmax(input, dim=-1)
#     nlloss = nn.NLLLoss(weight=alpha, reduction='none', ignore_index=ignore_index)
#     ce = (log_p, target)

#     # get true class column from each row
#     all_rows = torch.arange(len(x))
#     log_pt = log_p[all_rows, y]

#     # compute focal term: (1 - pt)^gamma
#     pt = log_pt.exp()
#     focal_term = (1 - pt)**gamma

#     # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
#     loss = focal_term * ce

#     if self.reduction == 'mean':
#         loss = loss.mean()
#     elif self.reduction == 'sum':
#         loss = loss.sum()

#     return loss
