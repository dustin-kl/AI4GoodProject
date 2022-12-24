import torch


def iou(y, y_hat):
    epsilon = 1e-6
    max_idx = torch.argmax(y_hat, 1, keepdim=True)
    one_hot = torch.FloatTensor(y_hat.shape).to(torch.device(y.device))
    one_hot.zero_()
    one_hot.scatter_(1, max_idx, 1)

    union = one_hot + y
    union = union.to(torch.bool).to(torch.float32)

    intersection = one_hot * y

    bg_inters = torch.sum(torch.clone(intersection[:, 0]), (1, 2))
    bg_unions = torch.sum(torch.clone(union[:, 0]), (1, 2))
    tc_inters = torch.sum(torch.clone(intersection[:, 1]), (1, 2))
    tc_unions = torch.sum(torch.clone(union[:, 1]), (1, 2))
    ar_inters = torch.sum(torch.clone(intersection[:, 2]), (1, 2))
    ar_unions = torch.sum(torch.clone(union[:, 2]), (1, 2))

    bg_iou = torch.nanmean(torch.div(bg_inters + epsilon, bg_unions + epsilon))
    tc_iou = torch.nanmean(torch.div(tc_inters + epsilon, tc_unions + epsilon))
    ar_iou = torch.nanmean(torch.div(ar_inters + epsilon, ar_unions + epsilon))

    return bg_iou, tc_iou, ar_iou
