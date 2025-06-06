import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict


try:
    from itertools import ifilterfalse
except ImportError: # py3k
    from itertools import filterfalse as ifilterfalse

# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def dice_loss(input_mask, cls_gt, ignore=255):
    num_objects = input_mask.shape[1]
    losses = []
    cls_gt = cls_gt.contiguous().view(-1)   # [bs*H*W]
    valid = (cls_gt != ignore)
    B, C, H, W = input_mask.size()
    input_mask = input_mask.permute(0, 2, 3, 1).contiguous().view(-1, num_objects)  # [bs*H*W, num_objects]
    input_mask = input_mask[valid.view(-1, 1).expand(-1, num_objects)].reshape(-1, num_objects)  # [w/o ignore_region, num_objects]
    cls_gt = cls_gt[valid]  # [w/o ignore_region,]
    for i in range(num_objects):
        mask = input_mask[:,i]#.flatten(start_dim=1)  # [w/o ignore_region,]
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt==(i+1)).float()#.flatten(start_dim=1)  # [w/o ignore_region,]
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return mean(losses) # return torch.cat(losses).mean()


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15, ignore_index=255):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p
        self.ignore_index = ignore_index

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target, ignore_index=self.ignore_index), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index).view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class LossComputer_VOST:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce = BootstrappedCE(config['start_warm'], config['end_warm'])

    def compute(self, data, num_objects, it):
        losses = defaultdict(int)

        b, t = data['rgb'].shape[:2]

        losses['total_loss'] = 0
        for ti in range(1, t):
            for bi in range(b):
                loss, p = self.bce(data[f'logits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,ti,0], it)
                losses['p'] += p / b / (t-1)
                losses[f'ce_loss_{ti}'] += loss / b

            losses['total_loss'] += losses['ce_loss_%d'%ti]
            losses[f'dice_loss_{ti}'] = dice_loss(data[f'masks_{ti}'], data['cls_gt'][:,ti,0])
            losses['total_loss'] += losses[f'dice_loss_{ti}']

        return losses
