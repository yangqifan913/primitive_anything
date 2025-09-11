import torch
import torch.nn as nn
from collections.abc import Mapping, Sequence
from torch.utils.data.dataloader import default_collate

def offset2bincount(offset):
    prepend_tensor = torch.tensor([0], device=offset.device, dtype=offset.dtype)
    
    # 确保offset是1维的
    if offset.dim() > 1:
        offset = offset.flatten()
    
    extended_offset = torch.cat((prepend_tensor, offset))
    diff_result = extended_offset[1:] - extended_offset[:-1]
    return diff_result
    
def offset2batch(offset):
    bincount = offset2bincount(offset)
    expanded = torch.arange(bincount.shape[0], device=offset.device).unsqueeze(1).expand(-1, bincount.max()) 
    mask = (torch.arange(bincount.max(), device=offset.device).view(1, -1) < bincount.view(-1, 1))  
    inverse = expanded[mask].flatten() 
    return inverse

def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()

def intersection_and_union(output, target, k, ignore_index=-1):
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def collate_fn(batch):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)
    
class MultiStepWithWarmupLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        warmup_rate=0.05,
        warmup_scale=1e-6,
        last_epoch=-1,
        verbose=False,
    ):
        milestones = [rate * total_steps for rate in milestones]

        def multi_step_with_warmup(s):
            factor = 1.0
            for i in range(len(milestones)):
                if s < milestones[i]:
                    break
                factor *= gamma

            if s <= warmup_rate * total_steps:
                warmup_coefficient = 1 - (1 - s / warmup_rate / total_steps) * (
                    1 - warmup_scale
                )
            else:
                warmup_coefficient = 1.0
            return warmup_coefficient * factor

        super().__init__(
            optimizer=optimizer,
            lr_lambda=multi_step_with_warmup,
            last_epoch=last_epoch,
            verbose=verbose,
        )

class OneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
        verbose=False,
    ):
        super().__init__(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
            verbose=verbose,
        )

class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def build_scheduler(optimizer, steps, CONFIG):
    if CONFIG.optimizer.scheduler.type == "MultiStepWithWarmupLR":
        scheduler = MultiStepWithWarmupLR(
            optimizer,
            CONFIG.optimizer.scheduler.millstones,
            steps,
            CONFIG.optimizer.scheduler.gamma,
            CONFIG.optimizer.scheduler.warmup_rate,
            CONFIG.optimizer.scheduler.warmup_scale
        )
    elif CONFIG.optimizer.scheduler.type == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer,
            CONFIG.optimizer.scheduler.max_lr,
            steps,
            CONFIG.optimizer.scheduler.pct_start,
        )
    else:
        raise ValueError(f"Scheduler type {CONFIG.optimizer.scheduler.type} not supported")

    return scheduler
