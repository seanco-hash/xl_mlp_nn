"""Optimizer."""
import torch


def construct_optimizer(optim_params, cfg):
    lr = cfg['SOLVER']['BASE_LR']
    wd = cfg['SOLVER']['WEIGHT_DECAY']
    if cfg['SOLVER']['OPTIMIZING_METHOD'] == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=lr,
            momentum=cfg['SOLVER']['MOMENTUM'],
            weight_decay=wd,
        )
    elif cfg['SOLVER']['OPTIMIZING_METHOD'] == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=wd,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg['SOLVER']['OPTIMIZING_METHOD'])
        )


# def update_lr(optimizer, epoch, cfg):
#     # Update the learning rate.
#     lr = timesformer.models.optimizer.get_epoch_lr(epoch, cfg)
#     timesformer.models.optimizer.set_lr(optimizer, lr)
