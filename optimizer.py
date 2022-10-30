"""Optimizer."""
import torch


def construct_optimizer(optim_params, cfg):
    lr = cfg['base_lr']
    wd = cfg['weight_decay']
    if cfg['optimizing_method'] == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=lr,
            momentum=cfg['momentum'],
            weight_decay=wd,
        )
    elif cfg['optimizing_method'] == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=wd,
        )
    elif cfg['optimizing_method'] == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=wd)
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg['optimizing_method'])
        )


# def update_lr(optimizer, epoch, cfg):
#     # Update the learning rate.
#     lr = timesformer.models.optimizer.get_epoch_lr(epoch, cfg)
#     timesformer.models.optimizer.set_lr(optimizer, lr)
