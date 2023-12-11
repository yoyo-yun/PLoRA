import torch.optim as Optim
from transformers import AdamW
from transformers import (get_linear_schedule_with_warmup,
                          get_constant_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)


def get_Adam_optim(__C, model):
    return Optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=__C.TRAIN.lr_base,
        betas=__C.TRAIN.opt_betas,
        eps=__C.TRAIN.opt_eps
    ), None


def get_AdamW_optim(__C, model):
    return Optim.AdamW(
        # params=filter(lambda p: p.requires_grad, model.parameters()),
        params=model.parameters(),
        lr=__C.TRAIN.lr_base,
        # betas=__C.TRAIN.opt_betas,
        # eps=__C.TRAIN.opt_eps
    ), None


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r


def get_Adam_optim_constant(config, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.TRAIN.lr_base, weight_decay=0.01, correct_bias=False)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=config.num_train_optimization_steps,
    #                                  num_warmup_steps=config.warmup_proportion * config.num_train_optimization_steps)
    scheduler = get_constant_schedule_with_warmup(optimizer,
                    num_warmup_steps=config.warmup_proportion * config.num_train_optimization_steps)
    return optimizer, scheduler


def get_Adam_optim_linear(config, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.TRAIN.lr_base, weight_decay=0.01, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=config.num_train_optimization_steps,
                                     num_warmup_steps=config.warmup_proportion * config.num_train_optimization_steps)
    return optimizer, scheduler


def get_AdamW_cosine_with_hard_restarts(config, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.TRAIN.lr_base, weight_decay=0.01, correct_bias=False)

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
        num_warmup_steps=config.warmup_proportion * config.num_train_optimization_steps,
        num_training_steps=config.num_train_optimization_steps,
        num_cycles=3)
    return optimizer, scheduler

