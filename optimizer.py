import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # State should be stored in this dictionary
                state = self.state[p]
                alpha = group["lr"]
                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, and correct_bias, as saved in
                # the constructor).
                #
                # 1- Update first and second moments of the gradients.
                # 2- Apply bias correction.
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given as the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                ### TODO
                # raise NotImplementedError
                device = "cuda" if torch.cuda.is_available() else "cpu"

                if "t" not in state.keys():
                    state["t"] = 0
                    state["mt"] = torch.zeros(p.data.shape).to(device)
                    state["vt"] = torch.zeros(p.data.shape).to(device)

                t = state["t"]
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                state["t"] = t + 1
                state["mt"] = beta1 * state["mt"] + (1 - beta1) * grad
                state["vt"] = beta2 * state["vt"] + (1 - beta2) * (
                    torch.mul(grad, grad)
                )

                state["alpha"] = (lr * math.sqrt(1 - beta2 ** state["t"])) / (
                    1 - beta1 ** state["t"]
                )
                p.data = p.data - state["alpha"] * (
                    state["mt"] / (torch.sqrt(state["vt"]) + group["eps"])
                )
                p.data = p.data - (lr * group["weight_decay"] * p.data)

        return loss


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


SCHEDULES = {
    "warmup_cosine": warmup_cosine,
    "warmup_constant": warmup_constant,
    "warmup_linear": warmup_linear,
}


class AdamWarmup(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(
        self,
        params,
        lr,
        warmup=-1,
        t_total=-1,
        schedule="warmup_linear",
        b1=0.9,
        b2=0.999,
        e=1e-8,
        weight_decay_rate=0.01,
        max_grad_norm=1.0,
    ):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(
                "Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup)
            )
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                "Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1)
            )
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                "Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2)
            )
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(
            lr=lr,
            schedule=schedule,
            warmup=warmup,
            t_total=t_total,
            b1=b1,
            b2=b2,
            e=e,
            weight_decay_rate=weight_decay_rate,
            max_grad_norm=max_grad_norm,
        )
        super(AdamWarmup, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        print("l_total=", len(self.param_groups))
        for group in self.param_groups:
            print("l_p=", len(group["params"]))
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group["t_total"] != -1:
                    schedule_fct = SCHEDULES[group["schedule"]]
                    lr_scheduled = group["lr"] * schedule_fct(
                        state["step"] / group["t_total"], group["warmup"]
                    )
                else:
                    lr_scheduled = group["lr"]
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """Move the optimizer state to a specified device"""
        for state in self.state.values():
            state["exp_avg"].to(device)
            state["exp_avg_sq"].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # State initialization
                state["step"] = initial_step
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["next_m"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["next_v"] = torch.zeros_like(p.data)

                next_m, next_v = state["next_m"], state["next_v"]
                beta1, beta2 = group["b1"], group["b2"]

                # Add grad clipping
                if group["max_grad_norm"] > 0:
                    clip_grad_norm_(p, group["max_grad_norm"])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group["e"])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group["weight_decay_rate"] > 0.0:
                    update += group["weight_decay_rate"] * p.data

                if group["t_total"] != -1:
                    schedule_fct = SCHEDULES[group["schedule"]]
                    lr_scheduled = group["lr"] * schedule_fct(
                        state["step"] / group["t_total"], group["warmup"]
                    )
                else:
                    lr_scheduled = group["lr"]

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state["step"] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss
