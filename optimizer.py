import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
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
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias
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

                m = 0
                v = 0
                t = 0

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                epsilon = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]
                
                grad = state
                while(grad not converged):
                    t += 1
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * grad**2
                    a = alpha * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                    theta = theta - a * m / (v.sqrt() + epsilon) + weight_decay * theta
                    grad -= alpha * m_hat / (v_hat.sqrt() + epsilon)


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

                device = "cuda" if torch.cuda.is_available() else "cpu"

                if "t" not in state.keys():
                    state["t"] = 0
                    state["mt"] = torch.zeros(p.data.shape).to(device)
                    state["vt"] = torch.zeros(p.data.shape).to(device)
                beta1, beta2 = group["betas"]

                state["t"] = state["t"] + 1
                state["mt"] = state["mt"].mul(beta1) + grad.mul(1 - beta1)
                state["vt"] = state["vt"].mul(beta2) + torch.mul(grad, grad).mul(1 - beta2)

                state["alpha"] = (group["lr"] * math.sqrt(1 - beta2 ** state["t"])) / (1 - beta1 ** state["t"])

                p.data = p.data.sub(state["mt"].div(torch.sqrt(state["vt"]).add_(group["eps"])).mul(state["alpha"]))
                p.data = p.data.sub(p.data.mul(group["weight_decay"]))

        return loss
