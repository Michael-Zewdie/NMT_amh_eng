from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model.utils.config import Config


def build_optimizer(model, cfg: Config) -> AdamW:
    # lr=1.0 is a placeholder — the actual learning rate at each step comes
    # entirely from the scheduler below (LambdaLR multiplies this base lr by
    # its lr_lambda return value).
    return AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)


def build_scheduler(optimizer: AdamW, cfg: Config) -> LambdaLR:
    """Inverse-sqrt warmup schedule (Vaswani et al. 2017):

        lr(step) = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)

    Requires the optimizer to have been built with lr=1.0 via build_optimizer.
    Under gradient accumulation, call scheduler.step() once per OPTIMIZER
    update, not per micro-batch — a classic bug source.
    """
    d_model = cfg.model.d_model
    warmup_steps = cfg.training.warmup_steps

    def lr_lambda(step: int) -> float:
        step = step + 1  # LambdaLR invokes this with step starting at 0
        return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
