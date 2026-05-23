"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    tail_weight: float = 1.0,
) -> torch.Tensor:
    # Train local open-loop stability at random positions, not only at the
    # beginning of each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    per_step = (pred_norm - target_norm).square().mean(dim=(0, 2))
    if float(tail_weight) <= 1.0 or int(horizon) <= 1:
        return per_step.mean()
    weights = torch.linspace(1.0, float(tail_weight), int(horizon), device=states.device)
    weights = weights / weights.mean()
    return (per_step * weights).mean()


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    one = one_step_delta_loss(model, states, actions, normalizer)
    horizon = int(loss_cfg.get("rollout_train_horizon", 5))
    warmup = int(cfg["eval"].get("warmup_steps", 5))
    tail_weight = float(loss_cfg.get("rollout_tail_weight", 1.0))
    roll = rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=horizon,
        tail_weight=tail_weight,
    )
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + float(loss_cfg.get("rollout_weight", 0.3)) * roll
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
