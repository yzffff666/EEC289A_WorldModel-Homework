"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    if bool(getattr(model, "use_gru", False)):
        hidden = model.initial_hidden(states.shape[0], states.device)
        losses = []
        for t in range(actions.shape[1]):
            obs = states[:, t]
            act = actions[:, t]
            target_delta = states[:, t + 1] - states[:, t]
            obs_norm = normalizer.normalize_obs(obs)
            act_norm = normalizer.normalize_act(act)
            target_norm = normalizer.normalize_delta(target_delta)
            pred_norm, hidden = model(obs_norm, act_norm, hidden)
            losses.append(F.mse_loss(pred_norm, target_norm, reduction="none").mean(dim=1))
        return torch.stack(losses, dim=1).mean()

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
    hard_fraction: float = 0.0,
    hard_weight: float = 0.0,
    vpt_focus_horizon: int = 0,
    vpt_margin: float = 0.25,
    vpt_weight: float = 0.0,
    vpt_hard_fraction: float = 0.0,
    vpt_hard_weight: float = 0.0,
    stability_bound: float = 0.0,
    stability_weight: float = 0.0,
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
    per_window_step = (pred_norm - target_norm).square().mean(dim=2)
    raw_per_window_step = per_window_step
    if float(tail_weight) > 1.0 and int(horizon) > 1:
        weights = torch.linspace(1.0, float(tail_weight), int(horizon), device=states.device)
        weights = weights / weights.mean()
        per_window_step = per_window_step * weights.unsqueeze(0)
    per_window = per_window_step.mean(dim=1)
    loss = per_window.mean()
    if float(hard_weight) > 0.0 and float(hard_fraction) > 0.0:
        k = max(1, int(per_window.shape[0] * float(hard_fraction)))
        hard = torch.topk(per_window, k=k, largest=True).values.mean()
        loss = loss + float(hard_weight) * hard
    if float(vpt_weight) > 0.0 and int(vpt_focus_horizon) > 0:
        focus = raw_per_window_step[:, : min(int(vpt_focus_horizon), int(horizon))]
        threshold_penalty = F.relu(focus - float(vpt_margin)).mean()
        loss = loss + float(vpt_weight) * threshold_penalty
        if float(vpt_hard_weight) > 0.0 and float(vpt_hard_fraction) > 0.0:
            per_window_excess = F.relu(focus - float(vpt_margin)).mean(dim=1)
            k = max(1, int(per_window_excess.shape[0] * float(vpt_hard_fraction)))
            hard_excess = torch.topk(per_window_excess, k=k, largest=True).values.mean()
            loss = loss + float(vpt_hard_weight) * hard_excess
    if float(stability_weight) > 0.0 and float(stability_bound) > 0.0:
        excess = F.relu(pred_norm.abs() - float(stability_bound))
        loss = loss + float(stability_weight) * excess.square().mean()
    return loss


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    one = one_step_delta_loss(model, states, actions, normalizer)
    horizon = int(loss_cfg.get("rollout_train_horizon", 5))
    warmup = int(cfg["eval"].get("warmup_steps", 5))
    tail_weight = float(loss_cfg.get("rollout_tail_weight", 1.0))
    hard_fraction = float(loss_cfg.get("rollout_hard_fraction", 0.0))
    hard_weight = float(loss_cfg.get("rollout_hard_weight", 0.0))
    vpt_focus_horizon = int(loss_cfg.get("vpt_focus_horizon", 0))
    vpt_margin = float(loss_cfg.get("vpt_margin", 0.25))
    vpt_weight = float(loss_cfg.get("vpt_weight", 0.0))
    vpt_hard_fraction = float(loss_cfg.get("vpt_hard_fraction", 0.0))
    vpt_hard_weight = float(loss_cfg.get("vpt_hard_weight", 0.0))
    stability_bound = float(loss_cfg.get("stability_bound", 0.0))
    stability_weight = float(loss_cfg.get("stability_weight", 0.0))
    roll = rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=horizon,
        tail_weight=tail_weight,
        hard_fraction=hard_fraction,
        hard_weight=hard_weight,
        vpt_focus_horizon=vpt_focus_horizon,
        vpt_margin=vpt_margin,
        vpt_weight=vpt_weight,
        vpt_hard_fraction=vpt_hard_fraction,
        vpt_hard_weight=vpt_hard_weight,
        stability_bound=stability_bound,
        stability_weight=stability_weight,
    )
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + float(loss_cfg.get("rollout_weight", 0.3)) * roll
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
