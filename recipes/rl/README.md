# RL fine-tuning recipes

This folder contains example configurations for RL fine-tuning methods built on
top of the "RL readiness" refactoring (`ModelOutput`, `BaseModel.compute_log_probs()`,
`eole.utils.generate_utils`, the `Trainer._train_step()` extension point).

## Status

| Algorithm | Status |
|-----------|--------|
| REINFORCE | Implemented (`reinforce.yaml`) |
| DPO       | Planned (needs a paired chosen/rejected preference data pipeline) |
| GRPO      | Planned (group-relative baseline over several rollouts per prompt) |
| PPO       | Planned (clipped surrogate objective + value model) |

Setting `training.rl_algorithm` to `dpo`, `grpo`, or `ppo` currently raises a
`NotImplementedError` — these names are reserved so existing configs keep
working once those algorithms land.

## REINFORCE

See `reinforce.yaml`. It fine-tunes an existing checkpoint (`train_from`) with
on-policy REINFORCE:

1. sample a completion from the current policy for each prompt in the batch;
2. score it with `rl_reward_metric` (any scorer registered in `eole.scorers`);
3. use `reward - batch_mean(reward)` as the advantage in the standard
   score-function policy-gradient loss, optionally regularized with a KL
   penalty against a frozen `rl_reference_model`.

Key `training:` fields:
- `rl_algorithm`: `reinforce` (only supported value today).
- `rl_reward_metric`: name of the reward scorer to use.
- `rl_baseline`: `batch_mean` (default, variance reduction) or `none`.
- `rl_kl_coef`: weight of the KL penalty vs. `rl_reference_model` (0 disables it).
- `rl_reference_model`: frozen checkpoint used for the KL penalty.
- `rl_gen_max_length` / `rl_gen_temperature` / `rl_gen_top_k` / `rl_gen_top_p`:
  sampling parameters for the rollouts.
