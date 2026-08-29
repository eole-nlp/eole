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

### How it works

`reinforce.yaml` fine-tunes an existing checkpoint (`train_from`) with
on-policy REINFORCE:

1. For each prompt in the batch, the **current policy** (the model being trained)
   samples a completion using the configured generation parameters.
2. The completion is scored by the configured `rl_reward_metric` scorer.
3. `reward - batch_mean(reward)` is used as the REINFORCE advantage, which
   reduces variance without needing a learned value function.
4. Optionally, a **KL penalty** against a frozen reference model is added to
   prevent the policy from drifting too far (controlled by `rl_kl_coef`).

### Running training

Adapt `reinforce.yaml` to your paths and then launch with:

```bash
eole train --config recipes/rl/reinforce.yaml
```

You can override any field on the command line:

```bash
eole train --config recipes/rl/reinforce.yaml \
    training.train_from my_model/checkpoint \
    training.model_path my_rl_output/model \
    training.rl_reward_metric BLEU \
    training.rl_kl_coef 0.05
```

### Reward metric (`rl_reward_metric`)

`rl_reward_metric` must be the name of any scorer registered in `eole.scorers`.
Available built-in scorers:

| Scorer name      | Reference-free? | Notes |
|------------------|-----------------|-------|
| `BLEU`           | No (needs `path_tgt`) | Fast lexical metric |
| `CHRF`           | No (needs `path_tgt`) | Character n-gram F-score |
| `TER`            | No (needs `path_tgt`) | Translation edit rate |
| `COMET`          | No (needs `path_tgt`) | Neural MT metric (requires `unbabel-comet`) |
| `COMET-KIWI`     | Yes | Quality-estimation COMET (no references needed) |
| `EOLE-COMET`     | No | COMET model converted with `eole convert COMET` |
| `EOLE-COMET-KIWI`| Yes | QE COMET model converted with `eole convert COMET` |
| `EOLE-XCOMET`    | No | xCOMET model converted with `eole convert COMET` |
| `EOLE-METRICX`   | No | MetricX model converted with `eole convert METRICX` |
| `EOLE-METRICX-QE`| Yes | MetricX QE model (reference-free) |
| `WER`            | No (needs `path_tgt`) | Word error rate |

Reference-free scorers (marked *Yes*) do not require `path_tgt`, but the field
is still needed for batch normalization — point it at a dummy file with one
placeholder token per line if you have no references.

**For real RLHF workloads**, replace `BLEU` with a quality-estimation scorer
such as `COMET-KIWI` or `EOLE-COMET-KIWI` so no reference translations are
required.

### Reference model (`rl_reference_model`) and KL penalty

The **reference model** (`rl_reference_model`) is a *separate frozen checkpoint*
used only to compute a KL-divergence penalty that keeps the policy from
diverging too far from its starting point. It is **not** the reward/scorer model.

Key design questions:

| Question | Answer |
|----------|--------|
| Can the reference model be the same checkpoint as `train_from`? | **Yes — this is the recommended setup.** Point `rl_reference_model` at the same checkpoint you start from. It stays frozen while the policy is updated. |
| Can the reference model be an averaged checkpoint (e.g. SWA)? | **Yes.** Any valid EOLE checkpoint works. Using an averaged checkpoint as the reference can provide a more stable anchor. |
| Can the reference model be the *currently training* model? | **No.** The reference must be a fixed, frozen checkpoint. Using a moving model would defeat the purpose of the KL penalty (it would always be zero). |
| What does `rl_kl_coef: 0` mean? | The KL penalty is disabled entirely; `rl_reference_model` is ignored and no reference model is loaded. Use this when you want pure REINFORCE without any KL regularization. |

```yaml
training:
    rl_kl_coef: 0.05                      # non-zero: reference model is loaded and frozen
    rl_reference_model: base_model/checkpoint  # same as train_from is fine
```

### Key `training:` fields

| Field | Default | Description |
|-------|---------|-------------|
| `rl_algorithm` | — | `reinforce` (required; `dpo`/`grpo`/`ppo` reserved for future use) |
| `rl_reward_metric` | — | Scorer name (required) |
| `rl_baseline` | `batch_mean` | `batch_mean` (subtract batch-mean reward) or `none` |
| `rl_kl_coef` | `0.0` | KL-penalty weight; `0` disables the penalty and the reference model |
| `rl_reference_model` | `null` | Frozen reference checkpoint; required when `rl_kl_coef != 0` |
| `rl_gen_max_length` | `256` | Maximum tokens to generate per rollout |
| `rl_gen_temperature` | `1.0` | Sampling temperature for rollouts |
| `rl_gen_top_k` | `0` | Top-k sampling (`0` = disabled) |
| `rl_gen_top_p` | `0.0` | Nucleus (top-p) sampling (`0` = disabled) |
