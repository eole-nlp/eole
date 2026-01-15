# flake8: noqa
import torch
import eole.modules.rope as rope_module
from eole.modules.rope import apply_rotary_emb

"""
Expected results
[ apply_rotary_emb Benchmark (ms) | torch.bfloat16 | heads=32 | D=128 | Rd=64 ]

             |                                Rotary Embedding                               
SeqLen       |          BS: 1           |          BS: 8           |          BS: 16          | 
             |   Eager Compile Eole_ops |   Eager Compile Eole_ops |   Eager Compile Eole_ops | 
------------------------------------------------------------------------------------------------
1            |   0.114   0.060   0.029  |   0.114   0.076   0.028  |   0.115   0.076   0.025  | 
128          |   0.114   0.070   0.029  |   0.112   0.080   0.027  |   0.157   0.079   0.028  | 
256          |   0.116   0.069   0.029  |   0.155   0.079   0.027  |   0.279   0.084   0.026  | 
512          |   0.117   0.070   0.029  |   0.276   0.082   0.027  |   0.576   0.178   0.041  | 
1024         |   0.113   0.069   0.031  |   0.573   0.177   0.041  |   1.305   0.359   0.174  | 
------------------------------------------------------------------------------------------------
"""


def get_avg_ms(fn, n_warmup=25, n_iter=100):
    with torch.no_grad():
        for _ in range(n_warmup):
            fn()

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(n_iter):
            fn()
        end.record()

        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_iter


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    BATCH_SIZES = [1, 8, 16]
    SEQ_LENS = [1, 128, 256, 512, 1024]
    HEADS = 32
    HEAD_DIM = 128  # D
    ROTARY_DIM = 64  # Rd
    INTERLEAVE = False

    # Formatting
    label_w = 12
    val_w = 7
    block_inner_w = (val_w * 3) + 3
    sep = " | "

    print(f"\n[ apply_rotary_emb Benchmark (ms) | {DTYPE} | heads={HEADS} | D={HEAD_DIM} | Rd={ROTARY_DIM} ]\n")

    # Model header
    m_header = f"{'':<{label_w}}{sep}" f"{'Rotary Embedding':^{block_inner_w*3 + 6}}"
    print(m_header)

    # Batch size header
    bs_row = f"{'SeqLen':<{label_w}}{sep}"
    for b in BATCH_SIZES:
        bs_row += f"{'BS: ' + str(b):^{block_inner_w}}{sep}"
    print(bs_row)

    # Method header
    meth_row = f"{'':<{label_w}}{sep}"
    for _ in BATCH_SIZES:
        meth_row += f"{'Eager':>{val_w}} {'Compile':>{val_w}} {'Eole_ops':>{val_w}}{sep}"
    print(meth_row)
    print("-" * len(meth_row))

    for S in SEQ_LENS:
        row = f"{S:<{label_w}}{sep}"

        for B in BATCH_SIZES:
            query = torch.randn(B, S, HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
            key = torch.randn(B, S, HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)

            # cos_sin shape: [S, Rd]
            cos = torch.randn(S, ROTARY_DIM // 2, device=DEVICE, dtype=DTYPE)
            sin = torch.randn(S, ROTARY_DIM // 2, device=DEVICE, dtype=DTYPE)
            cos_sin = torch.cat([cos, sin], dim=-1)

            # ---------- EAGER ----------
            rope_module._eole_ops = False

            def eager_fn():
                apply_rotary_emb(query, key, cos_sin, INTERLEAVE)

            t_eager = get_avg_ms(eager_fn)

            # ---------- COMPILE ----------
            rope_module._eole_ops = False
            compiled_fn = torch.compile(lambda q, k: apply_rotary_emb(q, k, cos_sin, INTERLEAVE))

            def comp_fn():
                compiled_fn(query, key)

            t_comp = get_avg_ms(comp_fn)

            # ---------- EOLE ----------
            rope_module._eole_ops = True

            def eole_fn():
                apply_rotary_emb(query, key, cos_sin, INTERLEAVE)

            t_eole = get_avg_ms(eole_fn)

            row += f"{t_eager:>{val_w}.3f} {t_comp:>{val_w}.3f} {t_eole:>{val_w}.3f} {sep}"

        print(row)

    print("-" * len(meth_row))
