# flake8: noqa
import torch
import eole.modules.rmsnorm as rmsnorm_module
from eole.modules.rmsnorm import RMSNorm, GemmaRMSNorm


"""
Expected results

[ Results in Milliseconds (ms) | Seq: 1024 | torch.bfloat16 ]

           |                                    RMSNorm                                     |                                  GemmaRMSNorm                                 
Hidden     |          BS: 1           |          BS: 8           |          BS: 16          |          BS: 1           |          BS: 8           |          BS: 16          | 
Size       |   Eager Compile Eole_ops |   Eager Compile Eole_ops |   Eager Compile Eole_ops |   Eager Compile Eole_ops |   Eager Compile Eole_ops |   Eager Compile Eole_ops | 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
512        |   0.062   0.061   0.015  |   0.062   0.068   0.016  |   0.101   0.068   0.021  |   0.076   0.061   0.015  |   0.075   0.067   0.015  |   0.093   0.068   0.021  | 
1024       |   0.063   0.062   0.015  |   0.099   0.066   0.025  |   0.238   0.068   0.045  |   0.076   0.062   0.015  |   0.093   0.068   0.025  |   0.287   0.067   0.045  | 
4096       |   0.063   0.061   0.017  |   0.749   0.073   0.093  |   1.581   0.179   0.179  |   0.076   0.058   0.022  |   0.863   0.075   0.139  |   1.768   0.179   0.274  | 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""


def get_avg_ms(model_instance, input_tensor, n_warmup=25, n_iter=100):
    with torch.no_grad():
        for _ in range(n_warmup):
            model_instance(input_tensor)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(n_iter):
            model_instance(input_tensor)
        end.record()
        torch.cuda.synchronize()
        # Returns milliseconds
        return start.elapsed_time(end) / n_iter


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    BATCH_SIZES = [1, 8, 16]
    HIDDEN_SIZES = [512, 1024, 4096]
    SEQ_LEN = 1024

    # Define column widths for perfect alignment
    label_w = 10
    val_w = 7  # Width for "0.000"
    block_inner_w = (val_w * 3) + 3  # width of "Eag Cmp Eol"
    sep = " | "

    print(f"\n[ Results in Milliseconds (ms) | Seq: {SEQ_LEN} | {DTYPE} ]\n")

    # 1. Model Header
    m_header = f"{'':<{label_w}}{sep}{'RMSNorm':^{block_inner_w*3 + 6}}{sep}{'GemmaRMSNorm':^{block_inner_w*3 + 6}}"
    print(m_header)

    # 2. Batch Size Header
    bs_row = f"{'Hidden':<{label_w}}{sep}"
    for _ in range(2):  # For both models
        for b in BATCH_SIZES:
            bs_row += f"{'BS: ' + str(b):^{block_inner_w}}{sep}"
    print(bs_row)

    # 3. Method Header
    meth_row = f"{'Size':<{label_w}}{sep}"
    for _ in range(6):  # 3 batch sizes * 2 models
        meth_row += f"{'Eager':>{val_w}} {'Compile':>{val_w}} {'Eole_ops':>{val_w}}{sep}"
    print(meth_row)
    print("-" * len(meth_row))

    for h in HIDDEN_SIZES:
        row = f"{h:<{label_w}}{sep}"

        for model_cls in [RMSNorm, GemmaRMSNorm]:
            for b in BATCH_SIZES:
                x = torch.randn(b, SEQ_LEN, h, device=DEVICE, dtype=DTYPE)

                # Baseline Eager
                rmsnorm_module._eole_ops = False
                m_eager = model_cls(h, eps=1e-6).to(DEVICE, DTYPE)
                t_eager = get_avg_ms(m_eager, x)

                # Compile
                m_comp = torch.compile(model_cls(h, eps=1e-6).to(DEVICE, DTYPE))
                t_comp = get_avg_ms(m_comp, x)

                # Eole
                rmsnorm_module._eole_ops = True
                m_eole = model_cls(h, eps=1e-6).to(DEVICE, DTYPE)
                t_eole = get_avg_ms(m_eole, x)

                row += f"{t_eager:>{val_w}.3f} {t_comp:>{val_w}.3f} {t_eole:>{val_w}.3f} {sep}"
        print(row)

    print("-" * len(meth_row))
