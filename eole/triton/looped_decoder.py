"""Fused pointwise operations for single-token Inkling 2 Mini decode."""

import triton
import triton.language as tl


@triton.jit
def _rms(value, width: tl.constexpr, eps: tl.constexpr):
    return value * tl.rsqrt(tl.sum(value * value, axis=0) / width + eps)


@triton.jit
def _gate_features(X, MESSAGE, OUT, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + i, i < H, 0).to(tl.float32)
    message = tl.load(MESSAGE + row * H + i, i < H, 0).to(tl.float32)
    tl.store(OUT + row * 2 * H + i, _rms(x, H, EPS), i < H)
    tl.store(OUT + row * 2 * H + H + i, _rms(message, H, EPS), i < H)


@triton.jit
def _gate_output(
    X,
    MESSAGE,
    GATES,
    NORM,
    OUT,
    H: tl.constexpr,
    NORM_STRIDE: tl.constexpr,
    TOKENS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + i, i < H, 0).to(tl.float32)
    message = tl.load(MESSAGE + row * H + i, i < H, 0).to(tl.float32)
    forget = tl.extra.cuda.libdevice.tanh(tl.load(GATES + row * 2 * H + i, i < H, 0))
    gain = tl.extra.cuda.libdevice.tanh(tl.load(GATES + row * 2 * H + H + i, i < H, 0))
    mixed = (1 - forget) * x + gain * message
    norm = tl.load(NORM + row // TOKENS * NORM_STRIDE + i, i < H, 0)
    tl.store(OUT + row * H + i, _rms(mixed, H, EPS) * norm, i < H)


def gate_features(x, message, dtype, eps):
    rows, width = x.shape
    output = x.new_empty((rows, 2 * width), dtype=dtype)
    _gate_features[(rows,)](x, message, output, width, eps, triton.next_power_of_2(width), num_warps=4)
    return output


def gate_output(x, message, gates, norm, dtype, eps):
    rows, width = x.shape
    output = x.new_empty((rows, width), dtype=dtype)
    _gate_output[(rows,)](
        x,
        message,
        gates,
        norm,
        output,
        width,
        norm.stride(0),
        rows // norm.size(0),
        eps,
        triton.next_power_of_2(width),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _conv_step(
    X,
    HISTORY,
    TAPS,
    DYNAMIC,
    OUT,
    C: tl.constexpr,
    K: tl.constexpr,
    CHANNELS_PER_HEAD: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < C
    value = tl.load(X + i, valid, 0).to(tl.float32)
    result = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(K):
        if tap < K - 1:
            current = tl.load(HISTORY + i * (K - 1) + tap, valid, 0)
        else:
            current = value
        weight = tl.load(TAPS + i * K + tap, valid, 0)
        if CHANNELS_PER_HEAD:
            weight += tl.load(DYNAMIC + i // CHANNELS_PER_HEAD * K + K - 1 - tap, valid, 0).to(tl.float32)
            result += current * weight
        else:
            result = tl.fma(current, weight, result)
        # Ascending tap order consumes each value before shifting it left.
        if tap:
            tl.store(HISTORY + i * (K - 1) + tap - 1, current, valid)
    tl.store(OUT + i, (result + value) * SCALE, valid)


def conv_step(x, history, taps, dynamic, heads, scale):
    channels, width = taps.shape
    output = x.new_empty(x.shape)
    block = 128
    _conv_step[(triton.cdiv(channels, block),)](
        x,
        history,
        taps,
        taps if dynamic is None else dynamic,
        output,
        channels,
        width,
        channels // heads if heads else 0,
        1 / scale,
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _conv_chunk(
    X,
    HISTORY,
    CHUNK,
    TAPS,
    DYNAMIC,
    OUT,
    C: tl.constexpr,
    K: tl.constexpr,
    CHUNK_STRIDE: tl.constexpr,
    CHANNELS_PER_HEAD: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(1)
    channel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = channel < C
    x = tl.load(X + token * C + channel, valid, 0).to(tl.float32)
    result = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(K):
        source = token + tap - (K - 1)
        left = tl.load(HISTORY + channel * (K - 1) + token + tap, valid & (source < 0), 0)
        right = tl.load(X + source * C + channel, valid & (source >= 0), 0).to(tl.float32)
        weight = tl.load(TAPS + channel * K + tap, valid, 0)
        if CHANNELS_PER_HEAD:
            head = channel // CHANNELS_PER_HEAD
            weight += tl.load(DYNAMIC + (token * (C // CHANNELS_PER_HEAD) + head) * K + K - 1 - tap, valid, 0)
            result += (left + right) * weight
        else:
            result = tl.fma(left + right, weight, result)
        if tap < K - 1:
            # Keep the previous history beside the candidate inputs so the
            # accepted prefix's new history is one contiguous slice per channel.
            tl.store(CHUNK + channel * CHUNK_STRIDE + tap, left, valid & (token == 0))
    tl.store(CHUNK + channel * CHUNK_STRIDE + K - 1 + token, x, valid)
    tl.store(OUT + token * C + channel, (result + x) * SCALE, valid)


def conv_chunk(x, history, chunk, taps, dynamic, heads, scale):
    channels, width = taps.shape
    output = x.new_empty(x.shape)
    block = 128
    _conv_chunk[(triton.cdiv(channels, block), x.size(1))](
        x.contiguous(),
        history,
        chunk,
        taps,
        taps if dynamic is None else dynamic,
        output,
        channels,
        width,
        chunk.size(-1),
        channels // heads if heads else 0,
        1 / scale,
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _normalize(
    X, WEIGHT, OUT, H: tl.constexpr, STRIDE: tl.constexpr, TOKENS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + i, i < H, 0).to(tl.float32)
    gain = tl.load(WEIGHT + row // TOKENS * STRIDE + i, i < H, 0)
    tl.store(OUT + row * H + i, _rms(x, H, EPS) * gain, i < H)


def normalize(x, weight, dtype, eps):
    width = x.size(-1)
    rows = x.numel() // width
    output = x.new_empty(x.shape, dtype=dtype)
    stride = weight.stride(0) if weight.ndim == 2 else 0
    tokens = rows // weight.size(0) if weight.ndim == 2 else 1
    _normalize[(rows,)](
        x.contiguous(), weight, output, width, stride, tokens, eps, triton.next_power_of_2(width), num_warps=4
    )
    return output


@triton.jit
def _clip_head(X, SCALE, OUT, H: tl.constexpr, LIMIT: tl.constexpr, LEARNED_HEADS: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(X + head * H + i, i < H, 0).to(tl.float32)
    if LEARNED_HEADS:
        x *= tl.load(SCALE + (head % LEARNED_HEADS) * H + i, i < H, 0)
    l2 = tl.sqrt(tl.sum(x * x, axis=0))
    factor = tl.minimum(1, LIMIT / (l2 + 1e-6))
    tl.store(OUT + head * H + i, x * factor, i < H)


def clip_heads(x, limit, weight):
    width = x.size(-1)
    output = x.new_empty(x.shape)
    _clip_head[(x.numel() // width,)](
        x.contiguous() if x.size(1) > 1 else x,
        x if weight is None else weight,
        output,
        width,
        limit,
        0 if weight is None else weight.size(0),
        triton.next_power_of_2(width),
        num_warps=1,
    )
    return output


@triton.jit
def _mix_states(
    STATES,
    QUERY,
    OUT,
    H: tl.constexpr,
    COUNT: tl.constexpr,
    QUERIES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    query = tl.program_id(0)
    token = tl.program_id(1)
    i = tl.arange(0, BLOCK_H)
    j = tl.arange(0, BLOCK_S)
    offsets = (token * COUNT + j[:, None]) * H + i[None, :]
    x = tl.load(STATES + offsets, (j[:, None] < COUNT) & (i[None, :] < H), 0).to(tl.float32)
    x *= tl.rsqrt(tl.sum(x * x, axis=1) / H + 1e-6)[:, None]
    q = tl.load(QUERY + query * H + i, i < H, 0)
    dot = tl.sum(x * q[None, :], axis=1) * (H**-0.5)
    logits = tl.where(j < COUNT, dot, -float("inf"))
    p = tl.exp(logits - tl.max(logits, axis=0))
    p /= tl.sum(p, axis=0)
    tl.store(OUT + (token * QUERIES + query) * H + i, tl.sum(x * p[:, None], axis=0), i < H)


def mix_states(states, queries):
    width = states.size(-1)
    count = states.size(-2)
    output = queries.new_empty((*states.shape[:-2], *queries.shape))
    _mix_states[(queries.size(0), states.numel() // (width * count))](
        states,
        queries,
        output,
        width,
        count,
        queries.size(0),
        triton.next_power_of_2(width),
        triton.next_power_of_2(count),
        num_warps=8,
        enable_fp_fusion=False,
    )
    return output


@triton.jit(do_not_specialize=["C"])
def _attention_probabilities(
    DOTS,
    PROFILES,
    SINK,
    POSITION,
    OUT,
    C,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    SINK_STRIDE: tl.constexpr,
    EXTENT: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    head = tl.program_id(0)
    token = tl.program_id(1)
    slot = tl.arange(0, BLOCK)
    position = tl.load(POSITION + token)
    if WINDOW:
        dist = position % C - slot
        dist = tl.where(dist < 0, dist + C, dist)
        valid = (dist <= position) & (dist < WINDOW)
    else:
        dist = position - slot
        valid = dist >= 0
    bias_offset = (token * HEADS + head) * EXTENT
    dot_offset = (head * TOKENS + token) * C + slot
    bias = tl.load(PROFILES + bias_offset + dist, (dist >= 0) & (dist < EXTENT) & (slot < C), 0)
    dot = tl.load(DOTS + dot_offset, slot < C, 0)
    scaled = (dot.to(tl.float32) * SCALE).to(tl.bfloat16)
    score = (scaled.to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    score = tl.where(valid & (slot < C), score, -float("inf"))
    sink = tl.load(SINK + token * SINK_STRIDE + head).to(tl.float32)
    maximum = tl.maximum(tl.max(score, axis=0), sink)
    mass = tl.exp(score - maximum)
    denominator = tl.sum(mass, axis=0) + tl.exp(sink - maximum)
    tl.store(OUT + dot_offset, mass / denominator, slot < C)


def attention_probabilities(dots, profiles, sink, position, window, scale):
    _, heads, tokens, capacity = dots.shape
    if tokens > 1:
        profiles = profiles.transpose(1, 2).contiguous().transpose(1, 2)
    output = dots.new_empty(dots.shape)
    _attention_probabilities[(heads, tokens)](
        dots,
        profiles,
        sink,
        position,
        output,
        capacity,
        tokens,
        heads,
        sink.stride(1),
        profiles.size(-1),
        window or 0,
        scale,
        triton.next_power_of_2(capacity),
        num_warps=4 if capacity <= 2048 else 8,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _situ(X, OUT, M: tl.constexpr, GATE: tl.constexpr, UP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(X + 2 * i, i < M, 0).to(tl.float32)
    up = tl.load(X + 2 * i + 1, i < M, 0).to(tl.float32)
    gate_factor = GATE * tl.extra.cuda.libdevice.tanh(gate / GATE)
    up_factor = UP * tl.extra.cuda.libdevice.tanh(up / UP)
    tl.store(OUT + i, gate_factor * tl.sigmoid(gate) * up_factor, i < M)


def situ(x, gate_limit, up_limit):
    output = x.new_empty((x.size(0), x.size(1) // 2))
    _situ[(triton.cdiv(output.numel(), 128),)](x, output, output.numel(), gate_limit, up_limit, 128, num_warps=4)
    return output


@triton.jit
def _sum_routes(X, WEIGHT, OUT, H: tl.constexpr, TOP: tl.constexpr, BLOCK: tl.constexpr, ROUTES: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    route = tl.arange(0, ROUTES)
    x = tl.load(X + route[:, None] * H + i[None, :], (route[:, None] < TOP) & (i[None, :] < H), 0).to(tl.float32)
    weight = tl.load(WEIGHT + route, route < TOP, 0)
    tl.store(OUT + i, tl.sum(x * weight[:, None], axis=0), i < H)


def sum_routes(values, weights, shape):
    top, width = values.shape
    output = values.new_empty(shape)
    _sum_routes[(triton.cdiv(width, 128),)](
        values, weights, output, width, top, 128, triton.next_power_of_2(top), num_warps=4
    )
    return output
