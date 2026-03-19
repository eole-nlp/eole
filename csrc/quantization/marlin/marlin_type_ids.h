#pragma once

#include "eole_scalar_type.hpp"

namespace marlin {

// Single import point for scalar type ids used by Marlin dispatch code.
// Keep these aliases centralized to avoid drift between marlin_dense.cu and
// marlin_moe_wna16.cu.
using vllm::FP16_ID;
using vllm::BF16_ID;
using vllm::U4B8_ID;
using vllm::U8B128_ID;
using vllm::U4_ID;
using vllm::U8_ID;

}  // namespace marlin
