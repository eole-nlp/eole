/**
 * eole_scalar_type.hpp
 *
 * Minimal scalar-type definitions required by marlin_template.h.
 * This is Eole's own file; it has zero dependency on the vLLM Python package
 * or shared library.  It lives in the vllm namespace solely so that
 * marlin_template.h (which hardcodes vllm::kFloat16 etc.) compiles unchanged.
 *
 * Weight-type ids MUST match vLLM checkpoint values:
 *   kU4B8   id=4   symmetric int4  (gptqmodel Marlin default)
 *   kU8B128 id=5   symmetric int8
 *   kU4     id=8   asymmetric int4 (AWQ)
 *   kU8     id=9   asymmetric int8
 *   kS4     id=10  signed int4
 *
 * Activation-type ids (1000+) are internal and never cross the Python
 * boundary; they just need to be unique.
 */

#pragma once
#include <cstdint>
#include <torch/extension.h>   // TORCH_CHECK

namespace vllm {

using ScalarTypeId = int64_t;

struct ScalarType {
  ScalarTypeId _id;
  int          _bits;
  bool         _is_float;
  const char*  _name;

  constexpr ScalarTypeId id()        const { return _id;   }
  constexpr int          size_bits() const { return _bits;  }
  constexpr bool operator==(const ScalarType& o) const { return _id == o._id; }
  constexpr bool operator!=(const ScalarType& o) const { return _id != o._id; }

  // Called with compile-time constants (template params), so constexpr is fine.
  static constexpr ScalarType from_id(ScalarTypeId id) {
    switch (id) {
      // Weight quant types – ids must match checkpoint values
      case  4: return kU4B8;
      case  5: return kU8B128;
      case  8: return kU4;
      case  9: return kU8;
      case 10: return kS4;
      // Activation / output types – internal ids
      case 1000: return kFloat16;
      case 1001: return kBFloat16;
      case 1002: return kS8;
      case 1003: return kFE4M3fn;
      case 1004: return kFE8M0fnu;
      case 1005: return kFE2M1f;
      default:
        // In device code constexpr path this is unreachable for valid inputs.
        return {-1, 0, false, "unknown"};
    }
  }

  // Static constant declarations (defined below)
  static const ScalarType kFloat16;
  static const ScalarType kBFloat16;
  static const ScalarType kS8;
  static const ScalarType kFE4M3fn;
  static const ScalarType kFE8M0fnu;
  static const ScalarType kFE2M1f;
  static const ScalarType kU4B8;
  static const ScalarType kU8B128;
  static const ScalarType kU4;
  static const ScalarType kU8;
  static const ScalarType kS4;
};

// Activation / output types (internal ids)
inline constexpr ScalarType ScalarType::kFloat16  = {1000, 16, true,  "float16"  };
inline constexpr ScalarType ScalarType::kBFloat16 = {1001, 16, true,  "bfloat16" };
inline constexpr ScalarType ScalarType::kS8       = {1002,  8, false, "int8"     };
inline constexpr ScalarType ScalarType::kFE4M3fn  = {1003,  8, true,  "fp8_e4m3" };
inline constexpr ScalarType ScalarType::kFE8M0fnu = {1004,  8, true,  "fp8_e8m0" };
inline constexpr ScalarType ScalarType::kFE2M1f   = {1005,  4, true,  "fp4_e2m1" };

// Weight quantisation types (ids must match vLLM checkpoints)
inline constexpr ScalarType ScalarType::kU4B8   = {  4,  4, false, "uint4b8"   };
inline constexpr ScalarType ScalarType::kU8B128 = {  5,  8, false, "uint8b128" };
inline constexpr ScalarType ScalarType::kU4     = {  8,  4, false, "uint4"     };
inline constexpr ScalarType ScalarType::kU8     = {  9,  8, false, "uint8"     };
inline constexpr ScalarType ScalarType::kS4     = { 10,  4, false, "int4"      };

// Module-level aliases so marlin_template.h can write e.g. vllm::kFloat16
static constexpr ScalarType kFloat16  = ScalarType::kFloat16;
static constexpr ScalarType kBFloat16 = ScalarType::kBFloat16;
static constexpr ScalarType kS8       = ScalarType::kS8;
static constexpr ScalarType kFE4M3fn  = ScalarType::kFE4M3fn;
static constexpr ScalarType kFE8M0fnu = ScalarType::kFE8M0fnu;
static constexpr ScalarType kFE2M1f   = ScalarType::kFE2M1f;
static constexpr ScalarType kU4B8     = ScalarType::kU4B8;
static constexpr ScalarType kU8B128   = ScalarType::kU8B128;
static constexpr ScalarType kU4       = ScalarType::kU4;
static constexpr ScalarType kU8       = ScalarType::kU8;
static constexpr ScalarType kS4       = ScalarType::kS4;

// ScalarTypeId convenience aliases used for kernel dispatch in marlin_dense.cu
// and marlin_moe_wna16.cu.  Defined here once to avoid duplication.
inline constexpr ScalarTypeId FP16_ID   = kFloat16.id();
inline constexpr ScalarTypeId BF16_ID   = kBFloat16.id();
inline constexpr ScalarTypeId U4B8_ID   = kU4B8.id();
inline constexpr ScalarTypeId U8B128_ID = kU8B128.id();
inline constexpr ScalarTypeId U4_ID     = kU4.id();
inline constexpr ScalarTypeId U8_ID     = kU8.id();

}  // namespace vllm
