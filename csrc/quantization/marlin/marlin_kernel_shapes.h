#pragma once

// Canonical Marlin thread/block shapes used by both dense and MoE dispatch.
// Keep this list single-sourced so the two dispatchers cannot drift.
#define MARLIN_FOR_EACH_SHAPE(OP)            \
  OP(256, 1,  8, 8, true)                   \
  OP(128, 1,  8, 4, true)                   \
  OP(128, 1,  4, 8, true)                   \
  OP(256, 1,  8, 8, false)                  \
  OP(128, 1,  8, 4, false)                  \
  OP(128, 1,  4, 8, false)                  \
  OP(256, 2, 16, 4, false)                  \
  OP(128, 2,  8, 4, false)                  \
  OP(128, 2,  4, 8, false)                  \
  OP(256, 3, 16, 4, false)                  \
  OP(128, 3,  8, 4, false)                  \
  OP(128, 3,  4, 8, false)                  \
  OP(256, 4, 16, 4, false)                  \
  OP(128, 4,  8, 4, false)                  \
  OP(128, 4,  4, 8, false)

#define MARLIN_FOR_EACH_SHAPE_WITH_GB(OP, GB) \
  OP(256, 1,  8, 8, true, GB)                 \
  OP(128, 1,  8, 4, true, GB)                 \
  OP(128, 1,  4, 8, true, GB)                 \
  OP(256, 1,  8, 8, false, GB)                \
  OP(128, 1,  8, 4, false, GB)                \
  OP(128, 1,  4, 8, false, GB)                \
  OP(256, 2, 16, 4, false, GB)                \
  OP(128, 2,  8, 4, false, GB)                \
  OP(128, 2,  4, 8, false, GB)                \
  OP(256, 3, 16, 4, false, GB)                \
  OP(128, 3,  8, 4, false, GB)                \
  OP(128, 3,  4, 8, false, GB)                \
  OP(256, 4, 16, 4, false, GB)                \
  OP(128, 4,  8, 4, false, GB)                \
  OP(128, 4,  4, 8, false, GB)
