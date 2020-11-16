// Copyright (c) 2020, SenseTime.

#ifndef PYTHON_COMPUTE_EXT_PAT_EXT_GELU_PYTORCH_HPP_  // NOLINT
#define PYTHON_COMPUTE_EXT_PAT_EXT_GELU_PYTORCH_HPP_  // NOLINT

#include <torch/extension.h>

void gelu_fwd_cpu(at::Tensor input, at::Tensor output);
void gelu_fwd_gpu(at::Tensor input, at::Tensor output);
void gelu_bwd_cpu(at::Tensor grad_out, at::Tensor input, at::Tensor output);
void gelu_bwd_gpu(at::Tensor grad_out, at::Tensor input, at::Tensor output);

#endif  // PYTHON_COMPUTE_EXT_PAT_EXT_GELU_PYTORCH_HPP_  // NOLINT
