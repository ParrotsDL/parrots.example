// Copyright (c) 2020, SenseTime.

#include "gelu_pytorch.hpp"
#include <iostream>
void gelu_fwd_interface(float*, float*, int64_t);
void gelu_bwd_interface(float*, float*, float*, int64_t);

void gelu_fwd_cpu(at::Tensor input, at::Tensor output) {
    output = input*torch::sigmoid(1.702*input);
}

void gelu_fwd_gpu(at::Tensor input, at::Tensor output) {
    TORCH_CHECK(input.dtype() == torch::kFloat32,
        "Datatype not implemented");
    auto ret = torch::zeros_like(input);
    int64_t size = ret.numel();
    gelu_fwd_interface(input.data_ptr<float>(),
            ret.data_ptr<float>(), size);
    output = ret;
}

void gelu_bwd_cpu(at::Tensor grad_out, at::Tensor input, at::Tensor output) {
        auto tmp = torch::sigmoid(1.702*input);
        output = grad_out*(tmp+1.702*input*tmp*(1.0-tmp));
}

void gelu_bwd_gpu(at::Tensor grad_out, at::Tensor input, at::Tensor output) {
    TORCH_CHECK(input.dtype() == torch::kFloat32,
        "Datatype not implemented");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32,
        "Datatype not implemented");
    auto ret = torch::zeros_like(input);
    int64_t size = ret.numel();
    gelu_bwd_interface(grad_out.data_ptr<float>(),
                       input.data_ptr<float>(),
                       ret.data_ptr<float>(), size);
    output = ret;
}
