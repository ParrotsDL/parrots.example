// Copyright (c) 2018, SenseTime.
#include <parrots/foundation/darraylite.hpp>

#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <parrots/darray/darraymath.hpp>

#include "parrots_cpp_helper.hpp"

using namespace parrots;  // NOLINT
using namespace pape;


void deform_conv_forward_cpu(
        CudaContext& ctx,
        const SSElement& attr,
        const OperatorBase::in_list_t& ins,
        OperatorBase::out_list_t& outs) {
    int kW, kH, dW, dH, padW, padH, dilationW, dilationH, groups, deformable_group;
    SSAttrs(attr)
        .get<int>("kW", kW)
        .get<int>("kH", kH)
        .get<int>("dW", dW)
        .get<int>("dH", dH)
        .get<int>("padW", padW)
        .get<int>("padH", padH)
        .get<int>("dilationW", dilationW)
        .get<int>("dilationH", dilationH)
        .get<int>("groups", groups)
        .get<int>("deformable_group", deformable_group)
        .done();
    throw "cpu version of deform_conv is not implemented!";
}

void deform_conv_backward_input_cpu(
        CudaContext& ctx,
        const SSElement& attr,
        const OperatorBase::in_list_t& ins,
        OperatorBase::out_list_t& outs) {
    int kW, kH, dW, dH, padW, padH, dilationW, dilationH, groups, deformable_group;
    SSAttrs(attr)
        .get<int>("kW", kW)
        .get<int>("kH", kH)
        .get<int>("dW", dW)
        .get<int>("dH", dH)
        .get<int>("padW", padW)
        .get<int>("padH", padH)
        .get<int>("dilationW", dilationW)
        .get<int>("dilationH", dilationH)
        .get<int>("groups", groups)
        .get<int>("deformable_group", deformable_group)
        .done();
    throw "cpu version of deform_conv is not implemented!";
}

void deform_conv_backward_parameters_cpu(
        CudaContext& ctx,
        const SSElement& attr,
        const OperatorBase::in_list_t& ins,
        OperatorBase::out_list_t& outs) {
    int kW, kH, dW, dH, padW, padH, dilationW, dilationH, groups, deformable_group;
    float scale;
    SSAttrs(attr)
        .get<int>("kW", kW)
        .get<int>("kH", kH)
        .get<int>("dW", dW)
        .get<int>("dH", dH)
        .get<int>("padW", padW)
        .get<int>("padH", padH)
        .get<int>("dilationW", dilationW)
        .get<int>("dilationH", dilationH)
        .get<int>("groups", groups)
        .get<int>("deformable_group", deformable_group)
        .get<float>("scale", scale)
        .done();
    throw "cpu version of deform_conv is not implemented!";
}

#ifdef PARROTS_USE_CUDA
void deformable_im2col(cudaStream_t stream, DArrayLite data_im,
                       DArrayLite data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int deformable_group, DArrayLite data_col);

void deformable_col2im(cudaStream_t stream, DArrayLite data_col,
                       DArrayLite data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int deformable_group, DArrayLite grad_im);

void deformable_col2im_coord(cudaStream_t stream, DArrayLite data_col,
                             DArrayLite data_im, DArrayLite data_offset,
                             const int channels, const int height,
                             const int width, const int ksize_h,
                             const int ksize_w, const int pad_h,
                             const int pad_w, const int stride_h,
                             const int stride_w, const int dilation_h,
                             const int dilation_w, const int deformable_group,
                             DArrayLite grad_offset);

void deform_conv_forward_cuda(
        CudaContext& ctx,
        const SSElement& attr,
        const OperatorBase::in_list_t& ins,
        OperatorBase::out_list_t& outs) {
    int kW, kH, dW, dH, padW, padH, dilationW, dilationH, groups, deformable_group;
    SSAttrs(attr)
        .get<int>("kW", kW)
        .get<int>("kH", kH)
        .get<int>("dW", dW)
        .get<int>("dH", dH)
        .get<int>("padW", padW)
        .get<int>("padH", padH)
        .get<int>("dilationW", dilationW)
        .get<int>("dilationH", dilationH)
        .get<int>("groups", groups)
        .get<int>("deformable_group", deformable_group)
        .done();

    auto input = ins[0];
    auto weight = ins[1];
    auto offset = ins[2];

    auto output = outs[0];
    auto columns = outs[1];

    bool batch = (input.ndims() == 4);
    if (!batch) {
        batch = false;
        input = input.view({1, input.dim(0), input.dim(1), input.dim(2)});
        offset = offset.view({1, offset.dim(0), offset.dim(1), offset.dim(2)});
    }

    size_t batchSize = input.dim(0);
    size_t nInputPlane = input.dim(1);
    size_t inputHeight = input.dim(2);
    size_t inputWidth = input.dim(3);

    size_t nOutputPlane = weight.dim(0);

    size_t outputWidth =
            (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    size_t outputHeight =
            (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    output = output.view({batchSize, groups, nOutputPlane / groups, outputHeight * outputWidth});
    output.setZeros(ctx.getStream());

    columns = columns.view({groups, nInputPlane / groups * kW * kH, outputHeight * outputWidth});
    columns.setZeros(ctx.getStream());

    weight = weight.view({groups, nOutputPlane / groups, nInputPlane / groups * kH * kW});

    for (size_t elt = 0; elt < batchSize; ++elt) {
        auto input_n = input[elt];
        auto offset_n = offset[elt];
        auto output_n = output[elt];

        deformable_im2col(getStreamNative<CudaDevice>(ctx.getStream()),
                          input_n, offset_n, nInputPlane,
                          inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
                          dilationH, dilationW, deformable_group, columns);

        for (int g = 0; g < groups; ++g) {
            auto columns_g = columns[g];
            auto weight_g = weight[g];
            auto output_g = output_n[g];

            pape::gemm(ctx, 1, false, weight_g, false, columns_g, 1, output_g);
        }
    }

    output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

    if (!batch) {
        output = output.view({nOutputPlane, outputHeight, outputWidth});
    }

    outs[0] = output;
}

void deform_conv_backward_input_cuda(
        CudaContext& ctx,
        const SSElement& attr,
        const OperatorBase::in_list_t& ins,
        OperatorBase::out_list_t& outs) {
    int kW, kH, dW, dH, padW, padH, dilationW, dilationH, groups, deformable_group;
    SSAttrs(attr)
        .get<int>("kW", kW)
        .get<int>("kH", kH)
        .get<int>("dW", dW)
        .get<int>("dH", dH)
        .get<int>("padW", padW)
        .get<int>("padH", padH)
        .get<int>("dilationW", dilationW)
        .get<int>("dilationH", dilationH)
        .get<int>("groups", groups)
        .get<int>("deformable_group", deformable_group)
        .done();

    auto input = ins[0];
    auto weight = ins[1];
    auto offset = ins[2];
    auto gradOutput = ins[3];

    auto gradInput = outs[0];
    auto gradOffset = outs[1];
    auto columns = outs[2];

    bool batch = (input.ndims() == 4);
    if (!batch) {
        input = input.view({1, input.dim(0), input.dim(1), input.dim(2)});
        offset = offset.view({1, offset.dim(0), offset.dim(1), offset.dim(2)});
        gradOutput = gradOutput.view({1, gradOutput.dim(0), gradOutput.dim(1), gradOutput.dim(2)});
        gradInput = gradOutput.view({1, gradInput.dim(0), gradInput.dim(1), gradInput.dim(2)});
        gradOffset = gradOffset.view({1, gradOffset.dim(0), gradOffset.dim(1), gradOffset.dim(2)});
    }

    size_t batchSize = input.dim(0);
    size_t nInputPlane = input.dim(1);
    size_t inputHeight = input.dim(2);
    size_t inputWidth = input.dim(3);

    size_t nOutputPlane = weight.dim(0);

    size_t outputWidth =
            (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    size_t outputHeight =
            (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    size_t m = nInputPlane / groups * kW * kH;
    size_t n = outputHeight * outputWidth;
    size_t k = nOutputPlane / groups;

    gradOutput = gradOutput.view({batchSize, groups, k, n});
    columns = columns.view({groups, m, n});
    weight = weight.view({groups, k, m});

    for (size_t elt = 0; elt < batchSize; ++elt) {
        auto gradInput_n = gradInput[elt];
        auto gradOffset_n = gradOffset[elt];
        auto input_n = input[elt];
        auto offset_n = offset[elt];
        auto gradOutput_n = gradOutput[elt];

        for (int g = 0; g < groups; ++g) {
            auto gradOutput_g = gradOutput_n[g];
            auto weight_g = weight[g];
            auto columns_g = columns[g];

            // columns_g = weight_g.t() * gradOutput_g
            pape::gemm(ctx, 1, true, weight_g, false, gradOutput_g, 0, columns_g);
        }

        deformable_col2im_coord(getStreamNative<CudaDevice>(ctx.getStream()),
                                columns, input_n, offset_n,
                                nInputPlane, inputHeight, inputWidth, kH, kW,
                                padH, padW, dH, dW, dilationH, dilationW,
                                deformable_group, gradOffset_n);

        deformable_col2im(getStreamNative<CudaDevice>(ctx.getStream()),
                          columns, offset_n, nInputPlane,
                          inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
                          dilationH, dilationW, deformable_group, gradInput_n);
    }

    gradOutput = gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

    if (!batch) {
        gradInput = gradOutput.view({nInputPlane, inputHeight, inputWidth});
        gradOffset = gradOffset.view({gradOffset.dim(1), gradOffset.dim(2), gradOffset.dim(3)});
    }

    outs[0] = gradInput;
    outs[1] = gradOffset;
}

void deform_conv_backward_parameters_cuda(
        CudaContext& ctx,
        const SSElement& attr,
        const OperatorBase::in_list_t& ins,
        OperatorBase::out_list_t& outs) {
    int kW, kH, dW, dH, padW, padH, dilationW, dilationH, groups, deformable_group;
    float scale;
    SSAttrs(attr)
        .get<int>("kW", kW)
        .get<int>("kH", kH)
        .get<int>("dW", dW)
        .get<int>("dH", dH)
        .get<int>("padW", padW)
        .get<int>("padH", padH)
        .get<int>("dilationW", dilationW)
        .get<int>("dilationH", dilationH)
        .get<int>("groups", groups)
        .get<int>("deformable_group", deformable_group)
        .get<float>("scale", scale)
        .done();

    auto input = ins[0];
    auto offset = ins[1];
    auto gradOutput = ins[2];

    auto gradWeight = outs[0];
    auto columns = outs[1];

    bool batch = (input.ndims() == 4);
    if (!batch) {
        input = input.view({1, input.dim(0), input.dim(1), input.dim(2)});
        gradOutput = gradOutput.view({1, gradOutput.dim(0), gradOutput.dim(1), gradOutput.dim(2)});
    }

    size_t batchSize = input.dim(0);
    size_t nInputPlane = input.dim(1);
    size_t inputHeight = input.dim(2);
    size_t inputWidth = input.dim(3);

    size_t nOutputPlane = gradWeight.dim(0);

    size_t outputWidth =
            (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    size_t outputHeight =
            (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    size_t m = nInputPlane / groups * kW * kH;
    size_t n = outputHeight * outputWidth;
    size_t k = nOutputPlane / groups;

    columns = columns.view({groups, m, n});
    gradOutput = gradOutput.view({batchSize, groups, k, n});
    gradWeight = gradWeight.view({groups, k, m});
    gradWeight.setZeros(ctx.getStream());

    for (size_t elt = 0; elt < batchSize; ++elt) {
        auto input_n = input[elt];
        auto offset_n = offset[elt];
        auto gradOutput_n = gradOutput[elt];

        deformable_im2col(getStreamNative<CudaDevice>(ctx.getStream()),
                          input_n, offset_n, nInputPlane, inputHeight, inputWidth,
                          kH, kW, padH, padW, dH, dW, dilationH, dilationW,
                          deformable_group, columns);

        for (int g = 0; g < groups; ++g) {
            auto columns_g = columns[g];
            auto gradOutput_g = gradOutput_n[g];
            auto gradWeight_g = gradWeight[g];

            // gradWeight_g += scale * gradOutput_g * columns_g.t()
            pape::gemm(ctx, scale, false, gradOutput_g, true, columns_g, 1, gradWeight_g);
        }
    }

    gradWeight = gradWeight.view({nOutputPlane, nInputPlane / groups, kH, kW});

    if (!batch) {
   }
    outs[0] = gradWeight;
}
#endif  // PARROTS_USE_CUDA


PARROTS_EXTENSION_REGISTER(deform_conv_forward)
    .attr("kW").attr("kH")
    .attr("dW").attr("dH")
    .attr("padW").attr("padH")
    .attr("dilationW").attr("dilationH")
    .attr("groups").attr("deformable_group")
    .input(3)
    .output(3)
    .apply(deform_conv_forward_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(deform_conv_forward_cuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(deform_conv_backward_input)
    .attr("kW").attr("kH")
    .attr("dW").attr("dH")
    .attr("padW").attr("padH")
    .attr("dilationW").attr("dilationH")
    .attr("groups").attr("deformable_group")
    .input(4)
    .output(3)
    .apply(deform_conv_backward_input_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(deform_conv_backward_input_cuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(deform_conv_backward_parameters)
    .attr("kW").attr("kH")
    .attr("dW").attr("dH")
    .attr("padW").attr("padH")
    .attr("dilationW").attr("dilationH")
    .attr("groups").attr("deformable_group")
    .attr("scale")
    .input(3)
    .output(3)
    .apply(deform_conv_backward_parameters_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(deform_conv_backward_parameters_cuda)
#endif
    .done();
