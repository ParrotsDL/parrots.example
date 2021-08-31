// Copyright (c) 2019, SenseTime.
#include <parrots/foundation/darraylite.hpp>

#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <parrots/darray/darraymath.hpp>

#include "parrots_cpp_helper.hpp"
using namespace parrots;
using namespace pape;

void modulated_deformable_im2col_cuda(ContextBase& ctx,
        const DArrayLite data_im, const DArrayLite data_offset,
        const DArrayLite data_mask, const int batch_size, const int channels,
        const int height_im, const int width_im, const int height_col,
        const int width_col, const int kernel_h, const int kenerl_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int deformable_group, DArrayLite data_col);

void modulated_deformable_col2im_cuda(ContextBase& ctx,
        const DArrayLite data_col, const DArrayLite data_offset,
        const DArrayLite data_mask, const int batch_size, const int channels,
        const int height_im, const int width_im, const int height_col,
        const int width_col, const int kernel_h, const int kenerl_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int deformable_group, DArrayLite grad_im);

void modulated_deformable_col2im_coord_cuda(ContextBase& ctx,
        const DArrayLite data_col, const DArrayLite data_im,
        const DArrayLite data_offset, const DArrayLite data_mask,
        const int batch_size, const int channels, const int height_im,
        const int width_im, const int height_col, const int width_col,
        const int kernel_h, const int kenerl_w, const int pad_h,
        const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int deformable_group,DArrayLite grad_offset,
        DArrayLite grad_mask);

void modulated_deform_conv_cuda_forward(
    CudaContext& ctx,
    const SSElement& attr,
    const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
    int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, deformable_group, with_bias;
    SSAttrs(attr)
        .get<int>("kernel_h", kernel_h)
        .get<int>("kernel_w", kernel_w)
        .get<int>("stride_h", stride_h)
        .get<int>("stride_w", stride_w)
        .get<int>("pad_h", pad_h)
        .get<int>("pad_w", pad_w)
        .get<int>("dilation_h", dilation_h)
        .get<int>("dilation_w", dilation_w)
        .get<int>("group", group)
        .get<int>("deformable_group", deformable_group)
        .get<int>("with_bias", with_bias)
        .done();

    auto input = ins[0];
    auto weight = ins[1];
    auto bias = ins[2];
    auto ones = ins[3];
    auto offset = ins[4];
    auto mask = ins[5];

    auto output = outs[0];
    auto columns = outs[1];

    const int batch = input.dim(0);
    const int channels = input.dim(1);
    const int height = input.dim(2);
    const int width = input.dim(3);

    const int channels_out = weight.dim(0);
    const int channels_kernel = weight.dim(1);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (ones.ndims() != 2 ||
        ones.dim(0) * ones.dim(1) < height_out * width_out)
    {
        ones = ctx.createDArrayLite(input.elemType(), DArrayShape(height_out, width_out));
        fill(ctx, ones, 1);
    }

    output = output.view({batch, channels_out, height_out, width_out});
    fill(ctx, output, 0.);
    columns = ctx.createDArrayLite(input.elemType(), DArrayShape(channels * kernel_h * kernel_w, height_out * width_out));
    fill(ctx, columns, 0.);

    output = output.view({output.dim(0), group, output.dim(1) / group, output.dim(2), output.dim(3)});

    for (int b = 0; b < batch; b++) {
        modulated_deformable_im2col_cuda(ctx, input[b], offset[b], mask[b],
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                         deformable_group, columns);

        // divide into group
        weight = weight.view({group, weight.dim(0) / group, weight.dim(1), weight.dim(2), weight.dim(3)});
        columns = columns.view({group, columns.dim(0) / group, columns.dim(1)});

        for (int g = 0; g < group; g++) {
            pape::gemm(ctx, 1, false, weight[g].view({weight.dim(1), weight.dim(2) * weight.dim(3) * weight.dim(4)}),
                         false, columns[g], 1, output[b][g]);
        }


        weight = weight.view({weight.dim(0) * weight.dim(1), weight.dim(2), weight.dim(3), weight.dim(4)});
        columns = columns.view({columns.dim(0) * columns.dim(1), columns.dim(2)});
    }
    output = output.view({output.dim(0), output.dim(1) * output.dim(2), output.dim(3), output.dim(4)});
    if (with_bias)
    {
        bias = bias.view({1, bias.dim(0), 1, 1});
        add(ctx, output, bias, output);
    }
}

void modulated_deform_conv_cuda_backward(
    CudaContext &ctx,
    const SSElement &attr,
    const OperatorBase::in_list_t &ins,
    OperatorBase::out_list_t &outs)
{
    int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, deformable_group, with_bias;
    SSAttrs(attr)
        .get<int>("kernel_h", kernel_h)
        .get<int>("kernel_w", kernel_w)
        .get<int>("stride_h", stride_h)
        .get<int>("stride_w", stride_w)
        .get<int>("pad_h", pad_h)
        .get<int>("pad_w", pad_w)
        .get<int>("dilation_h", dilation_h)
        .get<int>("dilation_w", dilation_w)
        .get<int>("group", group)
        .get<int>("deformable_group", deformable_group)
        .get<int>("with_bias", with_bias)
        .done();

    auto input = ins[0];
    auto weight = ins[1];
    auto bias = ins[2];
    auto ones = ins[3];
    auto offset = ins[4];
    auto mask = ins[5];

    auto columns = outs[0];
    auto grad_input = outs[1];
    auto grad_weight = outs[2];
    auto grad_bias = outs[3];
    auto grad_offset = outs[4];
    auto grad_mask = outs[5];
    auto grad_output = outs[6];

    const int batch = input.dim(0);
    const int channels = input.dim(1);
    const int height = input.dim(2);
    const int width = input.dim(3);

    const int channels_kernel = weight.dim(1);


    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (ones.ndims() != 2 ||
        ones.dim(0) * ones.dim(1) < height_out * width_out)
    {
        // Resize plane and fill with ones...
        ones = ctx.createDArrayLite(input.elemType(), DArrayShape(height_out, width_out));
        fill(ctx, ones, 1.);
    }

    grad_input = grad_input.view({batch, channels, height, width});
    columns = ctx.createDArrayLite(input.elemType(), DArrayShape(channels * kernel_h * kernel_w, height_out * width_out));
    fill(ctx, columns, 0.);

    grad_output = grad_output.view({grad_output.dim(0), group, grad_output.dim(1) / group, grad_output.dim(2), grad_output.dim(3)});

    for (int b = 0; b < batch; b++)
    {
        // divide int group
        columns = columns.view({group, columns.dim(0) / group, columns.dim(1)});
        weight = weight.view({group, weight.dim(0) / group, weight.dim(1), weight.dim(2), weight.dim(3)});

        for (int g = 0; g < group; g++)
        {
            pape::gemm(ctx, 1, true, weight[g].view({weight.dim(1), weight.dim(2) * weight.dim(3) * weight.dim(4)}),
                         false, grad_output[b][g].view({1, grad_output.dim(2) * grad_output.dim(3) * grad_output.dim(4)}),
                      0, columns[g]);
        }

        columns = columns.view({columns.dim(0) * columns.dim(1), columns.dim (2)});
        weight = weight.view({weight.dim(0) * weight.dim(1), weight.dim(2), weight.dim(3), weight.dim(4)});

        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cuda(ctx, columns, input[b], offset[b], mask[b],
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset[b], grad_mask[b]);
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(ctx, columns, offset[b], mask[b],
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input[b]);

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        modulated_deformable_im2col_cuda(ctx, input[b], offset[b], mask[b],
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         columns);

        columns = columns.view({group, columns.dim(0) / group, columns.dim(1)});
        grad_weight = grad_weight.view({group, grad_weight.dim(0) / group, grad_weight.dim(1), grad_weight.dim(2), grad_weight.dim(3)});
        if (with_bias)
            grad_bias = grad_bias.view({group, grad_bias.dim(0) / group});

        for (int g = 0; g < group; g++)
        {
            grad_weight[g] = grad_weight[g].view({1, grad_weight.dim(1) * grad_weight.dim(2) * grad_weight.dim(3) * grad_weight.dim(3)});
            pape::gemm(ctx, 1, false, grad_output[b][g].view({grad_output.dim(2), grad_output.dim(3) * grad_output.dim(4)}),
                         true, columns[g],
                      1, grad_weight[g]);

            if (with_bias)
            {
                grad_bias[g] = grad_bias[g].view({grad_bias.dim(1), 1});
                pape::gemm(ctx, 1, false, grad_bias[b][g].view({grad_bias.dim(2), grad_bias.dim(3) * grad_bias.dim(4)}),
                             false, ones.view({ones.dim(0) * ones.dim(1), 1}),
                          1, grad_bias[g]);
            }
        }

        columns = columns.view({columns.dim(0) * columns.dim(1), columns.dim(2)});
        grad_weight = grad_weight.view({grad_weight.dim(0) * grad_weight.dim(1), grad_weight.dim(2), grad_weight.dim(3), grad_weight.dim(4)});
        if (with_bias)
            grad_bias = grad_bias.view(DArrayShape{grad_bias.dim(0) * grad_bias.dim(1)});
    }
    grad_output = grad_output.view({grad_output.dim(0) * grad_output.dim(1), grad_output.dim(2), grad_output.dim(3), grad_output.dim(4)});
}

PARROTS_EXTENSION_REGISTER(modulated_deform_conv_cuda_forward)
    .attr("kernel_h")
    .attr("kernel_w")
    .attr("stride_h")
    .attr("stride_w")
    .attr("pad_h")
    .attr("pad_w")
    .attr("dilation_h")
    .attr("dilation_w")
    .attr("group")
    .attr("deformable_group")
    .attr("with_bias")
    .input(6)
    .output(2)
    .apply(modulated_deform_conv_cuda_forward)
    .done();

PARROTS_EXTENSION_REGISTER(modulated_deform_conv_cuda_backward)
    .attr("kernel_h")
    .attr("kernel_w")
    .attr("stride_h")
    .attr("stride_w")
    .attr("pad_h")
    .attr("pad_w")
    .attr("dilation_h")
    .attr("dilation_w")
    .attr("group")
    .attr("deformable_group")
    .attr("with_bias")
    .input(6)
    .output(7)
    .apply(modulated_deform_conv_cuda_backward)
    .done();
