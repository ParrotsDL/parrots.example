// Copyright (c) 2018, SenseTime.

#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>


using namespace parrots;  // NOLINT

float bilinear_interpolate_cpu(
        const float* bottom_data,
        const int height, const int width,
        float y, float x, const int index /* index for debug only*/) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        // empty
        return 0;
    }

    if (y <= 0) {
        y = 0;
    }
    if (x <= 0) {
        x = 0;
    }

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (float)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (float)x_low;
    } else {
        x_high = x_low + 1;
    }

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1. -ly, hx = 1. - lx;
    // do bilinear interpolation
    float v1 = bottom_data[y_low * width + x_low];
    float v2 = bottom_data[y_low * width + x_high];
    float v3 = bottom_data[y_high * width + x_low];
    float v4 = bottom_data[y_high * width + x_high];
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

void bilinear_interpolate_gradient_cpu(
        const int height, const int width, float y, float x,
        const int index, float diff_val, float *offset_bottom_diff) {
    float w1, w2,w3, w4;
    int x_low, x_high, y_low, y_high;
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        // empty
        w1 = w2 = w3 = w4 = 0.;
        x_low = x_high = y_low = y_high = -1;
        return;
    }

    if (y <= 0) {
        y = 0;
    }
    if (x <= 0) {
        x = 0;
    }

    y_low = (int)y;
    x_low = (int)x;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (float)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (float)x_low;
    } else {
        x_high = x_low + 1;
    }

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1. - ly, hx = 1. - lx;

    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
        offset_bottom_diff[y_low * width + x_low] = diff_val * w1;
        offset_bottom_diff[y_low * width + x_high] = diff_val * w2;
        offset_bottom_diff[y_high * width + x_low] = diff_val * w3;
        offset_bottom_diff[y_high * width + x_high] = diff_val * w4;
    }
    return;
}

void forward_cpu_kernel(const int index, const float* bottom_data, 
        const float spatial_scale, const int height, const int width,
        const int channels, const int aligned_height, const int aligned_width,
        const int sampling_ratio, const float* bottom_rois, float* top_data) {
    // (n, c, ph, pw) is an element in the aligned output
    int pw = index % aligned_width;
    int ph = (index / aligned_width) % aligned_height;
    int c  = (index / aligned_width / aligned_height) % channels;
    int n  = index / aligned_width / aligned_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    float roi_start_w = offset_bottom_rois[1] * spatial_scale;
    float roi_start_h = offset_bottom_rois[2] * spatial_scale;
    float roi_end_w = offset_bottom_rois[3] * spatial_scale;
    float roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    float roi_width = fmaxf(roi_end_w - roi_start_w, 1.f);
    float roi_height = fmaxf(roi_end_h - roi_start_h, 1.f);
    float bin_size_h = roi_height / aligned_height;
    float bin_size_w = roi_width / aligned_width;

    const float* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / aligned_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / aligned_width);

    // We do average (integral) pooling inside a bin
    const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    float output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
        const float y = roi_start_h + ph * bin_size_h +
            (iy + .5f) * bin_size_h / roi_bin_grid_h;  // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            const float x = roi_start_w + pw * bin_size_w +
            (ix + .5f) * bin_size_w / roi_bin_grid_w;

            float val = bilinear_interpolate_cpu(
                offset_bottom_data, height, width, y, x, index);
            output_val += val;
        }
    }
    output_val /= count;

    top_data[index] = output_val;
}

void roi_align_forward_cpu(HostContext &ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    int aligned_height, aligned_width, sampling_ratio; 
    float spatial_scale;
    SSAttrs(attr)
        .get<int>("aligned_height", aligned_height)
        .get<int>("aligned_width", aligned_width)
        .get<float>("spatial_scale", spatial_scale)
        .get<int>("sampling_ratio", sampling_ratio)
        .done();

    const auto& features = ins[0];
    const auto& rois = ins[1];

    auto& output = outs[0];    

    // TODO(lizhouyang): check if darrays are contiguous.

    auto features_view = features.ptr<float>();
    auto rois_view = rois.ptr<float>();
    auto output_view = output.ptr<float>();

    // Number of ROIs
    int num_rois = rois.shape().dim(0);
    //int size_rois = rois.shape().dim(1);

    // data height
    int data_height = features.shape().dim(2);
    // data width
    int data_width = features.shape().dim(3);
    // Number of channels
    int num_channels = features.shape().dim(1);
    int out_count = num_rois*num_channels*aligned_height*aligned_width;
    for (int index = 0; index < out_count; ++index) {
        forward_cpu_kernel(index, features_view, spatial_scale,
                data_height, data_width, num_channels, 
                aligned_height, aligned_width,
                sampling_ratio, rois_view, output_view);
    }
}

void backward_cpu_kernel(const int index, const float* top_diff, 
        const float spatial_scale, const int height, const int width,
        const int channels, const int aligned_height, const int aligned_width,
        const int sampling_ratio, float* bottom_diff, const float* bottom_rois) {
        // (n, c, ph, pw) is an element in the aligned output
    int pw = index % aligned_width;
    int ph = (index / aligned_width) % aligned_height;
    int c  = (index / aligned_width / aligned_height) % channels;
    int n  = index / aligned_width / aligned_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    float roi_start_w = offset_bottom_rois[1] * spatial_scale;
    float roi_start_h = offset_bottom_rois[2] * spatial_scale;
    float roi_end_w = offset_bottom_rois[3] * spatial_scale;
    float roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    float roi_width = fmaxf(roi_end_w - roi_start_w, 1.f);
    float roi_height = fmaxf(roi_end_h - roi_start_h, 1.f);
    float bin_size_h = roi_height / aligned_height;
    float bin_size_w = roi_width / aligned_width;

    float* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * aligned_height * aligned_width;
    const float* offset_top_diff = top_diff + top_offset;
    const float top_diff_this_bin = offset_top_diff[ph * aligned_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / aligned_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / aligned_width);

    // We do average (integral) pooling inside a bin
    const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {// e.g., iy = 0, 1
        const float y = roi_start_h + ph * bin_size_h +
            (iy + .5f) * bin_size_h / roi_bin_grid_h; // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            const float x = roi_start_w + pw * bin_size_w +
                (ix + .5f) * bin_size_w / roi_bin_grid_w;
            bilinear_interpolate_gradient_cpu(
                    height, width,
                    y, x, index,
                    top_diff_this_bin/count,
                    offset_bottom_diff);
        } // ix
    } // iy
} // 

void roi_align_backward_cpu(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    int aligned_height, aligned_width, sampling_ratio;
    float spatial_scale;
    SSAttrs(attr)
        .get<int>("aligned_height", aligned_height)
        .get<int>("aligned_width", aligned_width)
        .get<float>("spatial_scale", spatial_scale)
        .get<int>("sampling_ratio", sampling_ratio)
        .done();

    const auto& top_grad = ins[0];
    const auto& rois = ins[1];

    auto& bottom_grad= outs[0];    

    // Grab the input tensor
    auto top_grad_flat = top_grad.ptr<float>();
    auto rois_flat = rois.ptr<float>();

    auto bottom_grad_flat = bottom_grad.ptr<float>();

    // Number of ROIs
    int num_rois = rois.shape().dim(0);
    //int size_rois = rois.shape().dim(1);

    // batch size
    //int batch_size = bottom_grad.shape().dim(0);
    // data height
    int data_height = bottom_grad.shape().dim(2);
    // data width
    int data_width = bottom_grad.shape().dim(3);
    // Number of channels
    int num_channels = bottom_grad.shape().dim(1);
    int out_count = num_rois*num_channels*aligned_height*aligned_width;
    for (int index = 0; index < out_count; ++index){
        backward_cpu_kernel(index, top_grad_flat, spatial_scale,
                data_height, data_width, num_channels, 
                aligned_height, aligned_width, sampling_ratio,
                bottom_grad_flat, rois_flat);
    }

}


#ifdef PARROTS_USE_CUDA

int ROIAlignForwardLaucher(
    const DArrayLite bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int aligned_height,
    const int aligned_width,  const int sampling_ratio, const DArrayLite bottom_rois,
    DArrayLite top_data, cudaStream_t stream);

void roi_align_forward_cuda(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    int aligned_height, aligned_width, sampling_ratio;
    float spatial_scale;
    SSAttrs(attr)
        .get<int>("aligned_height", aligned_height)
        .get<int>("aligned_width", aligned_width)
        .get<float>("spatial_scale", spatial_scale)
        .get<int>("sampling_ratio", sampling_ratio)
        .done();

    const auto& features = ins[0];
    const auto& rois = ins[1];

    auto& output = outs[0];    

    // Number of ROIs
    int num_rois = rois.shape().dim(0);
    int size_rois = rois.shape().dim(1);
    PARROTS_CHECKARGS(size_rois==5);

    // data height
    int data_height = features.shape().dim(2);
    // data width
    int data_width = features.shape().dim(3);
    // Number of channels
    int num_channels = features.shape().dim(1);
    
    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
    ROIAlignForwardLaucher(
        features, spatial_scale, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, sampling_ratio, rois,
        output, stream);
}
int ROIAlignBackwardLaucher(const DArrayLite top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int aligned_height,
    const int aligned_width,  const int sampling_ratio, const DArrayLite bottom_rois,
    DArrayLite bottom_diff, cudaStream_t stream);

void roi_align_backward_cuda(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    int aligned_height, aligned_width, sampling_ratio;
    float spatial_scale;
    SSAttrs(attr)
        .get<int>("aligned_height", aligned_height)
        .get<int>("aligned_width", aligned_width)
        .get<float>("spatial_scale", spatial_scale)
        .get<int>("sampling_ratio", sampling_ratio)
        .done();

    const auto& top_grad = ins[0];
    const auto& rois = ins[1];

    auto& bottom_grad= outs[0];    

    // Number of ROIs
    int num_rois = rois.shape().dim(0);
    int size_rois = rois.shape().dim(1);
    PARROTS_CHECKARGS(size_rois==5);

    // batch size
    int batch_size = bottom_grad.shape().dim(0);
    // data height
    int data_height = bottom_grad.shape().dim(2);
    // data width
    int data_width = bottom_grad.shape().dim(3);
    // Number of channels
    int num_channels = bottom_grad.shape().dim(1);
    
    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
    ROIAlignBackwardLaucher(
        top_grad, spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, sampling_ratio, rois,
        bottom_grad, stream);
}

#endif

PARROTS_EXTENSION_REGISTER(roi_align_forward)
    .attr("aligned_height")
    .attr("aligned_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .input(2)
    .output(1)
    .apply(roi_align_forward_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(roi_align_forward_cuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(roi_align_backward)
    .attr("aligned_height")
    .attr("spatial_scale")
    .attr("aligned_width")
    .attr("sampling_ratio")
    .input(2)
    .output(1)
    .apply(roi_align_backward_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(roi_align_backward_cuda)
#endif
    .done();

