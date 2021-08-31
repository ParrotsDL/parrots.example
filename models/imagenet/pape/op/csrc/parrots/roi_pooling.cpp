// Copyright (c) 2018, SenseTime.

#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <parrots/darray/darraymath.hpp>

using namespace parrots;  


void roiPoolingForwardHost(HostContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs){
    int pooled_height;
    int pooled_width;
    float spatial_scale;
    SSAttrs(attr).get<int>("pooled_height", pooled_height)
        .get<int>("pooled_width", pooled_width)
        .get<float>("spatial_scale", spatial_scale)
        .done();

    const auto& features = ins[0];
    const auto& rois = ins[1];
    auto& output = outs[0];
    auto data_flat = features.ptr<float>();
    auto rois_flat = rois.ptr<float>();

    auto output_flat = output.ptr<float>();

    int num_rois = rois.shape().dim(0);
    int size_rois = rois.shape().dim(1);
    int batch_size = features.shape().dim(0);
    PARROTS_CHECKARGS(batch_size == 1);
    int data_height = features.shape().dim(1);
    int data_width = features.shape().dim(2);
    int num_channels = features.shape().dim(3);

    fill(ctx, output, -1);
    int index_roi = 0;
    int index_output = 0;
    int n;
    for (n = 0; n < num_rois; ++n){
        int roi_batch_ind = rois_flat[index_roi + 0];
        int roi_start_w = round(rois_flat[index_roi + 1] * spatial_scale);
        int roi_start_h = round(rois_flat[index_roi + 2] * spatial_scale);
        int roi_end_w = round(rois_flat[index_roi + 3] * spatial_scale);
        int roi_end_h = round(rois_flat[index_roi + 4] * spatial_scale);
        int roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);
        int roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1);
        float bin_size_h = (float)(roi_height) / (float)(pooled_height);
        float bin_size_w = (float)(roi_width) / (float)(pooled_width);

        int index_data = roi_batch_ind * data_height * data_width * num_channels;
        const int output_area = pooled_width * pooled_height;

        int c, ph, pw;
        for (ph = 0; ph < pooled_height; ++ph){
            for (pw = 0; pw < pooled_width; ++pw){
                int hstart = (floor((float)(ph) * bin_size_h));
                int wstart = (floor((float)(pw) * bin_size_w));
                int hend = (ceil((float)(ph + 1) * bin_size_h));
                int wend = (ceil((float)(pw + 1) * bin_size_w));

                hstart = fminf(fmaxf(hstart + roi_start_h, 0), data_height);
                hend = fminf(fmaxf(hend + roi_start_h, 0), data_height);
                wstart = fminf(fmaxf(wstart + roi_start_w, 0), data_width);
                wend = fminf(fmaxf(wend + roi_start_w, 0), data_width);

                const int pool_index = index_output + (ph * pooled_width + pw);
                int is_empty = (hend <= hstart) || (wend <= wstart);
                if (is_empty){
                    for (c = 0; c < num_channels * output_area; c += output_area){
                        output_flat[pool_index + c] = 0;
                    }
                }
                else{
                    int h, w, c;
                    for (h = hstart; h < hend; ++h){
                        for (w = wstart; w < wend; ++w){
                            for (c = 0; c < num_channels; ++c){
                                const int index = (h * data_width + w) * num_channels + c;
                                if (data_flat[index_data + index] > output_flat[pool_index + c * output_area]){
                                    output_flat[pool_index + c * output_area] = data_flat[index_data + index];
                                }
                            }
                        }
                    }
                }
            }
        }

        index_roi += size_rois;
        index_output += pooled_height * pooled_width * num_channels;
    }
}

void roiPoolingBackwardHost(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs){
    int pooled_height;
    int pooled_width;
    float spatial_scale;
    SSAttrs(attr).get<int>("pooled_height", pooled_height)
        .get<int>("pooled_width", pooled_width)
        .get<float>("spatial_scale", spatial_scale)
        .done();
    throw "CPU version of roipool backward is not implemented!!!";
}

#ifdef PARROTS_USE_CUDA

int ROIPoolForwardLaucher(
    const DArrayLite bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const DArrayLite bottom_rois,
    DArrayLite top_data, DArrayLite argmax_data, cudaStream_t stream);


int ROIPoolBackwardLaucher(const DArrayLite top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const DArrayLite bottom_rois,
    DArrayLite bottom_diff, const DArrayLite argmax_data, cudaStream_t stream);

void roiPoolingForwardCuda(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs){
    int pooled_height;
    int pooled_width;
    float spatial_scale;
    SSAttrs(attr).get<int>("pooled_height", pooled_height)
        .get<int>("pooled_width", pooled_width)
        .get<float>("spatial_scale", spatial_scale)
        .done();

    const auto& features = ins[0];
    const auto& rois = ins[1];

    auto& output = outs[0];
    auto& argmax = outs[1];

    int num_rois = rois.shape().dim(0);
    int size_rois = rois.shape().dim(1);
    PARROTS_CHECKARGS(size_rois == 5);

    int data_height = features.shape().dim(2);
    int data_width = features.shape().dim(3);
    int num_channels = features.shape().dim(1);

    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());

    ROIPoolForwardLaucher(
        features, spatial_scale, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois,
        output, argmax, stream);
}

void roiPoolingBackwardCuda(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs){
    int pooled_height;
    int pooled_width;
    float spatial_scale;
    SSAttrs(attr).get<int>("pooled_height", pooled_height)
        .get<int>("pooled_width", pooled_width)
        .get<float>("spatial_scale", spatial_scale)
        .done();

    const auto& top_grad = ins[0];
    const auto& rois = ins[1];
    const auto& argmax = ins[2];

    auto& bottom_grad = outs[0];
    int num_rois = rois.shape().dim(0);
    int size_rois = rois.shape().dim(1);

    PARROTS_CHECKARGS(size_rois == 5);
    int batch_size = bottom_grad.shape().dim(0);
    int data_height = bottom_grad.shape().dim(2);
    int data_width = bottom_grad.shape().dim(3);
    int num_channels = bottom_grad.shape().dim(1);

    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());

    ROIPoolBackwardLaucher(
        top_grad, spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois,
        bottom_grad, argmax, stream);
}
#endif  // PARROTS_USE_CUDA


PARROTS_EXTENSION_REGISTER(roi_pooling_forward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .input(2)
    .output(2)
    .apply(roiPoolingForwardHost)
#ifdef PARROTS_USE_CUDA
    .apply(roiPoolingForwardCuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(roi_pooling_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .input(3)
    .output(1)
    .apply(roiPoolingBackwardHost)
#ifdef PARROTS_USE_CUDA
    .apply(roiPoolingBackwardCuda)
#endif
    .done();
