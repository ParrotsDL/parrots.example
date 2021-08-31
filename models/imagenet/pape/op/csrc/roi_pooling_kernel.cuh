#ifndef ROI_POOLING_KERNEL_CUH
#define ROI_POOLING_KERNEL_CUH

#include <cuda.h>

#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


template <typename scalar_t>
__global__ void ROIPoolForward(
    const int nthreads, 
    const scalar_t* bottom_data,
    const scalar_t spatial_scale, const int height, const int width,
    const int channels, const int pooled_height, const int pooled_width,
    const scalar_t* bottom_rois, scalar_t* top_data, int* argmax_data){
    CUDA_KERNEL_LOOP(index, nthreads){
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c  = (index / pooled_width / pooled_height) % channels;
        int n  = index / pooled_width / pooled_height / channels;

        int roi_batch_ind = bottom_rois[n * 5 + 0];
        int roi_start_w = round(bottom_rois[n * 5 + 1] * spatial_scale);
        int roi_start_h = round(bottom_rois[n * 5 + 2] * spatial_scale);
        int roi_end_w = round(bottom_rois[n * 5 + 3] * spatial_scale);
        int roi_end_h = round(bottom_rois[n * 5 + 4] * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1);
        int roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);
        scalar_t bin_size_h = (scalar_t)(roi_height) / (scalar_t)(pooled_height);
        scalar_t bin_size_w = (scalar_t)(roi_width) / (scalar_t)(pooled_width);

        int hstart = (int)(floor((scalar_t)(ph) * bin_size_h));
        int wstart = (int)(floor((scalar_t)(pw) * bin_size_w));
        int hend = (int)(ceil((scalar_t)(ph + 1) * bin_size_h));
        int wend = (int)(ceil((scalar_t)(pw + 1) * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = fminf(fmaxf(hstart + roi_start_h, 0), height);
        hend = fminf(fmaxf(hend + roi_start_h, 0), height);
        wstart = fminf(fmaxf(wstart + roi_start_w, 0), width);
        wend = fminf(fmaxf(wend + roi_start_w, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        scalar_t maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        int maxidx = -1;
        // bottom_data += roi_batch_ind * channels * height * width;

        int bottom_data_batch_offset = roi_batch_ind * channels * height * width;
        int bottom_data_offset = bottom_data_batch_offset + c * height * width;

        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                // int bottom_index = (h * width + w) * channels + c;
                // int bottom_index = (c * height + h) * width + w;
                int bottom_index = h * width + w;
                if (bottom_data[bottom_data_offset + bottom_index] > maxval) {
                    maxval = bottom_data[bottom_data_offset + bottom_index];
                    maxidx = bottom_data_offset + bottom_index;
                }
            }
        }
        top_data[index] = maxval;
        if (argmax_data != NULL)
            argmax_data[index] = maxidx;
    }
}


template <typename scalar_t>
__global__ void ROIPoolBackward(const int nthreads, const scalar_t* top_diff,
    const int* argmax_data, const int num_rois, const scalar_t spatial_scale,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width, scalar_t* bottom_diff,
    const scalar_t* bottom_rois) {
    CUDA_1D_KERNEL_LOOP(index, nthreads){

        // (n, c, ph, pw) is an element in the pooled output
        int n = index;
        int w = n % width;
        n /= width;
        int h = n % height;
        n /= height;
        int c = n % channels;
        n /= channels;

        scalar_t gradient = 0;
        // Accumulate gradient over all ROIs that pooled this element
        for (int roi_n = 0; roi_n < num_rois; ++roi_n){
            const scalar_t* offset_bottom_rois = bottom_rois + roi_n * 5;
            int roi_batch_ind = offset_bottom_rois[0];
            // Skip if ROI's batch index doesn't match n
            if (n != roi_batch_ind) {
                continue;
            }

            int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
            int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
            int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
            int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

            // Skip if ROI doesn't include (h, w)
            const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                               h >= roi_start_h && h <= roi_end_h);
            if (!in_roi) {
                continue;
            }

            int offset = roi_n * pooled_height * pooled_width * channels;
            const scalar_t* offset_top_diff = top_diff + offset;
            const int* offset_argmax_data = argmax_data + offset;

            // Compute feasible set of pooled units that could have pooled
            // this bottom unit

            // Force malformed ROIs to be 1x1
            int roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1);
            int roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);

            scalar_t bin_size_h = (scalar_t)(roi_height) / (scalar_t)(pooled_height);
            scalar_t bin_size_w = (scalar_t)(roi_width) / (scalar_t)(pooled_width);

            int phstart = floor((scalar_t)(h - roi_start_h) / bin_size_h);
            int phend = ceil((scalar_t)(h - roi_start_h + 1) / bin_size_h);
            int pwstart = floor((scalar_t)(w - roi_start_w) / bin_size_w);
            int pwend = ceil((scalar_t)(w - roi_start_w + 1) / bin_size_w);

            phstart = fminf(fmaxf(phstart, 0), pooled_height);
            phend = fminf(fmaxf(phend, 0), pooled_height);
            pwstart = fminf(fmaxf(pwstart, 0), pooled_width);
            pwend = fminf(fmaxf(pwend, 0), pooled_width);

            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    if (offset_argmax_data[(c * pooled_height + ph) * pooled_width + pw] == index)
                    {
                        gradient += offset_top_diff[(c * pooled_height + ph) * pooled_width + pw];
                    }
                }
            }
        }
        bottom_diff[index] = gradient;
  }
}

#endif

