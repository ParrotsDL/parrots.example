#ifndef PSROI_MASK_POOLING_KERNEL_CUH
#define PSROI_MASK_POOLING_KERNEL_CUH
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
__global__ void PSROIMaskPoolingForward(
    const int nthreads,
    const scalar_t* bottom_data,
    const float spatial_scale,
    const float roi_scale,
    const float bin_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const scalar_t* bottom_rois,
    const int output_dim,
    const int group_size,
    scalar_t* top_data,
    int* mapping_channel,
    const int shape) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        const scalar_t *rois = bottom_rois + n * shape;

        const int roi_batch_ind = static_cast<int>(rois[0]);

        const scalar_t x1 = rois[1];
        const scalar_t y1 = rois[2];
        const scalar_t x2 = rois[3];
        const scalar_t y2 = rois[4];

        scalar_t w = x2 - x1;
        scalar_t h = y2 - y1;

        scalar_t xc = (x1 + x2) * float(0.5);
        scalar_t yc = (y1 + y2) * float(0.5);

        // Rescale RoIs with regard to roi_scale
        scalar_t xx1 = xc - w * roi_scale * float(0.5);
        scalar_t xx2 = xc + w * roi_scale * float(0.5);
        scalar_t yy1 = yc - h * roi_scale * float(0.5);
        scalar_t yy2 = yc + h * roi_scale * float(0.5);

        scalar_t roi_start_w = round(xx1) * spatial_scale;
        scalar_t roi_start_h = round(yy1) * spatial_scale;
        scalar_t roi_end_w = (round(xx2) + float(1.)) * spatial_scale;
        scalar_t roi_end_h = (round(yy2) + float(1.)) * spatial_scale;

        // Force too small ROIs to be 1 x 1
        scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)(0.1));  // avoid 0
        scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)(0.1));

        // Compute w and h at bottom
        scalar_t bin_size_h = roi_height / static_cast<float>(pooled_height);
        scalar_t bin_size_w = roi_width / static_cast<float>(pooled_width);

        scalar_t delta_h = (bin_size_h * bin_scale - bin_size_h) * float(0.5);
        scalar_t delta_w = (bin_size_w * bin_scale - bin_size_w) * float(0.5);

        int hstart = static_cast<int>(
            floor((static_cast<float>(ph) * bin_size_h + roi_start_h) - delta_h));
        int wstart = static_cast<int>(
            floor((static_cast<float>(pw)* bin_size_w + roi_start_w) - delta_w));
        int hend = static_cast<int>(
            ceil((static_cast<float>(ph + 1) * bin_size_h + roi_start_h) + delta_h));
        int wend = static_cast<int>(
            ceil((static_cast<float>(pw + 1) * bin_size_w + roi_start_w) + delta_w));
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        int gw = pw;
        int gh = ph;
        int c = (ctop * group_size + gh) * group_size + gw;

        const scalar_t *input = bottom_data + (roi_batch_ind * channels + c) * height * width;
        scalar_t out_sum = 0;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int bottom_index = h * width + w;
            out_sum += input[bottom_index];
          }
        }

        scalar_t bin_area = (hend - hstart) * (wend - wstart);
        top_data[index] = is_empty ? scalar_t(0.) : (out_sum / bin_area);
        mapping_channel[index] = c;
    }
  }

template <typename scalar_t>
__global__ void PSROIMaskPoolingBackward(
    const int nthreads,
    const scalar_t* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const float spatial_scale,
    const float roi_scale,
    const float bin_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    scalar_t* bottom_diff,
    const scalar_t* bottom_rois,
    const int shape) {
    CUDA_KERNEL_LOOP(index, nthreads) {
         // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int n = index / pooled_width / pooled_height / output_dim;

        const scalar_t *rois = bottom_rois + n*shape;
        //bottom_rois += n * shape;

        const int roi_batch_ind = static_cast<int>(rois[0]);

        const scalar_t x1 = rois[1];
        const scalar_t y1 = rois[2];
        const scalar_t x2 = rois[3];
        const scalar_t y2 = rois[4];

        scalar_t w = x2 - x1;
        scalar_t h = y2 - y1;

        scalar_t xc = (x1 + x2) * float(0.5);
        scalar_t yc = (y1 + y2) * float(0.5);

        // Rescale RoIs with regard to roi_scale
        scalar_t xx1 = xc - w * roi_scale * float(0.5);
        scalar_t xx2 = xc + w * roi_scale * float(0.5);
        scalar_t yy1 = yc - h * roi_scale * float(0.5);
        scalar_t yy2 = yc + h * roi_scale * float(0.5);

        scalar_t roi_start_w = round(xx1) * spatial_scale;
        scalar_t roi_start_h = round(yy1) * spatial_scale;
        scalar_t roi_end_w = (round(xx2) + float(1.)) * spatial_scale;
        scalar_t roi_end_h = (round(yy2) + float(1.)) * spatial_scale;

        // Force too small ROIs to be 1 x 1
        scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)(0.1));  // avoid 0
        scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)(0.1));

        // Compute w and h at bottom
        scalar_t bin_size_h = roi_height / static_cast<float>(pooled_height);
        scalar_t bin_size_w = roi_width / static_cast<float>(pooled_width);

        scalar_t delta_h = (bin_size_h * bin_scale - bin_size_h) * float(0.5);
        scalar_t delta_w = (bin_size_w * bin_scale - bin_size_w) * float(0.5);

        int hstart = static_cast<int>(
            floor((static_cast<float>(ph) * bin_size_h + roi_start_h) - delta_h));
        int wstart = static_cast<int>(
            floor((static_cast<float>(pw)* bin_size_w + roi_start_w) - delta_w));
        int hend = static_cast<int>(
            ceil((static_cast<float>(ph + 1) * bin_size_h + roi_start_h) + delta_h));
        int wend = static_cast<int>(
            ceil((static_cast<float>(pw + 1) * bin_size_w + roi_start_w) + delta_w));
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Compute c at bottom
        int c = mapping_channel[index];
        scalar_t* offset_bottom_diff = bottom_diff +
            (roi_batch_ind * channels + c) * height * width;
        scalar_t bin_area = (hend - hstart)*(wend - wstart);
        scalar_t diff_val = is_empty ? scalar_t(0.) : top_diff[index] / bin_area;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int bottom_index = h * width + w;
            // caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
            atomicAdd(offset_bottom_diff + bottom_index, diff_val);
          }
        }
    }
  }
#endif

