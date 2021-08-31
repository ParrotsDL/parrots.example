#ifndef PSROI_POOLING_KERNEL_CUH
#define PSROI_POOLING_KERNEL_CUH


#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void PSROIPoolingForward(
    const int nthreads,
    const scalar_t* bottom_data,
    const scalar_t spatial_scale,
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
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      int roi_batch_ind = (int)bottom_rois[n * shape + 0];
      scalar_t roi_start_w = static_cast<scalar_t>(round(bottom_rois[n * shape + 1])) * spatial_scale;
      scalar_t roi_start_h = static_cast<scalar_t>(round(bottom_rois[n * shape + 2])) * spatial_scale;
      scalar_t roi_end_w = static_cast<scalar_t>(round(bottom_rois[n * shape + 3]) + 1.) * spatial_scale;
      scalar_t roi_end_h = static_cast<scalar_t>(round(bottom_rois[n * shape + 4]) + 1.) * spatial_scale;

      scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)(0.1)); //avoid 0
      scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)(0.1));

      scalar_t bin_size_h = roi_height / static_cast<scalar_t>(pooled_height);
      scalar_t bin_size_w = roi_width / static_cast<scalar_t>(pooled_width);

      int hstart = (int)floor(static_cast<scalar_t>(ph) * bin_size_h
                          + roi_start_h);
      int wstart = (int)floor(static_cast<scalar_t>(pw)* bin_size_w
                          + roi_start_w);
      int hend = (int)ceil(static_cast<scalar_t>(ph + 1) * bin_size_h
                        + roi_start_h);
      int wend = (int)ceil(static_cast<scalar_t>(pw + 1) * bin_size_w
                        + roi_start_w);
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0),width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int gw = pw;
      int gh = ph;
      int c = (ctop*group_size + gh)*group_size + gw;

      bottom_data += (roi_batch_ind * channels + c) * height * width;
      scalar_t out_sum = 0;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          out_sum += bottom_data[bottom_index];
        }
      }

      scalar_t bin_area = (hend - hstart)*(wend - wstart);
      top_data[index] = is_empty? scalar_t(0.) : out_sum/bin_area;
      mapping_channel[index] = c;
    }
  }

template <typename scalar_t>
__global__ void PSROIPoolingBackward(
    const int nthreads,
    const scalar_t* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const scalar_t spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    scalar_t* bottom_diff,
    const scalar_t* bottom_rois,
    const int shape) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      int roi_batch_ind = (int)bottom_rois[n * shape + 0];
      scalar_t roi_start_w = static_cast<scalar_t>(round(bottom_rois[n * shape + 1])) * spatial_scale;
      scalar_t roi_start_h = static_cast<scalar_t>(round(bottom_rois[n * shape + 2])) * spatial_scale;
      scalar_t roi_end_w = static_cast<scalar_t>(round(bottom_rois[n * shape + 3]) + 1.) * spatial_scale;
      scalar_t roi_end_h = static_cast<scalar_t>(round(bottom_rois[n * shape + 4]) + 1.) * spatial_scale;

      scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)(0.1)); //avoid 0
      scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)(0.1));

      scalar_t bin_size_h = roi_height / static_cast<scalar_t>(pooled_height);
      scalar_t bin_size_w = roi_width / static_cast<scalar_t>(pooled_width);

      int hstart = (int)floor(static_cast<scalar_t>(ph)* bin_size_h
        + roi_start_h);
      int wstart = (int)floor(static_cast<scalar_t>(pw)* bin_size_w
        + roi_start_w);
      int hend = (int)ceil(static_cast<scalar_t>(ph + 1) * bin_size_h
        + roi_start_h);
      int wend = (int)ceil(static_cast<scalar_t>(pw + 1) * bin_size_w
        + roi_start_w);
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int c = mapping_channel[index];
      scalar_t* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
      scalar_t bin_area = (hend - hstart)*(wend - wstart);
      scalar_t diff_val = is_empty ? scalar_t(0.) : top_diff[index] / bin_area;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          atomicAdd(offset_bottom_diff + bottom_index, diff_val);
        }
      }
    }
  }

#endif

