#ifndef PSROI_ALIGN_KERNEL_CUH
#define PSROI_ALIGN_KERNEL_CUH

#include <cuda.h>

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


/*** Forward ***/
template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(
    const scalar_t *bottom_data, const int height, const int width, scalar_t y,
    scalar_t x, const int index /* index for debug only*/) {
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
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  scalar_t v1 = bottom_data[y_low * width + x_low];
  scalar_t v2 = bottom_data[y_low * width + x_high];
  scalar_t v3 = bottom_data[y_high * width + x_low];
  scalar_t v4 = bottom_data[y_high * width + x_high];
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename scalar_t>
__global__ void
PSROIAlignForward(const int nthreads, const scalar_t *bottom_data,
                  const scalar_t spatial_scale, const int channels,
                  const int height, const int width, const int pooled_height,
                  const int pooled_width, const scalar_t *bottom_rois,
                  const int output_dim, const int group_size,
                  const int sampling_ratio, scalar_t *top_data,
                  int *mapping_channel, const int shape) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    // liqq 2016/09/25
    // bottom_rois += n * shape;

    int roi_batch_ind = (int)bottom_rois[n * shape + 0];
    scalar_t roi_start_w =
        static_cast<scalar_t>(bottom_rois[n * shape + 1]) * spatial_scale;
    scalar_t roi_start_h =
        static_cast<scalar_t>(bottom_rois[n * shape + 2]) * spatial_scale;
    scalar_t roi_end_w =
        static_cast<scalar_t>(bottom_rois[n * shape + 3] + 1.) * spatial_scale;
    scalar_t roi_end_h =
        static_cast<scalar_t>(bottom_rois[n * shape + 4] + 1.) * spatial_scale;

    // Force too small ROIs to be 1x1
    scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)(0.1)); // avoid 0
    scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)(0.1));

    // Compute w and h at bottom
    scalar_t bin_size_h = roi_height / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = roi_width / static_cast<scalar_t>(pooled_width);
    int gw = pw;
    int gh = ph;
    int c = (ctop * group_size + gh) * group_size + gw;

    const scalar_t *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : (int)ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : (int)ceil(roi_width / pooled_width);

    // use max pooling
    scalar_t maxval = -1E+10;
    int maxidx = -1;

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const scalar_t y =
          roi_start_h + ph * bin_size_h +
          (iy + .5f) * bin_size_h / roi_bin_grid_h; // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t x = roi_start_w + pw * bin_size_w +
                           (ix + .5f) * bin_size_w / roi_bin_grid_w;

        scalar_t val = bilinear_interpolate(offset_bottom_data, height, width,
                                            y, x, index);
        int bottom_index = iy * roi_bin_grid_w + ix;
        if (val > maxval) {
          maxval = val;
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    mapping_channel[index] = maxidx;
  }
}


/*** Backward ***/
template <typename scalar_t>
inline __device__ scalar_t gpu_atomic_add(scalar_t val, scalar_t *address);
template <typename scalar_t>
inline __device__ scalar_t gpu_atomic_add(scalar_t val, scalar_t *address) {
  return atomicAdd(address, val);
}
template <typename scalar_t>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, scalar_t y, scalar_t x, scalar_t &w1,
    scalar_t &w2, scalar_t &w3, scalar_t &w4, int &x_low, int &x_high,
    int &y_low, int &y_high, const int index /* index for debug only*/) {
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
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly, hx = 1. - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename scalar_t>
__global__ void PSROIAlignBackward(
    const int nthreads, const scalar_t *top_diff, const int *mapping_channel,
    const scalar_t spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int output_dim, const int group_size, const int sampling_ratio,
    scalar_t *bottom_diff, const scalar_t *bottom_rois, const int shape) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    // liqq 2016/09/25
    // bottom_rois += n * shape;
    // Do not using rounding; this implementation detail is critical
    int roi_batch_ind = (int)bottom_rois[n * shape + 0];
    scalar_t roi_start_w =
        static_cast<scalar_t>(bottom_rois[n * shape + 1]) * spatial_scale;
    scalar_t roi_start_h =
        static_cast<scalar_t>(bottom_rois[n * shape + 2]) * spatial_scale;
    scalar_t roi_end_w =
        static_cast<scalar_t>(bottom_rois[n * shape + 3] + 1.) * spatial_scale;
    scalar_t roi_end_h =
        static_cast<scalar_t>(bottom_rois[n * shape + 4] + 1.) * spatial_scale;

    // Force too small ROIs to be 1x1
    scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)(0.1)); // avoid 0
    scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)(0.1));

    // Compute w and h at bottom
    scalar_t bin_size_h = roi_height / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = roi_width / static_cast<scalar_t>(pooled_width);

    int gw = pw;
    int gh = ph;
    int c = (ctop * group_size + gh) * group_size + gw;

    scalar_t *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * output_dim + ctop) * pooled_height * pooled_width;
    scalar_t top_diff_this_bin =
        top_diff[top_offset + ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : (int)ceil(roi_height / pooled_height); // e.g. = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : (int)ceil(roi_width / pooled_width);

    int maxidx = mapping_channel[top_offset + ph * pooled_width + pw];
    int iy = maxidx / roi_bin_grid_w;
    int ix = maxidx % roi_bin_grid_w;

    scalar_t y = roi_start_h + ph * bin_size_h +
                 static_cast<float>(iy + .5f) * bin_size_h /
                     static_cast<float>(roi_bin_grid_h); // e.g. 0.5, 1.5
    scalar_t x = roi_start_w + pw * bin_size_w +
                 static_cast<float>(ix + .5f) * bin_size_w /
                     static_cast<float>(roi_bin_grid_w);

    scalar_t w1, w2, w3, w4;
    int x_low, x_high, y_low, y_high;

    // bilinear_interpolation_gradient
    bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4, x_low,
                                  x_high, y_low, y_high, index);

    scalar_t g1 = top_diff_this_bin * w1;
    scalar_t g2 = top_diff_this_bin * w2;
    scalar_t g3 = top_diff_this_bin * w3;
    scalar_t g4 = top_diff_this_bin * w4;

    if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
      gpu_atomic_add<scalar_t>(g1, offset_bottom_diff + y_low * width + x_low);
      gpu_atomic_add<scalar_t>(g2, offset_bottom_diff + y_low * width + x_high);
      gpu_atomic_add<scalar_t>(g3, offset_bottom_diff + y_high * width + x_low);
      gpu_atomic_add<scalar_t>(g4,
                               offset_bottom_diff + y_high * width + x_high);
    }
  }
}

#endif