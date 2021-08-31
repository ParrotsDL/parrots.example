#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include "pytorch_cpp_helper.hpp"

using at::Tensor;

#define DIVUP(x, y) (((x) + (y) - 1) / (y))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

void _softnms(int boxes_num, Tensor boxes_dev, Tensor mask_dev, float sigma,
              float n_thresh, unsigned int method, float overlap_thresh);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int gpu_softnms(Tensor boxes, Tensor order, Tensor num_out, float sigma, float n_thresh,
            unsigned int method, float overlap_thresh) {
  // boxes has to be sorted
  // THArgCheck(THLongTensor_isContiguous(keep), 0, "boxes must be contiguous");
  // THArgCheck(THCudaTensor_isContiguous(state, boxes), 2, "boxes must be contiguous");
  // CHECK_INPUT(keep);
  CHECK_INPUT(boxes);
  boxes = boxes.contiguous();
  auto options_gpu =
  at::TensorOptions()
    .dtype(at::kLong)
    .layout(at::kStrided)
    .device({at::kCUDA})
    .requires_grad(false);
  auto options_cpu =
  at::TensorOptions()
    .dtype(at::kLong)
    .layout(at::kStrided)
    .device({at::kCPU})
    .requires_grad(false);
  // Number of ROIs
  long boxes_num = boxes.size(0);
  long boxes_dim = boxes.size(1);

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  Tensor keep = at::zeros({boxes_num}, options_gpu);

  _softnms(boxes_num, boxes, keep, sigma, n_thresh, method, overlap_thresh);

  Tensor keep_cpu = at::zeros({boxes_num}, options_cpu);
  keep_cpu.copy_(keep);
  unsigned long long *keep_flat = (unsigned long long *)keep_cpu.data<long>();
  long *order_flat = order.data<long>();

  long i, j=0;
  for (i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(keep_flat[nblock] & (1ULL << inblock))) {
      order_flat[j++] = i; 
    }
  }
  long * num_out_flat = num_out.data<long>();
  num_out_flat[0] = j;
  return 1;
}


int cpu_softnms(Tensor boxes, Tensor order, Tensor areas, Tensor num_out,
                float sigma, float n_thresh, unsigned int method, float overlap_thresh){
    long boxes_num = boxes.size(0);
    long boxes_dim = boxes.size(1);
    float * boxes_flat = boxes.data<float>();
    long * order_flat = order.data<long>();
    float * areas_flat = areas.data<float>();

    // nominal indices
    long i, ti, pos, maxpos;
    // temp variables for box i's (the box currently under consideration)
    float ix1, iy1, ix2, iy2, iarea, iscore;
    // variables for computing overlap with box pos (lower scoring box)
    float xx1, yy1, xx2, yy2;
    float w, h;
    float inter, ovr;
    float maxscore, weight;
    // changes for scores need to be considered as sorting.
    for (i=0; i < boxes_num; ++i) {
        maxscore = boxes_flat[i * boxes_dim + 4];
        maxpos = i;

        ix1 = boxes_flat[i * boxes_dim];
        iy1 = boxes_flat[i * boxes_dim + 1];
        ix2 = boxes_flat[i * boxes_dim + 2];
        iy2 = boxes_flat[i * boxes_dim + 3];
        iscore = boxes_flat[i * boxes_dim + 4];
        iarea = areas_flat[i];
        ti = order_flat[i];

        pos = i + 1;
        while (pos < boxes_num){
            if (maxscore < boxes_flat[pos * boxes_dim + 4]){
                maxscore = boxes_flat[pos * boxes_dim + 4];
                maxpos = pos;
            }
            pos ++;
        }
        boxes_flat[i * boxes_dim] = boxes_flat[maxpos * boxes_dim];
        boxes_flat[i * boxes_dim + 1] = boxes_flat[maxpos * boxes_dim + 1];
        boxes_flat[i * boxes_dim + 2] = boxes_flat[maxpos * boxes_dim + 2];
        boxes_flat[i * boxes_dim + 3] = boxes_flat[maxpos * boxes_dim + 3];
        boxes_flat[i * boxes_dim + 4] = boxes_flat[maxpos * boxes_dim + 4];
        order_flat[i] = order_flat[maxpos];
        areas_flat[i] = areas_flat[maxpos];

        boxes_flat[maxpos * boxes_dim] = ix1;
        boxes_flat[maxpos * boxes_dim + 1] = iy1;
        boxes_flat[maxpos * boxes_dim + 2] = ix2;
        boxes_flat[maxpos * boxes_dim + 3] = iy2;
        boxes_flat[maxpos * boxes_dim + 4] = iscore;
        order_flat[maxpos] = ti;
        areas_flat[maxpos] = iarea;

        ix1 = boxes_flat[i * boxes_dim];
        iy1 = boxes_flat[i * boxes_dim + 1];
        ix2 = boxes_flat[i * boxes_dim + 2];
        iy2 = boxes_flat[i * boxes_dim + 3];
        iscore = boxes_flat[i * boxes_dim + 4];
        
        iarea = areas_flat[i];

        for (pos = i + 1; pos < boxes_num; pos++) {
            xx1 = fmaxf(ix1, boxes_flat[pos * boxes_dim]);
            yy1 = fmaxf(iy1, boxes_flat[pos * boxes_dim + 1]);
            xx2 = fminf(ix2, boxes_flat[pos * boxes_dim + 2]);
            yy2 = fminf(iy2, boxes_flat[pos * boxes_dim + 3]);
            w = fmaxf(0.0, xx2 - xx1 + 1);
            h = fmaxf(0.0, yy2 - yy1 + 1);
            inter = w * h;
            ovr = inter / (iarea + areas_flat[pos] - inter);
            weight = 1.0;
            if (method == 1){  // linear
                if (ovr > n_thresh){
                    weight = 1.0 - ovr;
                }
            }
            else if (method == 2){   //gaussian
                weight = exp(-(ovr * ovr)/sigma);
            }
            else if (ovr >= n_thresh) {   // naive_nms
                weight = 0;
            }
            boxes_flat[pos * boxes_dim + 4] *= weight;
            if (boxes_flat[pos * boxes_dim + 4] < overlap_thresh){
                boxes_flat[pos * boxes_dim] = boxes_flat[(boxes_num-1) * boxes_dim];
                boxes_flat[pos * boxes_dim + 1] = boxes_flat[(boxes_num-1) * boxes_dim + 1];
                boxes_flat[pos * boxes_dim + 2] = boxes_flat[(boxes_num-1) * boxes_dim + 2];
                boxes_flat[pos * boxes_dim + 3] = boxes_flat[(boxes_num-1) * boxes_dim + 3];
                boxes_flat[pos * boxes_dim + 4] = boxes_flat[(boxes_num-1) * boxes_dim + 4];
                order_flat[pos] = order_flat[boxes_num - 1];
                boxes_num --;
                pos --;
            }
        }
    }
    long * num_out_flat = num_out.data<long>();
    num_out_flat[0] = boxes_num;
    return 1;
}


int softnms(Tensor boxes, Tensor areas, Tensor order, Tensor num_out,
              float sigma, float n_thresh, float overlap_thresh, unsigned int method){
    if (boxes.type().is_cuda()){
        // AT_ERROR("No support for gpu softnms!");
        return gpu_softnms(boxes, order, num_out, sigma, n_thresh, method, overlap_thresh);
    }
    else{
        return cpu_softnms(boxes, order, areas, num_out, sigma, n_thresh, method, overlap_thresh);
    }
}
