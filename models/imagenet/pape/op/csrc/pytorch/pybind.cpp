#include "pytorch_cpp_helper.hpp"

using namespace at;

int deform_conv_forward(Tensor input, Tensor weight,
                        Tensor offset, Tensor output,
                        Tensor columns, Tensor ones, int kW,
                        int kH, int dW, int dH, int padW, int padH,
                        int dilationH, int dilationW, int groups,
                        int deformable_group);

int deform_conv_backward_input(
    Tensor input, Tensor weight, Tensor offset, Tensor gradOutput,
    Tensor gradInput, Tensor gradOffset,
    Tensor columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationH, int dilationW, int groups, int deformable_group);

int deform_conv_backward_parameters(
    Tensor input, Tensor offset, Tensor gradOutput,
    Tensor gradWeight, Tensor columns, Tensor ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW, int groups, int deformable_group,
    float scale);

void deform_psroi_pooling_cuda_forward(at::Tensor input, at::Tensor bbox,
                                       at::Tensor trans,
                                       at::Tensor out, at::Tensor top_count,
                                       const int no_trans,
                                       const float spatial_scale,
                                       const int output_dim,
                                       const int group_size,
                                       const int pooled_size,
                                       const int part_size,
                                       const int sample_per_part,
                                       const float trans_std);

void deform_psroi_pooling_cuda_backward(at::Tensor out_grad,
                                        at::Tensor input, at::Tensor bbox,
                                        at::Tensor trans, at::Tensor top_count,
                                        at::Tensor input_grad, at::Tensor trans_grad,
                                        const int no_trans,
                                        const float spatial_scale,
                                        const int output_dim,
                                        const int group_size,
                                        const int pooled_size,
                                        const int part_size,
                                        const int sample_per_part,
                                        const float trans_std);

int focal_loss_sigmoid_forward(
                           Tensor logits,
                           Tensor targets,
                           Tensor losses,
                           int N,
                           float weight_pos,
                           float gamma, 
                           float alpha,
                           int num_classes);

int focal_loss_sigmoid_backward(
                           Tensor logits,
                           Tensor targets,
                           Tensor dX_data,
                           int N,
                           float weight_pos,
                           float gamma,
                           float alpha,
                           int num_classes);

int focal_loss_softmax_forward(
                           Tensor logits,
                           Tensor targets,
                           Tensor losses,
                           Tensor priors,
                           int N,
                           float weight_pos,
                           float gamma, 
                           float alpha,
                           int num_classes);

int focal_loss_softmax_backward(
                           Tensor logits,
                           Tensor targets,
                           Tensor priors,
                           Tensor dX_data,
                           Tensor buff,
                           int N,
                           float weight_pos,
                           float gamma,
                           float alpha,
                           int num_classes);

void iou_overlaps(Tensor bboxes1, Tensor bboxes2, Tensor output, const int mode, const int offset);

void modulated_deform_conv_cuda_forward(at::Tensor input, at::Tensor weight,
                                        at::Tensor bias, at::Tensor ones,
                                        at::Tensor offset, at::Tensor mask,
                                        at::Tensor output, at::Tensor columns,
                                        int kernel_h, int kernel_w,
                                        const int stride_h, const int stride_w,
                                        const int pad_h, const int pad_w,
                                        const int dilation_h, const int dilation_w, const int group,
                                        const int deformable_group, const bool with_bias);

void modulated_deform_conv_cuda_backward(at::Tensor input, at::Tensor weight,
                                         at::Tensor bias, at::Tensor ones,
                                         at::Tensor offset, at::Tensor mask,
                                         at::Tensor columns,
                                         at::Tensor grad_input, at::Tensor grad_weight,
                                         at::Tensor grad_bias, at::Tensor grad_offset,
                                         at::Tensor grad_mask, at::Tensor grad_output,
                                         int kernel_h, int kernel_w,
                                         int stride_h, int stride_w,
                                         int pad_h, int pad_w,
                                         int dilation_h, int dilation_w, int group,
                                         int deformable_group, const bool with_bias);

int nms(Tensor boxes, Tensor order, Tensor areas, Tensor keep, Tensor num_out, float nms_overlap_thresh, int offset);

int psroi_align_forward_cuda(Tensor features,
                             Tensor rois,
                             Tensor output,
                             Tensor mapping_channel,
                             int pooled_height,
                             int pooled_width,
                             int output_dim,
                             float spatial_scale,
                             int sampling_ratio
                             );

int psroi_align_backward_cuda(Tensor top_grad,
                              Tensor rois,
                              Tensor mapping_channel,
                              Tensor bottom_grad,
                              int pooled_height,
                              int pooled_width,
                              int output_dim,
                              float spatial_scale,
                              int sampling_ratio);

int psroi_mask_pooling_forward(Tensor features,
            Tensor rois,
            Tensor output,
            Tensor mapping_channel,
            int pooled_height,
            int pooled_width,
            int output_dim,
            float spatial_scale,
            float roi_scale,
            float bin_scale);

int psroi_mask_pooling_backward(Tensor top_grad,
             Tensor rois,
             Tensor mapping_channel,
             Tensor bottom_grad,
             int pooled_height,
             int pooled_width,
             int output_dim,
             float spatial_scale,
             float roi_scale,
             float bin_scale);

int psroi_pooling_forward(Tensor features,
            Tensor rois,
            Tensor output,
            Tensor mapping_channel,
            int pooled_height,
            int pooled_width,
            int output_dim,
            float spatial_scale);

int psroi_pooling_backward(Tensor top_grad,
             Tensor rois,
             Tensor mapping_channel,
             Tensor bottom_grad,
             int pooled_height,
             int pooled_width,
             int output_dim,
             float spatial_scale);


int roi_align_forward(Tensor features, Tensor rois, Tensor output,
        int aligned_height, int aligned_width,
        float spatial_scale, int sampling_ratio);

int roi_align_backward(Tensor top_grad, Tensor rois, Tensor bottom_grad,
        int aligned_height, int aligned_width,
        float spatial_scale, int sampling_ratio);

int roi_pooling_forward(
    Tensor features,
    Tensor rois,
    Tensor output,
    Tensor argmax,
    int pooled_height,
    int pooled_width,
    float spatial_scale);

int roi_pooling_backward(
    Tensor  top_grad,
    Tensor  rois,
    Tensor  argmax,
    Tensor bottom_grad,
    int pooled_height, 
    int pooled_width, 
    float spatial_scale);

int softnms(Tensor boxes, Tensor areas, Tensor order, Tensor num_out,
              float sigma, float n_thresh, float overlap_thresh, unsigned int method);


void syncbn_forward_step1(const at::Tensor input, at::Tensor mean,
         size_t n, size_t c, size_t h, size_t w);

void syncbn_forward_step2(const at::Tensor input, at::Tensor mean, at::Tensor var,
         size_t n, size_t c, size_t h, size_t w);

void syncbn_forward_step3(const at::Tensor input, at::Tensor mean, at::Tensor var,
         const at::Tensor weight, const at::Tensor bias, at::Tensor running_mean,
         at::Tensor running_var, at::Tensor std, at::Tensor output, 
         size_t n, size_t c, size_t h, size_t w, size_t group_size, const float eps,
         const float momentum);

void syncbn_backward_step1(const at::Tensor input, const at::Tensor mean,
         const at::Tensor std, const at::Tensor grad_output, at::Tensor weight_diff,
         at::Tensor bias_diff, size_t n, size_t c, size_t h, size_t w);

void syncbn_backward_step2(const at::Tensor input, const at::Tensor mean, 
         const at::Tensor weight, const at::Tensor weight_diff, const at::Tensor bias_diff,
         const at::Tensor std, const at::Tensor grad_output, at::Tensor grad_input, size_t n, size_t c, size_t h, size_t w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("offset"),
          py::arg("output"),
          py::arg("columns"),
          py::arg("ones"),
          py::arg("kW"),
          py::arg("kH"),
          py::arg("dW"),
          py::arg("dH"),
          py::arg("padH"),
          py::arg("padW"),
          py::arg("dilationH"),
          py::arg("dilationW"),
          py::arg("groups"),
          py::arg("deformable_group"));
  m.def("deform_conv_backward_input", &deform_conv_backward_input, "deform_conv_backward_input",
          py::arg("input"),
          py::arg("weight"),
          py::arg("offset"),
          py::arg("gradOutput"),
          py::arg("gradInput"),
          py::arg("gradOffset"),
          py::arg("columns"),
          py::arg("kW"),
          py::arg("kH"),
          py::arg("dW"),
          py::arg("dH"),
          py::arg("padH"),
          py::arg("padW"),
          py::arg("dilationH"),
          py::arg("dilationW"),
          py::arg("groups"),
          py::arg("deformable_group"));
  m.def("deform_conv_backward_parameters", &deform_conv_backward_parameters, "deform_conv_backward_parameters",
          py::arg("input"),
          py::arg("offset"),
          py::arg("gradOutput"),
          py::arg("gradWeight"),
          py::arg("columns"),
          py::arg("ones"),
          py::arg("kW"),
          py::arg("kH"),
          py::arg("dW"),
          py::arg("dH"),
          py::arg("padH"),
          py::arg("padW"),
          py::arg("dilationH"),
          py::arg("dilationW"),
          py::arg("groups"),
          py::arg("deformable_group"),
          py::arg("scale"));
    m.def("deform_psroi_pooling_cuda_forward", &deform_psroi_pooling_cuda_forward,
          "deform psroi pooling forward(CUDA)",
          py::arg("data"),
          py::arg("rois"),
          py::arg("offset"),
          py::arg("output"),
          py::arg("output_count"),
          py::arg("no_trans"),
          py::arg("spatial_scale"),
          py::arg("output_dim"),
          py::arg("group_size"),
          py::arg("pooled_size"),
          py::arg("part_size"),
          py::arg("sample_per_part"),
          py::arg("trans_std"));
    m.def("deform_psroi_pooling_cuda_backward", &deform_psroi_pooling_cuda_backward,
          "deform psroi pooling backward(CUDA)",
          py::arg("grad_output"),
          py::arg("data"),
          py::arg("rois"),
          py::arg("offset"),
          py::arg("output_count"),
          py::arg("grad_input"),
          py::arg("grad_offset"),
          py::arg("no_trans"),
          py::arg("spatial_scale"),
          py::arg("output_dim"),
          py::arg("group_size"),
          py::arg("pooled_size"),
          py::arg("part_size"),
          py::arg("sample_per_part"),
          py::arg("trans_std"));
  m.def("focal_loss_sigmoid_forward", &focal_loss_sigmoid_forward, "focal_loss_sigmoid_forward ",
        py::arg("logits"),
        py::arg("targets"),
        py::arg("losses"),
        py::arg("N"),
        py::arg("weight_pos"),
        py::arg("gamma"),
        py::arg("alpha"),
        py::arg("num_classes"));
  m.def("focal_loss_sigmoid_backward", &focal_loss_sigmoid_backward, "focal_loss_sigmoid_backward",
        py::arg("logits"),
        py::arg("targets"),
        py::arg("dX_data"),
        py::arg("N"),
        py::arg("weight_pos"),
        py::arg("gamma"),
        py::arg("alpha"),
        py::arg("num_classes"));
  m.def("focal_loss_softmax_forward", &focal_loss_softmax_forward, "focal_loss_softmax_forward",
        py::arg("logits"),
        py::arg("targets"),
        py::arg("losses"),
        py::arg("priors"),
        py::arg("N"),
        py::arg("weight_pos"),
        py::arg("gamma"),
        py::arg("alpha"),
        py::arg("num_classes"));
  m.def("focal_loss_softmax_backward", &focal_loss_softmax_backward, "focal_loss_softmax_backward",
        py::arg("logits"),
        py::arg("targets"),
        py::arg("priors"),
        py::arg("dX_data"),
        py::arg("buff"),
        py::arg("N"),
        py::arg("weight_pos"),
        py::arg("gamma"),
        py::arg("alpha"),
        py::arg("num_classes"));
    m.def("iou_overlap", &iou_overlaps, "iou_overlaps_ext",
          py::arg("bboxes1"),
          py::arg("bboxes2"),
          py::arg("output"),
          py::arg("mode"),
          py::arg("offset"));
    m.def("modulated_deform_conv_cuda_forward", &modulated_deform_conv_cuda_forward,
          "modulated deform conv forward (CUDA)",
          py::arg("input"), py::arg("weight"),
          py::arg("bias"), py::arg("ones"),
          py::arg("offset"), py::arg("mask"),
          py::arg("output"), py::arg("columns"),
          py::arg("kernel_h"), py::arg("kernel_w"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("pad_h"), py::arg("pad_w"),
          py::arg("dilation_h"), py::arg("dilation_w"), py::arg("group"),
          py::arg("deformable_group"), py::arg("with_bias"));

    m.def("modulated_deform_conv_cuda_backward", &modulated_deform_conv_cuda_backward,
          "modulated deform conv backward (CUDA)",
          py::arg("input"), py::arg("weight"),
          py::arg("bias"), py::arg("ones"),
          py::arg("offset"), py::arg("mask"),
          py::arg("columns"),
          py::arg("grad_input"), py::arg("grad_weight"),
          py::arg("grad_bias"), py::arg("grad_offset"),
          py::arg("grad_mask"), py::arg("grad_output"),
          py::arg("kernel_h"), py::arg("kernel_w"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("pad_h"), py::arg("pad_w"),
          py::arg("dilation_h"), py::arg("dilation_w"), py::arg("group"),
          py::arg("deformable_group"), py::arg("with_bias"));
  m.def("nms", &nms, "naive_nms (CPU/CUDA) ",
        py::arg("boxes"),
        py::arg("order"),
        py::arg("areas"),
        py::arg("keep"),
        py::arg("num_out"),
        py::arg("nms_overlap_thresh"),
        py::arg("offset"));
  m.def("psroi_align_forward", &psroi_align_forward_cuda, "roi_align forward",
        py::arg("features"),
        py::arg("rois"),
        py::arg("output"),
        py::arg("mapping_channel"),
        py::arg("pooled_height"),
        py::arg("pooled_width"),
        py::arg("output_dim"),
        py::arg("spatial_scale"),
        py::arg("sampling_ratio"));
  m.def("psroi_align_backward", &psroi_align_backward_cuda, "roi_align backward",
        py::arg("grad_output"),
        py::arg("rois"),
        py::arg("mapping_channel"),
        py::arg("grad_input"),
        py::arg("pooled_height"),
        py::arg("pooled_width"),
        py::arg("output_dim"),
        py::arg("spatial_scale"),
        py::arg("sampling_ratio"));
  m.def("psroi_mask_pooling_forward", &psroi_mask_pooling_forward, "psroi_mask_pooling forward",
        py::arg("features"),
        py::arg("rois"),
        py::arg("output"),
        py::arg("mapping_channel"),
        py::arg("pooled_height"),
        py::arg("pooled_width"),
        py::arg("output_dim"),
        py::arg("spatial_scale"),
        py::arg("roi_scale"),
        py::arg("bin_scale"));
  m.def("psroi_mask_pooling_backward", &psroi_mask_pooling_backward, "psroi_mask_pooling backward",
        py::arg("grad_output"),
        py::arg("rois"),
        py::arg("mapping_channel"),
        py::arg("grad_input"),
        py::arg("pooled_height"),
        py::arg("pooled_width"),
        py::arg("output_dim"),
        py::arg("spatial_scale"),
        py::arg("roi_scale"),
        py::arg("bin_scale"));
  m.def("psroi_pooling_forward", &psroi_pooling_forward, "psroi_pooling forward",
        py::arg("features"),
        py::arg("rois"),
        py::arg("mapping_channel"),
        py::arg("bottom_grad"),
        py::arg("pooled_height"),
        py::arg("pooled_width"),
        py::arg("output_dim"),
        py::arg("spatial_scale"));
  m.def("psroi_pooling_backward", &psroi_pooling_backward, "psroi_pooling backward",
        py::arg("top_grad"),
        py::arg("rois"),
        py::arg("mapping_channel"),
        py::arg("bottom_grad"),
        py::arg("pooled_height"),
        py::arg("pooled_width"),
        py::arg("output_dim"),
        py::arg("spatial_scale"));
  m.def("roi_align_forward", &roi_align_forward, "roi_align forward",
        py::arg("features"),
        py::arg("rois"),
        py::arg("output"),
        py::arg("aligned_height"),
        py::arg("aligned_width"),
        py::arg("spatial_scale"),
        py::arg("sampling_ratio"));
  m.def("roi_align_backward", &roi_align_backward, "roi_align backward",
        py::arg("grad_output"),
        py::arg("rois"),
        py::arg("grad_input"),
        py::arg("aligned_height"),
        py::arg("aligned_width"),
        py::arg("spatial_scale"),
        py::arg("sampling_ratio"));
  m.def("roi_pooling_forward", &roi_pooling_forward, "roi_pooling forward",
        py::arg("features"),
        py::arg("rois"),
        py::arg("output"),
        py::arg("argmax"),
        py::arg("pooled_height"),
        py::arg("pooled_width"),
        py::arg("spatial_scale"));

  m.def("roi_pooling_backward", &roi_pooling_backward, "roi_pooling backward",
        py::arg("top_grad"),
        py::arg("rois"),
        py::arg("argmax"),
        py::arg("bottom_grad"),
        py::arg("pooled_height"),
        py::arg("pooled_width"),
        py::arg("spatial_scale"));
  m.def("softnms", &softnms, "softnms (CPU/CUDA) ",
        py::arg("boxes"),
        py::arg("areas"),
        py::arg("order"),
        py::arg("num_out"),
        py::arg("sigma"),
        py::arg("n_thresh"),
        py::arg("overlap_thresh"),
        py::arg("method"));
  m.def("syncbn_forward_step1", &syncbn_forward_step1, "SyncBN forward_step1 (CUDA)",
        py::arg("input"),
        py::arg("mean"),
        py::arg("n"),
        py::arg("c"),
        py::arg("h"),
        py::arg("w"));
  m.def("syncbn_forward_step2", &syncbn_forward_step2, "SyncBN forward_step2 (CUDA)",
        py::arg("input"),
        py::arg("mean"),
        py::arg("var"),
        py::arg("n"),
        py::arg("c"),
        py::arg("h"),
        py::arg("w"));
  m.def("syncbn_forward_step3", &syncbn_forward_step3, "SyncBN forward_step3 (CUDA)",
        py::arg("input"),
        py::arg("mean"),
        py::arg("var"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("running_mean"),
        py::arg("running_var"),
        py::arg("std"),
        py::arg("output"),
        py::arg("n"),
        py::arg("c"),
        py::arg("h"),
        py::arg("w"),
        py::arg("group_size"),
        py::arg("eps"),
        py::arg("momentum"));
  m.def("syncbn_backward_step1", &syncbn_backward_step1, "SyncBN backward_step1 (CUDA)",
        py::arg("input"),
        py::arg("mean"),
        py::arg("std"),
        py::arg("grad_output"),
        py::arg("weight_diff"),
        py::arg("bias_diff"),
        py::arg("n"),
        py::arg("c"),
        py::arg("h"),
        py::arg("w"));
  m.def("syncbn_backward_step2", &syncbn_backward_step2, "SyncBN backward_step2 (CUDA)",
        py::arg("input"),
        py::arg("mean"),
        py::arg("weight"),
        py::arg("weight_diff"),
        py::arg("bias_diff"),
        py::arg("std"),
        py::arg("grad_output"),
        py::arg("grad_input"),
        py::arg("n"),
        py::arg("c"),
        py::arg("h"),
        py::arg("w"));
}