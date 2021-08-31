// Copyright (c) 2018, SenseTime.

#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>


using namespace parrots;  // NOLINT

void focal_loss_sigmoid_forward_cpu(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    int N;
    float weight_pos;
    float gamma;
    float alpha;
    int num_classes;
    SSAttrs(attr).get<int>("N", N)
        .get<float>("weight_pos", weight_pos)
        .get<float>("gamma", gamma)
        .get<float>("alpha", alpha)
        .get<int>("num_classes", num_classes)
        .done();
    PARROTS_NOT_IMPL << "Not suppport cpu focal loss sigmoid forward!";
}

void focal_loss_sigmoid_backward_cpu(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    int N;
    float weight_pos;
    float gamma;
    float alpha;
    int num_classes;
    SSAttrs(attr).get<int>("N", N)
        .get<float>("weight_pos", weight_pos)
        .get<float>("gamma", gamma)
        .get<float>("alpha", alpha)
        .get<int>("num_classes", num_classes)
        .done();
    PARROTS_NOT_IMPL << "Not suppport cpu focal loss sigmoid backward!";
}

void focal_loss_softmax_forward_cpu(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    int N;
    float weight_pos;
    float gamma;
    float alpha;
    int num_classes;
    SSAttrs(attr).get<int>("N", N)
        .get<float>("weight_pos", weight_pos)
        .get<float>("gamma", gamma)
        .get<float>("alpha", alpha)
        .get<int>("num_classes", num_classes)
        .done();
    PARROTS_NOT_IMPL << "Not suppport cpu focal loss softmax forward!";
}

void focal_loss_softmax_backward_cpu(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    int N;
    float weight_pos;
    float gamma;
    float alpha;
    int num_classes;
    SSAttrs(attr).get<int>("N", N)
        .get<float>("weight_pos", weight_pos)
        .get<float>("gamma", gamma)
        .get<float>("alpha", alpha)
        .get<int>("num_classes", num_classes)
        .done();
    PARROTS_NOT_IMPL << "Not suppport cpu focal loss softmax backward!";
}

#ifdef PARROTS_USE_CUDA
int SigmoidFocalLossForwardLaucher(
    const int N, const DArrayLite logits,
    const DArrayLite targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, DArrayLite losses, cudaStream_t stream);

int SigmoidFocalLossBackwardLaucher(
    const int N, const DArrayLite logits,
    const DArrayLite targets, DArrayLite dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes,
    cudaStream_t stream);

int SoftmaxFocalLossForwardLaucher(
    const int N, const DArrayLite logits,
    const DArrayLite targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, DArrayLite losses,
    DArrayLite priors, cudaStream_t stream);

int SoftmaxFocalLossBackwardLaucher(
    const int N, const DArrayLite logits,
    const DArrayLite targets, DArrayLite dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes,
    const DArrayLite priors, DArrayLite buff, cudaStream_t stream);

void focal_loss_sigmoid_forward_cuda(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    int N;
    float weight_pos;
    float gamma;
    float alpha;
    int num_classes;
    SSAttrs(attr).get<int>("N", N)
        .get<float>("weight_pos", weight_pos)
        .get<float>("gamma", gamma)
        .get<float>("alpha", alpha)
        .get<int>("num_classes", num_classes)
        .done();
    
    // get inputs and outputs
    const auto& logits = ins[0];
    const auto& targets = ins[1];

    auto& losses = outs[0];


    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());

    SigmoidFocalLossForwardLaucher(
        N, logits, targets, weight_pos, 
        gamma, alpha, num_classes, losses, stream);
}

void focal_loss_sigmoid_backward_cuda(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    int N;
    float weight_pos;
    float gamma;
    float alpha;
    int num_classes;
    SSAttrs(attr).get<int>("N", N)
        .get<float>("weight_pos", weight_pos)
        .get<float>("gamma", gamma)
        .get<float>("alpha", alpha)
        .get<int>("num_classes", num_classes)
        .done();
    
    // get inputs and outputs
    const auto& logits = ins[0];
    const auto& targets = ins[1];

    auto& dX_data = outs[0];


    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
    SigmoidFocalLossBackwardLaucher(
        N, logits, targets, dX_data,
        weight_pos, gamma, alpha, num_classes, stream);
}

void focal_loss_softmax_forward_cuda(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    int N;
    float weight_pos;
    float gamma;
    float alpha;
    int num_classes;
    SSAttrs(attr).get<int>("N", N)
        .get<float>("weight_pos", weight_pos)
        .get<float>("gamma", gamma)
        .get<float>("alpha", alpha)
        .get<int>("num_classes", num_classes)
        .done();
    
    // get inputs and outputs
    const auto& logits = ins[0];
    const auto& targets = ins[1];

    auto& losses = outs[0];
    auto& priors = outs[1];


    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());

    SoftmaxFocalLossForwardLaucher(
        N, logits, targets, weight_pos, 
        gamma, alpha, num_classes, losses, priors, stream);
}

void focal_loss_softmax_backward_cuda(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    int N;
    float weight_pos;
    float gamma;
    float alpha;
    int num_classes;
    SSAttrs(attr).get<int>("N", N)
        .get<float>("weight_pos", weight_pos)
        .get<float>("gamma", gamma)
        .get<float>("alpha", alpha)
        .get<int>("num_classes", num_classes)
        .done();
    
    // get inputs and outputs
    const auto& logits = ins[0];
    const auto& targets = ins[1];
    const auto& priors = ins[2];

    auto& dX_data = outs[0];
    auto& buff = outs[1];


    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
    SoftmaxFocalLossBackwardLaucher(
        N, logits, targets, dX_data,
        weight_pos, gamma, alpha, num_classes, priors, buff, stream);
}
#endif  // PARROTS_USE_CUDA


PARROTS_EXTENSION_REGISTER(focal_loss_sigmoid_forward)
    .attr("N")
    .attr("weight_pos")
    .attr("gamma")
    .attr("alpha")
    .attr("num_classes")
    .input(2)
    .output(1)
    .apply(focal_loss_sigmoid_forward_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(focal_loss_sigmoid_forward_cuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(focal_loss_sigmoid_backward)
    .attr("N")
    .attr("weight_pos")
    .attr("gamma")
    .attr("alpha")
    .attr("num_classes")
    .input(2)
    .output(1)
    .apply(focal_loss_sigmoid_backward_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(focal_loss_sigmoid_backward_cuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(focal_loss_softmax_forward)
    .attr("N")
    .attr("weight_pos")
    .attr("gamma")
    .attr("alpha")
    .attr("num_classes")
    .input(2)
    .output(2)
    .apply(focal_loss_softmax_forward_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(focal_loss_softmax_forward_cuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(focal_loss_softmax_backward)
    .attr("N")
    .attr("weight_pos")
    .attr("gamma")
    .attr("alpha")
    .attr("num_classes")
    .input(3)
    .output(2)
    .apply(focal_loss_softmax_backward_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(focal_loss_softmax_backward_cuda)
#endif
    .done();
