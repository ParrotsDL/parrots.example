#include "pytorch_cpp_helper.hpp"
#include <math.h>
#include <THC/THC.h>
#include <assert.h>
#include <stdio.h>
using namespace at;
int SoftmaxFocalLossForwardLaucher(
    const int N, Tensor logits,
    Tensor targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, Tensor losses,
    Tensor priors);

int SoftmaxFocalLossBackwardLaucher(
    const int N, Tensor logits, Tensor targets,
    Tensor dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes, 
    Tensor priors, Tensor buff);

int SigmoidFocalLossForwardLaucher(
    const int N, Tensor logits,
    Tensor targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, Tensor losses);

int SigmoidFocalLossBackwardLaucher(
    const int N, Tensor logits, Tensor targets,
    Tensor dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int focal_loss_sigmoid_forward(
                           Tensor logits,
                           Tensor targets,
                           Tensor losses,
                           int N,
                           float weight_pos,
                           float gamma, 
                           float alpha,
                           int num_classes){
    if(!logits.type().is_cuda()){
        AT_ERROR("Not support cpu focal_loss sigmoid!");
    }
    else{
        CHECK_INPUT(logits);
        CHECK_INPUT(targets);
        CHECK_INPUT(losses);

        SigmoidFocalLossForwardLaucher(
            N, logits, targets, weight_pos, 
            gamma, alpha, num_classes, losses);

        return 1;
   }
}

int focal_loss_sigmoid_backward(
                           Tensor logits,
                           Tensor targets,
                           Tensor dX_data,
                           int N,
                           float weight_pos,
                           float gamma,
                           float alpha,
                           int num_classes){
    if(!logits.type().is_cuda()){
        AT_ERROR("Not support cpu focal_loss sigmoid!"); 
    }
    else{
        CHECK_INPUT(logits);
        CHECK_INPUT(targets);
        CHECK_INPUT(dX_data);

        SigmoidFocalLossBackwardLaucher(
            N, logits, targets, dX_data,
            weight_pos, gamma, alpha, num_classes);

        return 1;
    }
}

int focal_loss_softmax_forward(
                           Tensor logits,
                           Tensor targets,
                           Tensor losses,
                           Tensor priors,
                           int N,
                           float weight_pos,
                           float gamma, 
                           float alpha,
                           int num_classes){
    if(!logits.type().is_cuda()){
        AT_ERROR("Not support cpu focal_loss softmax!");
    }
    else{ 
        CHECK_INPUT(logits);
        CHECK_INPUT(targets);
        CHECK_INPUT(losses);
        CHECK_INPUT(priors);

        SoftmaxFocalLossForwardLaucher(
            N, logits, targets, weight_pos, 
            gamma, alpha, num_classes, losses, priors);

        return 1;
    }
}

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
                           int num_classes){
    if(!logits.type().is_cuda()){
        AT_ERROR("Not support cpu focal_loss softmax!");
    }
    else{
        CHECK_INPUT(logits);
        CHECK_INPUT(targets);
        CHECK_INPUT(dX_data);
        CHECK_INPUT(priors);
        CHECK_INPUT(buff);
    
        SoftmaxFocalLossBackwardLaucher(
            N, logits, targets, dX_data,
            weight_pos, gamma, alpha, num_classes, priors, buff);

        return 1;
    }
}
