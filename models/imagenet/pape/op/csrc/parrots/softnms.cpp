#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <parrots/foundation/exceptions.hpp>


using namespace parrots;  // NOLINT

void softnmsApplyHost(HostContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    float sigma, n_thresh, overlap_thresh;
    int64_t method;
    SSAttrs(attr).get<float>("sigma", sigma)
                 .get<float>("n_thresh", n_thresh)
                 .get<float>("overlap_thresh", overlap_thresh)
                 .get<int64_t>("method", method)
                 .done();

    auto& boxes = outs[0];
    auto& areas = outs[1];
    auto& order = outs[2];
    auto& num_out = outs[3];

    size_t boxes_num = boxes.shape().dim(0);
    size_t boxes_dim = boxes.shape().dim(1);
    // reserve output space
    // if (!keep_out) {
    //     keep_out = ctx.createDArrayLite(DArraySpec::array(Prim::Int64, boxes_num));
    // }
    if (!num_out) {
        num_out = ctx.createDArrayLite(DArraySpec::scalar(Prim::Int64));
    }

    auto boxes_ptr = boxes.ptr<float>();
    auto order_ptr = order.ptr<int64_t>();
    auto areas_ptr = areas.ptr<float>();

    // nominal indices
    unsigned int i, ti, pos, maxpos;
    // temp variables for box i's (the box currently under consideration)
    float ix1, iy1, ix2, iy2, iarea, iscore;
    // variables for computing overlap with box j (lower scoring box)
    float xx1, yy1, xx2, yy2;
    float w, h;
    float inter, ovr, maxscore, weight;

    for (i=0; i < boxes_num; ++i) {
        maxscore = boxes_ptr[i * boxes_dim + 4];
        maxpos = i;
        
        ix1 = boxes_ptr[i * boxes_dim];
        iy1 = boxes_ptr[i * boxes_dim + 1];
        ix2 = boxes_ptr[i * boxes_dim + 2];
        iy2 = boxes_ptr[i * boxes_dim + 3];
        iscore = boxes_ptr[i * boxes_dim + 4];
        iarea = areas_ptr[i];
        ti = order_ptr[i];

        pos = i + 1;
        while (pos < boxes_num){
            if (maxscore < boxes_ptr[pos * boxes_dim + 4]){
                maxscore = boxes_ptr[pos * boxes_dim + 4];
                maxpos = pos;
            }
            pos ++;
        }
        if (maxpos > i){
            boxes_ptr[i * boxes_dim] = boxes_ptr[maxpos * boxes_dim];
            boxes_ptr[i * boxes_dim + 1] = boxes_ptr[maxpos * boxes_dim + 1];
            boxes_ptr[i * boxes_dim + 2] = boxes_ptr[maxpos * boxes_dim + 2];
            boxes_ptr[i * boxes_dim + 3] = boxes_ptr[maxpos * boxes_dim + 3];
            boxes_ptr[i * boxes_dim + 4] = boxes_ptr[maxpos * boxes_dim + 4];
            order_ptr[i] = order_ptr[maxpos];
            areas_ptr[i] = areas_ptr[maxpos];

            boxes_ptr[maxpos * boxes_dim] = ix1;
            boxes_ptr[maxpos * boxes_dim + 1] = ix2;
            boxes_ptr[maxpos * boxes_dim + 2] = iy1;
            boxes_ptr[maxpos * boxes_dim + 3] = iy2;
            boxes_ptr[maxpos * boxes_dim + 4] = iscore;
            order_ptr[maxpos] = ti;
            areas_ptr[maxpos] = iarea;

            ix1 = boxes_ptr[i * boxes_dim];
            iy1 = boxes_ptr[i * boxes_dim + 1];
            ix2 = boxes_ptr[i * boxes_dim + 2];
            iy2 = boxes_ptr[i * boxes_dim + 3];
            iscore = boxes_ptr[i * boxes_dim + 4];
            iarea = areas_ptr[i];
        }

        for (pos = i + 1; pos < boxes_num; ++pos) {
            xx1 = fmaxf(ix1, boxes_ptr[pos * boxes_dim]);
            yy1 = fmaxf(iy1, boxes_ptr[pos * boxes_dim + 1]);
            xx2 = fminf(ix2, boxes_ptr[pos * boxes_dim + 2]);
            yy2 = fminf(iy2, boxes_ptr[pos * boxes_dim + 3]);
            w = fmaxf(0.0, xx2 - xx1 + 1);
            h = fmaxf(0.0, yy2 - yy1 + 1);
            inter = w * h;
            ovr = inter / (iarea + areas_ptr[pos] - inter);
            weight = 1.0;
            if (method == 1){  //linear
                if (ovr > n_thresh){
                    weight = 1.0 - ovr;
                }
            }
            else if (method == 2){  // gaussian
                weight = exp(-(ovr * ovr)/sigma);
            }
            else if (ovr >= n_thresh){  // naive_nms
                weight = 0;
            }
            boxes_ptr[pos * boxes_dim + 4] *= weight;
            if (boxes_ptr[pos * boxes_dim + 4] < overlap_thresh){
                boxes_ptr[pos * boxes_dim] = boxes_ptr[(boxes_num - 1) * boxes_dim];
                boxes_ptr[pos * boxes_dim + 1] = boxes_ptr[(boxes_num - 1) * boxes_dim + 1];
                boxes_ptr[pos * boxes_dim + 2] = boxes_ptr[(boxes_num - 1) * boxes_dim + 2];
                boxes_ptr[pos * boxes_dim + 3] = boxes_ptr[(boxes_num - 1) * boxes_dim + 3];
                boxes_ptr[pos * boxes_dim + 4] = boxes_ptr[(boxes_num - 1) * boxes_dim + 4];
                order_ptr[pos] = order_ptr[boxes_num - 1];
                boxes_num --;
                pos --;
            }
        }
    }
    setScalar(num_out, int64_t{boxes_num});
}


#ifdef PARROTS_USE_CUDA
void softnmsApplyCuda(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    PARROTS_NOT_IMPL;
}
#endif  // PARROTS_USE_CUDA


PARROTS_EXTENSION_REGISTER(softnms)
    .attr("sigma")
    .attr("n_thresh")
    .attr("overlap_thresh")
    .attr("method")
    .input(0)
    .output(4)
    .apply(softnmsApplyHost)
#ifdef PARROTS_USE_CUDA
    .apply(softnmsApplyCuda)
#endif
    .done();
