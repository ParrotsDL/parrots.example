#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

using namespace parrots;  // NOLINT

void nmsApplyHost(HostContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    float nms_overlap_thresh;
    int offset;
    SSAttrs(attr).get<float>("nms_overlap_thresh", nms_overlap_thresh)
                 .get<int>("offset", offset).done();

    const auto& boxes = ins[0];
    const auto& order = ins[1];
    const auto& areas = ins[2];

    auto& keep_out = outs[0];
    auto& num_out = outs[1];

    // TODO(lizhouyang): check if darrays are contiguous.

    size_t boxes_num = boxes.shape().dim(0);
    size_t boxes_dim = boxes.shape().dim(1);
    // reserve output space
    if (!keep_out) {
        keep_out = ctx.createDArrayLite(DArraySpec::array(Prim::Int64, boxes_num));
    }
    if (!num_out) {
        num_out = ctx.createDArrayLite(DArraySpec::scalar(Prim::Int64));
    }

    auto keep_out_ptr = keep_out.ptr<int64_t>();
    auto boxes_ptr = boxes.ptr<float>();
    auto order_ptr = order.ptr<int64_t>();
    auto areas_ptr = areas.ptr<float>();

    auto suppressed = ctx.createDArrayLite(DArraySpec::array(Prim::Bool, boxes_num));
    suppressed.setZeros(syncStream());
    auto suppressed_ptr = suppressed.ptr<bool>();

    // nominal indices
    int i, j;
    // sorted indices
    size_t _i, _j;
    // temp variables for box i's (the box currently under consideration)
    float ix1, iy1, ix2, iy2, iarea;
    // variables for computing overlap with box j (lower scoring box)
    float xx1, yy1, xx2, yy2;
    float w, h;
    float inter, ovr;

    size_t num_to_keep = 0;
    for (_i=0; _i < boxes_num; ++_i) {
        i = order_ptr[_i];
        if (suppressed_ptr[i] == 1) {
            continue;
        }
        keep_out_ptr[num_to_keep++] = i;
        ix1 = boxes_ptr[i * boxes_dim];
        iy1 = boxes_ptr[i * boxes_dim + 1];
        ix2 = boxes_ptr[i * boxes_dim + 2];
        iy2 = boxes_ptr[i * boxes_dim + 3];
        iarea = areas_ptr[i];
        for (_j = _i + 1; _j < boxes_num; ++_j) {
            j = order_ptr[_j];
            if (suppressed_ptr[j] == 1) {
                continue;
            }
            xx1 = fmaxf(ix1, boxes_ptr[j * boxes_dim]);
            yy1 = fmaxf(iy1, boxes_ptr[j * boxes_dim + 1]);
            xx2 = fminf(ix2, boxes_ptr[j * boxes_dim + 2]);
            yy2 = fminf(iy2, boxes_ptr[j * boxes_dim + 3]);
            w = fmaxf(0.0, xx2 - xx1 + offset);
            h = fmaxf(0.0, yy2 - yy1 + offset);
            inter = w * h;
            ovr = inter / (iarea + areas_ptr[j] - inter);
            if (ovr >= nms_overlap_thresh) {
                suppressed_ptr[j] = 1;
            }
        }
    }
    setScalar(num_out, int64_t{num_to_keep});
}


#ifdef PARROTS_USE_CUDA
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

void _nms(int boxes_num, const DArrayLite boxes_dev,
          DArrayLite mask_dev, float nms_overlap_thresh, int offset);

void nmsApplyCuda(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs) {
    float nms_overlap_thresh;
    int offset;
    SSAttrs(attr).get<float>("nms_overlap_thresh", nms_overlap_thresh)
                 .get<int>("offset", offset).done();
    
    const auto& boxes = ins[0];
    // const auto& order = ins[1];
    // const auto& areas = ins[2];

    auto& keep_out = outs[0];
    auto& num_out = outs[1];

    // TODO(lizhouyang): check if darrays are contiguous.

    size_t boxes_num = boxes.shape().dim(0);

    // reserve output space
    if (!keep_out) {
        keep_out = ctx.createDArrayLite(DArraySpec::array(Prim::Int64, boxes_num), getHostProxy());
    }
    if (!num_out) {
        num_out = ctx.createDArrayLite(DArraySpec::scalar(Prim::Int64), getHostProxy());
    }

    auto keep_out_ptr = keep_out.ptr<int64_t>();

    const size_t col_blocks = DIVUP(boxes_num, threadsPerBlock);
    auto cuda_mask = ctx.createDArrayLite(DArraySpec::array(Prim::Int64, DArrayShape(boxes_num, col_blocks)));

    _nms(boxes_num, boxes, cuda_mask, nms_overlap_thresh, offset);

    auto host_mask = ctx.createDArrayLite(cuda_mask, getHostProxy());
    auto host_mask_ptr = host_mask.ptr<int64_t>();
    auto remv = ctx.createDArrayLite(DArraySpec::array(Prim::Int64, col_blocks), getHostProxy());
    remv.setZeros(syncStream());
    auto remv_ptr = remv.ptr<int64_t>();
    size_t num_to_keep = 0;

    int i, j;
    for (i = 0; i < boxes_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(remv_ptr[nblock] & (1ULL << inblock))) {
            keep_out_ptr[num_to_keep++] = i;
            int64_t *p = host_mask_ptr + i * col_blocks;
            for (j = nblock; j < col_blocks; j++) {
                remv_ptr[j] |= p[j];
            }
        }
    }
    setScalar(num_out, int64_t{num_to_keep});
}
#endif  // PARROTS_USE_CUDA


PARROTS_EXTENSION_REGISTER(nms)
    .attr("nms_overlap_thresh")
    .attr("offset")
    .input(3)
    .output(2)
    .apply(nmsApplyHost)
#ifdef PARROTS_USE_CUDA
    .apply(nmsApplyCuda)
#endif
    .done();
