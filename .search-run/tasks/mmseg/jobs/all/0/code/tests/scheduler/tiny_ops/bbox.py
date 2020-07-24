import torch
import time

rois = torch.rand((2, 182400, 4), device='cuda')
cls_pred = torch.rand((2, 182400, 1), device='cuda')
b_ix = 0
roi_min_size = 0
image_info = [[800, 1067, 1.6666666666666667, 480, 640, True], [800, 1202, 1.8779342723004695, 426, 640, False]]

def filter_by_size(boxes, min_size, start_index=0):
    s = start_index
    w = boxes[:, s + 2] - boxes[:, s + 0] + 1
    h = boxes[:, s + 3] - boxes[:, s + 1] + 1
    filter_inds = (w > min_size) & (h > min_size)
    return boxes[filter_inds], filter_inds

def clip_bbox(bbox, img_size):
    h, w = img_size[:2]
    bbox[:, 0] = torch.clamp(bbox[:, 0], min=0, max=w - 1)
    bbox[:, 1] = torch.clamp(bbox[:, 1], min=0, max=h - 1)
    bbox[:, 2] = torch.clamp(bbox[:, 2], min=0, max=w - 1)
    bbox[:, 3] = torch.clamp(bbox[:, 3], min=0, max=h - 1)
    return bbox

def func():
    image_rois = rois[b_ix]
    mage_rois = clip_bbox(image_rois, image_info[b_ix])
    image_rois, filter_inds = filter_by_size(image_rois, roi_min_size)
    image_cls_pred = cls_pred[b_ix][filter_inds]

# warm up
for i in range(10):
    func()
torch.cuda.synchronize()
start = time.time()

for i in range(2000):
    func()
torch.cuda.synchronize()
end = time.time()
print(end-start)
