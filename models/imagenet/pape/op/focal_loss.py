from torch.autograd import Function
from ..utils import ext_loader
ext_module = ext_loader.load_ext('op_ext',
                                 ['focal_loss_sigmoid_forward',
                                  'focal_loss_sigmoid_backward',
                                  'focal_loss_softmax_forward',
                                  'focal_loss_softmax_backward'])


class SigmoidFocalLossFunction(Function):
    """
    SigmoidFocalLoss 的一种实现方法，在论文 Focal Loss for Dense Object Detection
    中首次提出。

    参数:
        - gamma(int): 缩放因子，用于平衡易分样本和难分样本的超参数
        - alpha(float, double): 用于平衡正样本和负样本的超参数
        - num_classes(int): 类别数

    输入:
        - preds(Tensor): 预测值，shape [Batch * h * w * num_anchors, num_classes]
        - targets(Tensor): 真实值，shape [Batch * h * w * num_anchors]
        - weight_pos(list): 位置权重
    """
    def __init__(self, gamma, alpha, num_classes):

        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

        self.weight_pos = None
        self.preds = None
        self.targets = None

    def forward(self, preds, targets, weight_pos):
        # preds shape: [Batch * h * w * num_anchors, num_classes]
        # targets shape: [Batch * h * w * num_anchors]
        preds_size = preds.size()
        targets_size = targets.size()

        assert preds_size[0] == targets_size[0]
        assert preds_size[1] == self.num_classes

        losses = preds.new(preds_size[0], preds_size[1]).zero_()
        weight_pos = float(weight_pos[0])
        N = preds_size[0] * preds_size[1]

        assert losses.is_contiguous()
        assert preds.is_contiguous()
        assert targets.is_contiguous()
        assert preds.is_cuda and targets.is_cuda

        ext_module.focal_loss_sigmoid_forward(
            preds,
            targets,
            losses,
            N=N,
            weight_pos=weight_pos,
            gamma=self.gamma,
            alpha=self.alpha,
            num_classes=self.num_classes)
        self.preds = preds
        self.targets = targets
        self.weight_pos = weight_pos
        return losses.sum()

    def backward(self, grad_output):
        # grad_output: 1.0 / num_of_gpus
        preds_size = self.preds.size()
        grad_input = self.preds.new(preds_size[0], preds_size[1]).zero_()
        N = preds_size[0] * preds_size[1]

        assert self.preds.is_contiguous()
        assert self.targets.is_contiguous()
        assert grad_input.is_contiguous()
        assert self.preds.is_cuda and self.targets.is_cuda and grad_input.is_cuda

        ext_module.focal_loss_sigmoid_backward(
            self.preds,
            self.targets,
            grad_input,
            N=N,
            weight_pos=self.weight_pos,
            gamma=self.gamma,
            alpha=self.alpha,
            num_classes=self.num_classes)

        grad_input = grad_input * grad_output
        return grad_input, None, None


class SoftmaxFocalLossFunction(Function):
    """
    SoftmaxFocalLoss 的实现方法。

    参数:
        - gamma(int): 缩放因子，用于平衡易分样本和难分样本的超参数
        - alpha(float, double): 用于平衡正样本和负样本的超参数
        - num_classes(int): 类别数

    输入:
        - preds(Tensor): 预测值，shape [Batch * h * w * num_anchors, num_classes]
        - targets(Tensor): 真实值，shape [Batch * h * w * num_anchors]
        - weight_pos(list): 位置权重
    """
    def __init__(self, gamma, alpha, num_classes):
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

        self.weight_pos = None
        self.preds = None
        self.targets = None

    def forward(self, preds, targets, weight_pos):
        # preds shape: [Batch * h * w * num_anchors, num_classes]
        # targets shape: [Batch * h * w * num_anchors]
        preds_size = preds.size()
        targets_size = targets.size()

        assert preds_size[0] == targets_size[0]
        assert preds_size[1] == self.num_classes

        losses = preds.new(preds_size[0]).zero_()
        priors = preds.new(preds_size[0], preds_size[1]).zero_()

        weight_pos = float(weight_pos[0])
        N = preds_size[0] * preds_size[1]

        assert losses.is_contiguous()
        assert preds.is_contiguous()
        assert targets.is_contiguous()
        assert priors.is_contiguous()
        assert preds.is_cuda and targets.is_cuda

        ext_module.focal_loss_softmax_forward(
            preds,
            targets,
            losses,
            priors,
            N=N,
            weight_pos=weight_pos,
            gamma=self.gamma,
            alpha=self.alpha,
            num_classes=self.num_classes)

        self.preds = preds
        self.targets = targets
        self.weight_pos = weight_pos
        self.priors = priors
        return losses.sum()

    def backward(self, grad_output):
        # grad_output: 1.0 / num_of_gpus
        preds_size = self.preds.size()
        grad_input = self.preds.new(preds_size[0], preds_size[1]).zero_()
        buff = self.preds.new(preds_size[0]).zero_()
        N = preds_size[0] * preds_size[1]

        assert self.preds.is_contiguous()
        assert self.targets.is_contiguous()
        assert grad_input.is_contiguous()
        assert buff.is_contiguous()
        assert self.preds.is_cuda and self.targets.is_cuda and grad_input.is_cuda and buff.is_cuda

        ext_module.focal_loss_softmax_backward(
            self.preds,
            self.targets,
            self.priors,
            grad_input,
            buff,
            N=N,
            weight_pos=self.weight_pos,
            gamma=self.gamma,
            alpha=self.alpha,
            num_classes=self.num_classes)
        grad_input = grad_input * grad_output
        return grad_input, None, None
