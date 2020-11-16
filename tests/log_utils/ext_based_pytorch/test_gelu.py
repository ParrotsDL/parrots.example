from parrots.utils import tester
import torch


class TestExtension(object):

    def _test_pytorch_ext_gelu(self, device='cpu'):
        import gelu_parrots
        x = torch.randn((1, 2, 3), device=device)
        x.requires_grad_()
        y = torch.empty((1, 2, 3), device=device)
        gelu_parrots.gelu_forward(x, y)
        assert y.shape == x.shape

        loss = torch.ones_like(y, device=device)
        xg = torch.empty((1, 2, 3), device=device)
        gelu_parrots.gelu_backward(loss, x, xg)
        assert x.shape == xg.shape

    @tester.skip_no_aten
    def test_pytorch_ext_gelu_cpu(self):
        self._test_pytorch_ext_gelu('cpu')

    @tester.skip_no_aten
    @tester.skip_no_cuda
    def test_pytorch_ext_gelu_cuda(self):
        self._test_pytorch_ext_gelu('cuda')
