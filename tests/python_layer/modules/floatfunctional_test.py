import logging
import torch
from torch.nn.quantized.modules import FloatFunctional


logging.basicConfig(level=logging.INFO)

f = FloatFunctional()
try:
    f()
except:
    logging.info("[PASSED] Forward Error Test")
else:
    logging.error("[FAILED] Forward Error Test")

def test(arch='cpu'):
    a = torch.tensor([3.0], device=arch)
    b = torch.tensor([4.0], device=arch)
    o1 = f.add(a, b)
    o2 = f.mul(a, b)
    o3 = f.add_relu(a, b)
    o4 = f.cat([a, b])

    t1 = torch.add(a, b)
    t2 = torch.mul(a, b)
    t3 = torch.relu(torch.add(a, b))
    t4 = torch.cat([a, b])

    try:
        r = torch.allclose(t1, o1) and torch.allclose(t2, o2) and torch.allclose(t3, o3) and torch.allclose(t4, o4)
    except:
        logging.error("[FAILED] Compute Result Test: {}".format(arch))
    else:
        if r:
            logging.info("[PASSED] Compute Result Test: {}".format(arch))
        else:
            logging.error("[FAILED] Compute Result Test: {}".format(arch))


if __name__ == '__main__':
    test(arch='cpu')
    test(arch='cuda')
    logging.info("[PASSED] FloatFunctional Test")
