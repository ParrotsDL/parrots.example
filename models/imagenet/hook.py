
import os
import torch
import pickle
import numpy as np
import numpy
from termcolor import colored
import copy

np.set_printoptions(suppress=True, precision=8)

### 更换版本记得改这里 ！！！！！！
CAMB_TORCH_VERSION = "1.6.0a0"
CAMB_PARROTS_VERSION = "parrots"
CPU_PYTORCH_VERSION = "1.3.1+cpu"

code_yellow = lambda x: colored(x, 'yellow')
code_red = lambda x: colored(x, 'red')
code_green = lambda x: colored(x, 'green')
code_blue = lambda x: colored(x, 'blue')
code_grey = lambda x: colored(x, 'grey')
code_magenta = lambda x: colored(x, 'magenta')
code_cyan = lambda x: colored(x, 'cyan')
code_grey_on_red = lambda x : colored(x, 'grey', 'on_red')
code_grey_on_green = lambda x : colored(x, 'grey', 'on_green')


'''
# 介绍
改类是进行精度比较的hook工具
首先在pytorch环境下运行
然后在parrots环境下运行
运行结束后会比较两次的结果

# 使用方法
hc = hookCompare()

m = AlexNet()
input = torch.randn(2, 3, 224, 224, requires_grad=True)

# 进行hook注册访问每一层的forward和backward输入输出
for name, mm in m.named_modules():
    print(code_blue(name), "---", code_blue(mm))
    mm.register_forward_hook(hc.hook(name, mm))
    mm.register_backward_hook(hc.hook(name, mm, tag='backward'))

input, m = hc.save_and_load(input, m)
m = m.train()
input, m = hc.to_cuda(input, m)
out = m(input)
out.backward(torch.ones_like(out))

hc.save_and_compare_hook()
'''

class hookCompare():
    def __init__(self, display_num=5):
        self.display_num = display_num
        self.l2_dist_threshold = 1
        self.torch_version = torch.__version__
        self.parrots_dict = {}
        self.name_list_forward = []
        self.name_list_backward = []
        self.pkl_save_path = "data/camb_torch_hook.pkl"
        self.torch_checkpoint_path = "data/checkpoint.pth"
        self.torch_input_path = "data/input.pth"
        self.torch_target_path = "data/target.pth"
        self.compared_updated_model_path = "data/compared_torch_model.pth"
        self.parrots_updated_model_path = "data/parrots_torch_model.pth"

        self._make_dirs(self.pkl_save_path)
        self._make_dirs(self.torch_checkpoint_path)
        self._make_dirs(self.torch_input_path)
        self._make_dirs(self.torch_target_path)
        self._make_dirs(self.compared_updated_model_path)
        self._make_dirs(self.parrots_updated_model_path)
        self._check()

        print("pytorch version:", code_green(torch.__version__))

    def _check(self):
        if self.torch_version == CAMB_PARROTS_VERSION:
            assert os.path.exists(self.pkl_save_path)
            assert os.path.exists(self.torch_checkpoint_path)
            assert os.path.exists(self.torch_input_path)

    def _make_dirs(self, path):
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

    def _insert(self, name, mm, tag, input, output):
        if "Sequential" in f"{mm}":
            return
        if name not in self.parrots_dict.keys():
            self.parrots_dict[name] = {}
        if tag == "forward":
            self.name_list_forward.append(name)
            self.parrots_dict[name]['forward'] = {"op": f"{mm}", "input": input, "output": output}
        else:
            self.name_list_backward.append(name)
            self.parrots_dict[name]['backward'] = {"op": f"{mm}", "input": output, "output": input}

    def _pkl_save(self):
        with open(self.pkl_save_path, "wb") as file:
            pickle.dump(self.parrots_dict, file)
            print("save file at {}".format(self.pkl_save_path))

    def _pkl_read(self):
        with open(self.pkl_save_path, "rb") as file:
            return pickle.load(file)

    def L2_dist(self, a, b):
        if a is None:
            return 0
        return np.linalg.norm(a - b)

    def _display(self, data):
        if data is None:
            return -1, -1, -1, -1, [-1], [-1]
        return np.abs(np.sum(data)), np.abs(np.mean(data)), np.max(data), np.min(data), data.shape, data.reshape(-1)[0:self.display_num]

    def _hook_diff(self, a, b):
        sum0, mean0, shape0, max0, min0, patch0 = self._display(a)
        sum1, mean1, shape1, max1, min1, patch1 = self._display(b)

        try:
            import prettytable as pt
            tb = pt.PrettyTable()
            tb.field_names = ['', 'compared', 'parrots', 'diff']
            tb.add_row(['sum:', sum0, sum1, sum0 - sum1])
            tb.add_row(['mean:', mean0, mean1, mean0 - mean1])
            tb.add_row(['max:', max0, max1, '-'])
            tb.add_row(['min:', min0, min1, '-'])
            tb.add_row(['shape:', shape0, shape1, '-'])
            print(tb)
            print(f"patch0: {patch0}")
            print(f"patch1: {patch1}")
            l2_dist = self.L2_dist(a, b)
            print(f"l2 dist: {l2_dist}")
            color_func = code_grey_on_green if l2_dist < self.l2_dist_threshold else code_grey_on_red 
            print(f"Diff:\n patch: {color_func(patch0 - patch1)}")
            print("\n")
        except ImportError:
            print(f"sum:\t{sum0}\t{sum1}")
            print(f"mean:\t{mean0}\t{mean1}")
            l2_dist = self.L2_dist(a, b)
            print(f"l2 dist: {l2_dist}")
            color_func = code_grey_on_green if l2_dist < self.l2_dist_threshold else code_grey_on_red 
            print("Diff:\n", color_func(f"sum: {sum0 - sum1}, \n mean: {mean0 - mean1}, \n patch: {patch0 - patch1}"))
            print("\n")

    # only use in compared torch
    def _pkl_compare(self):
        print("================== end ====================\n\n\n\n\n")
        print("================== start compare hook ===================\n")
        for name in self.name_list_forward:
            if not name in self.compare_dict.keys() or not name in self.parrots_dict.keys() or name == "":
                print(f"The key {code_grey(name)} does not match")
                continue

            compare_input = self.compare_dict[name]['forward']['input']
            parrots_input = self.parrots_dict[name]['forward']['input']
            compare_output = self.compare_dict[name]['forward']['output']
            parrots_output = self.parrots_dict[name]['forward']['output']

            print(f"Layer {code_yellow(name)} {code_red('forward')} {code_green(self.compare_dict[name]['forward']['op'])}")
            print("Input:")
            for x, y in zip(compare_input, parrots_input):
                self._hook_diff(x, y)
            print("Output:")
            for x, y in zip(compare_output, parrots_output):
                self._hook_diff(x, y)
            print()

        for name in self.name_list_backward:
            if not name in self.compare_dict.keys() or not name in self.parrots_dict.keys() or name == "":
                print(f"The key {code_grey(name)} does not match")
                continue
            print(f"Layer {code_yellow(name)} {code_red('backward')} {code_green(self.compare_dict[name]['backward']['op'])}")

            compare_input = self.compare_dict[name]['forward']['input']
            parrots_input = self.parrots_dict[name]['forward']['input']
            compare_output = self.compare_dict[name]['backward']['output']
            parrots_output = self.parrots_dict[name]['backward']['output']
            
            # linear shape is dismatch between two platforms, thus skip to compute _hook_diff
            if "Linear" not in self.compare_dict[name]['backward']['op']:
                # self._hook_diff(compare_output, parrots_output)
                print("Input:")
                for x, y in zip(compare_input, parrots_input):
                    self._hook_diff(x, y)
                print("Output:")
                for x, y in zip(compare_output, parrots_output):
                    self._hook_diff(x, y)
            print()
        print("================== end compare hook ===================\n\n\n")

    # 保存并且打印比较结果
    def save_and_compare_hook(self):
        if self.torch_version != CAMB_PARROTS_VERSION:
            self._pkl_save()
        else:
            self.compare_dict = self._pkl_read()
            self._pkl_compare()

    def get_data_numpy(self, data):
        if isinstance(data, (torch.Tensor)):
            data = data.clone().detach().cpu().numpy()
            return data
        elif isinstance(data, dict):
            return {k: self.get_data_numpy(v) for k, v in data.items()}
        elif isinstance(data, (tuple, list)):
            return data.__class__(self.get_data_numpy(v) for v in data)
        else:
            return data

    def hook(self, name, mm, tag='forward'):
        hook_func = self.get_data_numpy
        def inner_hook(m, input, output):
            self._insert(name, mm, tag, hook_func(input), hook_func(output))
        return inner_hook

    def save_and_load(self, input, target, model):
        if self.torch_version != CAMB_PARROTS_VERSION:
            torch.save(model.state_dict(), self.torch_checkpoint_path)
            torch.save(input, self.torch_input_path)
            torch.save(target, self.torch_target_path)
        else:
            input = torch.load(self.torch_input_path)
            target = torch.load(self.torch_target_path)
            model.load_state_dict(torch.load(self.torch_checkpoint_path))

        return input, target, model

    def save_updated_model(self, model):
        if self.torch_version != CAMB_PARROTS_VERSION:
            torch.save(model.state_dict(), self.compared_updated_model_path)
        else:
            torch.cuda.synchronize()
            torch.save(model.state_dict(), self.parrots_updated_model_path)
            torch.cuda.synchronize()

    def compare_updated_model(self, model):
        if self.torch_version == CAMB_PARROTS_VERSION:
            assert os.path.isfile(self.compared_updated_model_path)
            assert os.path.isfile(self.parrots_updated_model_path)

            parrots_model = copy.deepcopy(model)
            compared_model = copy.deepcopy(model)

            compared_model.load_state_dict(torch.load(self.compared_updated_model_path))
            parrots_model.load_state_dict(torch.load(self.parrots_updated_model_path))

            compare_state_dict = compared_model.state_dict()
            parrots_state_dict = parrots_model.state_dict()

            print("================== start compare updated grad ===================\n")
            for p in compared_model.state_dict():
                compared_param = compare_state_dict[p].detach().cpu().numpy()
                parrots_param = parrots_state_dict[p].detach().cpu().numpy()
                print(code_yellow(p), code_yellow(compared_param.shape))
                
                self._hook_diff(compared_param, parrots_param)
            print("================== end compare updated grad ===================\n")


    # suggest rewrite outswide class
    def to_cuda(self, input, model, qb=31):
        if self.torch_version == CAMB_PARROTS_VERSION:
            # torch.cuda.set_device(4)
            model = model.to_memory_format(torch.channels_last)
            model = model.cuda()
            input = input.cuda()
            if input.ndims == 4:
                input = input.contiguous(torch.channels_last)
        elif self.torch_version == CPU_PYTORCH_VERSION:
            pass
        else:
            torch.set_printoptions(10)
            import torch_mlu.core.mlu_model as ct
            # ct.set_device(4)
            ct.set_cnml_enabled(False)
            ct.set_quantized_bitwidth(qb)
            model = model.to(ct.mlu_device())
            input = input.to(ct.mlu_device())
        
        return input, model
