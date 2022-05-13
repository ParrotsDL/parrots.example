# import atexit
# import time
# from threading import Timer, Thread

# import torch

# l = []

# def f():
#     Timer(2, f).start()
#     print("time:", time.time())

# def f2():
#     Timer(2 - ((time.time() - starttime) % 2), f2).start()
#     # print("time:", time.time())
#     l.append(time.time())

# def g():
#     while True:
#         # print("time:", time.time())
#         # print("alloc:", torch.cuda.memory_allocated()/1024/1024)
#         l.append(torch.cuda.memory_allocated()/1024/1024)
#         time.sleep(2 - ((time.time() - starttime) % 2))
#     # while 1:
#     #     print("time:", time.time())
#     #     time.sleep(2)

# def h(*args, **kwargs):
#     print(len(l))


# atexit.register(h)
# # t = Timer(2, f2)
# t = Thread(target=g)
# t.daemon = True
# starttime = time.time()
# t.start()

from collections import defaultdict
import json
import os

proc_id = os.environ.get('SLURM_PROCID')
profile_name = 'profile_' + proc_id + '.txt'


class Node:
    def __init__(self, id, name):
        self.id = id
        self.children = []
        self.name = name
        self.ops = []
        self.time = 0

    def add_children(self, node):
        self.children.append(node)

    def is_leaf(self):
        return len(self.children) == 0

    def add_time(self, t):
        self.time += t

    def get_time(self):
        if self.time > 0:
            return self.time
        for node in self.children:
            self.time += node.get_time()
        return self.time



class Op:
    def __init__(self, name, start, end):
        pass


class ModuleProfiler:
    _module = None
    def __init__(self, model, begin_it=200, end_it=210):
        assert ModuleProfiler._module is None, "can only Instance ModuleProfiler once"
        ModuleProfiler._module = model
        import torch
        assert isinstance(model, torch.nn.Module)
        assert isinstance(begin_it, int)
        assert isinstance(end_it, int)
        assert begin_it < end_it
        self.model = model
        self.begin_it = begin_it
        self.end_it = end_it
        self.it = 0
        self.id2node = {}
        self.op_times = defaultdict(float)
        model.register_forward_pre_hook(self.main_module_forward_pre_hook)
        self.root = self.dfs_for_module(model, 'main')
        self.id2scope = {}


    def main_module_forward_pre_hook(self, *args):
        import parrots
        if self.it == self.begin_it:
            parrots.runtime.profile(enable=True, file=profile_name, use_scope=True)
        elif self.it == self.end_it:
            parrots.binding.wait_all()
            parrots.runtime.profile(enable=False)
            parrots.log_utils.flush_profile_record()
            self.analyze_profile()
            exit()
        self.it += 1

    def make_leaf_module_forward_pre_hook(self, m_id):
        def hook(*args):
            if self.it >= self.begin_it:
                import parrots
                scope = parrots.scope.named_scope(str(m_id))
                self.id2scope[m_id] = scope
        return hook

    def make_leaf_module_forward_hook(self, m_id):
        def hook(*args):
            if self.it >= self.begin_it:
                self.id2scope[m_id].__exit__(None, None, None)
        return hook

    def dfs_for_module(self, model, name):
        m_id = id(model)
        node = Node(m_id, name)
        self.id2node[m_id] = node
        for n, m in model.named_children():
            node.add_children(self.dfs_for_module(m, n))
        if node.is_leaf():
            model.register_forward_pre_hook(self.make_leaf_module_forward_pre_hook(m_id))
            model.register_forward_hook(self.make_leaf_module_forward_hook(m_id))
        return node

    def analyze_profile(self):
        with open(profile_name, 'r') as f:
            first_line = f.readline()
            for _ in range(int(first_line)):
                f.readline()
            for l in f:
                record = json.loads(l)
                begin = record['begin']
                end = record['end']
                duration = end - begin
                scope = record['args']['scope']
                name = record['name']
                for s in scope.split('/')[::-1]:
                    try:
                        m_id = int(s)
                        if m_id in self.id2node and self.id2node[m_id].is_leaf():
                            self.id2node[m_id].add_time(duration)
                            self.op_times[name] += duration
                            break
                    except:
                        pass
        for node in self.id2node.values():
            node.get_time()


def start_profile():
    pass

def stop_profile():
    pass