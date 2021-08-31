class PapeModel(object):
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        if name in ['module']:
            return self
        elif name in ['apply', 'cuda', 'cpu', 'type', 'float', 'double', 'half', 'to', 'train']:
            def wrap_func(*args, **kwargs):
                try:
                    self.model.__getattribute__(name)(*args, **kwargs)
                except AttributeError:
                    self.model.__getattr__(name)(*args, **kwargs)
                return self
            return wrap_func
        else:
            try:
                return self.model.__getattribute__(name)
            except AttributeError:
                return self.model.__getattr__(name)

    def __call__(self, *args, **kwargs):
        try:
            return self.__getattribute__("forward")(*args, **kwargs)
        except AttributeError:
            return self.model.__call__(*args, **kwargs)

    def __setstate__(self, state):
        return self.model.__setstate__(state)

    def __getstate__(self):
        return self.model.__getstate__()

    def __str__(self):
        return "{}\n{}".format(self.__class__.__name__, self.model.__str__())

    def __repr__(self):
        return "{}\n{}".format(self.__class__.__name__, self.model.__repr__())


class PapeOptimizer(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __getattr__(self, name):
        try:
            return self.optimizer.__getattribute__(name)
        except AttributeError:
            return self.optimizer.__getattr__(name)
