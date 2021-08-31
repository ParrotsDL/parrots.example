import logging
from ..utils.base import PapeModel
from ..utils.op import is_bn_module
from ..utils.tensor import to_float, to_half

logger = logging.getLogger()


class HalfModel(PapeModel):
    """
    Half precision (also named mixed precision) training model. In fact, if all the modules
    is half precision, the model always won't converge to a good preformance. So we need
    to use float parameter and float input/output in some modules. This class provides three
    parameters to control the float module.


    Arguments:
        * model (torch.nn.Module or pape.PapeModel): model to be transformed to Half Precision Model
        * float_bn (bool, optional): If True, all the BatchNorm modules will call bn.float(), but the input
          and output type is unchanged
        * float_module_type (dict, optional): A dict to set the module's type and its input and
          output type. e.g. ``{ nn.CrossEntropyLoss: ("float", "") }`` means all ``nn.CrossEntropyLoss`` module
          will call module.float(), and its input type will be transformed to float and its output type keeps
          unchanged. If both input and output type are unchange, set ``{ nn.CrossEntropyLoss: None}``
        * float_module_name (dict, optional): Just like ``float_module_type``, but the dict's key is the name
          of the module. e.g. ``{ "layer1": ("float", "half"), "layer2.conv": ("float", "") }``

    .. note::
        ``float_module_name`` can overide ``float_module_type``, ``float_module_type`` can overide ``float_bn``.
    """
    def __init__(self, model, float_bn=True, float_module_type=None, float_module_name=None):
        super(HalfModel, self).__init__(model)
        self.model.half()

        def make_float_forward(m, name="", config=None):
            pre_func = lambda x: x  # noqa
            post_func = lambda x: x  # noqa
            if config:
                assert len(config) == 2, "mod config need two element, but {} get {}".format(mod_type, config)
                assert config[0] in ("float", "half", ""), "parameter error {}".format(config[0])
                assert config[1] in ("float", "half", ""), "parameter error {}".format(config[1])
                func_dict = {'float': to_float, 'half': to_half}
                pre_func = func_dict[config[0]]
                post_func = func_dict[config[1]]

            if hasattr(m, "pape_old_forward_"):
                m.forward = m.pape_old_forward_
                logger.warning("!reset float module {} {}".format(name, config))
            m.pape_old_forward_ = m.forward

            def lambda_forward(*args, **kwargs):
                out = m.pape_old_forward_(*pre_func(args), **pre_func(kwargs))
                return post_func(out)
            m.forward = lambda_forward

        if float_bn:
            for name, mod in self.model.named_modules():
                if is_bn_module(mod):
                    mod.float()
                    logger.info("use float module {} {}".format(name, '("", "")'))

        if float_module_type:
            for mod_type, config in float_module_type.items():
                # assert torch.nn.Module will fail, I don't know the reason.
                # assert isinstance(mod_type, torch.nn.Module), "parameter error {}".format(mod_type)
                assert isinstance(mod_type, type), "parameter error {}".format(mod_type)

                for name, mod in self.model.named_modules():
                    if isinstance(mod, mod_type):
                        mod.float()
                        make_float_forward(mod, name, config)
                        logger.info("use float module {} {}".format(name, config))

        if float_module_name:
            for mod_name, config in float_module_name.items():
                assert isinstance(mod_name, str), "parameter error {}".format(mod_name)

                finded = False
                for name, mod in self.model.named_modules():
                    if name == mod_name:
                        mod.float()
                        make_float_forward(mod, name, config)
                        finded = True
                        logger.info("use float module {} {}".format(name, config))
                if not finded:
                    logger.warning("not find module {}".format(mod_name))

    def __call__(self, *args, **kwargs):
        args, kwargs = to_half(args), to_half(kwargs)
        try:
            return self.__getattribute__("forward")(*args, **kwargs)
        except AttributeError:
            return self.model.__call__(*args, **kwargs)
