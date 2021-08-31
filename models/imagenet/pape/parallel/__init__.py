from ..environ import pytorch_version_number, parrots_version_number

if pytorch_version_number != 0:
    from .distributed_model import DistributedPyTorchModel as DistributedModel
else:
    if parrots_version_number <= 400:
        import parrots  # noqa
        from parrots import config
        config.set_attr('engine', 'vect_allreduce', value=False)  # this will set in parrots0.5
    from .distributed_model import DistributedParrotsModel as DistributedModel


__all__ = [
  "DistributedModel",
]
