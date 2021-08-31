import re
import torch


def version2six(version):
    if version is None or len(version) == 0:
        return 0
    m = re.match(r'(\d+).(\d+).(\d+)', version)
    assert m, 'version {} is illegal'.format(version)
    vsix = int(m.group(1)) * 10000 + int(m.group(2)) * 100 + int(m.group(3))
    return vsix


def get_env_version():
    if torch.__version__ == 'parrots':
        import parrots
        return None, parrots.__version__
    else:
        return torch.__version__, None


def get_env_version_number():
    return [version2six(v) for v in get_env_version()]
