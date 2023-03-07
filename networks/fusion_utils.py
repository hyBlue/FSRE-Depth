from .cma import CMA
from .roiformer.roiformer import RoiFormer


def get_fusion_module_class(fusion_type: str):
    if fusion_type == 'cma':
        return CMA
    elif fusion_type == 'roiformer':
        return RoiFormer
    else:
        raise NotImplementedError()
