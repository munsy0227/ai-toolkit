from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None


def get_accelerator() -> Accelerator:
    global global_accelerator
    if global_accelerator is None:
        global_accelerator = Accelerator()
    return global_accelerator

def unwrap_model(model):
    try:
        accelerator = get_accelerator()
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
    except Exception as e:
        pass
    return model

import torch

_orig_prepare_model = Accelerator.prepare_model
def _safe_prepare_model(self, model, **kwargs):
    try:
        return _orig_prepare_model(self, model, **kwargs)
    except TypeError as e:
        if "device()" in str(e) and "NoneType" in str(e):
            # CPU 파라미터가 껴 있을 때 → 그냥 model 반환
            return model.to(self.device)
        raise
Accelerator.prepare_model = _safe_prepare_model
