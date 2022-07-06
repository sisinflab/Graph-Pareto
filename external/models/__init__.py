def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break

import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .gcn.GCN import GCN
        from .lightgcn.LightGCN import LightGCN
        from .ultragcn import UltraGCN
