from mmdet_planing.utils.registry import Registry, build_from_cfg



MODELS = Registry('model')
def build_anchor_generator(cfg, default_args=None):
    return build_from_cfg(cfg, MODELS, default_args)