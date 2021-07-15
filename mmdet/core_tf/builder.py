
from mmdet.utils.registry import Registry, build_from_cfg
ANCHOR_GENERATORS = Registry('Anchor generator Space')
MATCHER  = Registry("Matcher Space")
IOU_CALCULATOR = Registry("IOU CALCULATOR SPACE")
SAMPLER = Registry('SAMPLER SPACE')
BBOX_CODER =Registry("BBOX_CODER SPACE")
TARGET_ASSIGNER = Registry("Assigner Space")
def build_target_assigner(cfg, default_args=None):return build_from_cfg(cfg, TARGET_ASSIGNER)
def build_bbox_coder(cfg, default_args=None):
    return build_from_cfg(cfg, BBOX_CODER, default_args)
def build_sampler(cfg, default_args=None):
    return build_from_cfg(cfg, SAMPLER, default_args)
def build_iou_calculator(cfg, default_args=None):
    return build_from_cfg(cfg, IOU_CALCULATOR, default_args)
def build_matcher(cfg, default_args=None):
    return build_from_cfg(cfg, MATCHER, default_args)
def build_anchor_generator(cfg, default_args=None):
    return build_from_cfg(cfg, ANCHOR_GENERATORS, default_args)