from mmdet.utils.registry import Registry,build_from_cfg

PIPELINE = Registry("Pipeline")
def build_pipeline(cfg, **default_args):
    """Builder of pipeline."""
    return build_from_cfg(cfg, PIPELINE, default_args)