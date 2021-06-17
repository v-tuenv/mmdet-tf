from mmdet_planing.test_builder.builder import MODELS

@MODELS.register_module()
class ModelBuild:
    def __init__(self,
                 strides,
                 ratios,
                 k=1) -> None:
        self.strides = strides
        self.ratios = ratios
        self.k=k
    