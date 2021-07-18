import warnings
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import size
import tensorflow_addons as tfa
from ..builder import NECKS
from ..common.conv_att_bn import ConvModule
class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self,size_image):
        super().__init__()
        self.size_image = size_image
    def get_config(self):
        a=super().get_config()
        a.update({"size_image":self.size_image})
    def call(self ,inputs):
        return tf.compat.v1.image.resize_nearest_neighbor(
                        inputs, self.size_image)


@NECKS.register_module()
class FPN(tf.keras.layers.Layer):
    r"""Feature Pyramid Network.
    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed
            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest',scale_factor=2),
                
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        
 

            
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        
        upsample_cfg = upsample_cfg.copy()
        upsample_cfg['interpolation'] = upsample_cfg.pop("mode",'nearest')
        interpolation = upsample_cfg['interpolation']
        if 'scale_factor' in upsample_cfg:
            upsample_cfg['size'] = upsample_cfg.pop("scale_factor")
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = []
        self.fpn_convs = []

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding='SAME',
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding='SAME',
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.fun_upsample = tf.keras.layers.UpSampling2D(**upsample_cfg)
        self.fun_max = tf.keras.layers.MaxPool2D(pool_size=1, strides=2)
        self.use_image_resize = 'size' in upsample_cfg
    @staticmethod
    def make_funtion_model(in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest',scale_factor=2),
                 return_funtion=False,
                 inputs = None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        
        num_ins = len(in_channels)
        upsample_cfg = upsample_cfg.copy()
        upsample_cfg['interpolation'] = upsample_cfg.pop("mode",'nearest')
        interpolation = upsample_cfg['interpolation']
        if 'scale_factor' in upsample_cfg:
            upsample_cfg['size'] = upsample_cfg.pop("scale_factor")
        if end_level == -1:
            backbone_end_level = num_ins
            assert num_outs >= num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                add_extra_convs = 'on_input'
            else:
                add_extra_convs = 'on_output'

        lateral_convs = []
        fpn_convs = []
        # print(in_channels)
        if inputs is None:
            inputs = [tf.keras.layers.Input(shape=(None,None,i)) for i in in_channels]
        for i in range(start_level, backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding='SAME',
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            lateral_convs.append(l_conv)
            fpn_convs.append(fpn_conv)
        extra_levels = num_outs - backbone_end_level + start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and add_extra_convs == 'on_input':
                    in_channels = in_channels[backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding='SAME',
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                fpn_convs.append(extra_fpn_conv)

        fun_upsample = tf.keras.layers.UpSampling2D(**upsample_cfg)
        fun_max = tf.keras.layers.MaxPool2D(pool_size=1, strides=2)
        use_image_resize = 'size' in upsample_cfg
        
        laterals = [
            lateral_conv(inputs[i + start_level])
            for i, lateral_conv in enumerate(lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if use_image_resize:
                laterals[i - 1] =tf.keras.layers.Add()([laterals[i-1],fun_upsample(laterals[i])])
            else:
                raise ValueError("implement this")
        outs = [
            fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        if num_outs > len(outs):
            if not add_extra_convs:
                for i in range(num_outs - used_backbone_levels):
                    outs.append(fun_max(outs[-1]))
            else:
                if add_extra_convs == 'on_input':
                    extra_source = inputs[backbone_end_level - 1]
                elif add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, num_outs):
                    if relu_before_extra_convs:
                        outs.append(fpn_convs[i](tf.keras.layers.Activation('relu')(outs[-1])))
                    else:
                        outs.append(fpn_convs[i](outs[-1]))

        return tf.keras.Model(inputs=inputs, outputs=tuple(outs))

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        """Forward function."""
        laterals = [
            lateral_conv(inputs[i + self.start_level], training=training)
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if self.use_image_resize:
                laterals[i - 1] = laterals[i-1] +  self.fun_upsample(laterals[i], training=training)
            else:
                prev_shape = laterals[i - 1].shape[-3:-1]
                up_s=tf.compat.v1.image.resize_nearest_neighbor(
                        laterals[i], [prev_shape[0], prev_shape[1]])
                
                laterals[i - 1] =  laterals[i - 1] + up_s
        outs = [
            self.fpn_convs[i](laterals[i], training=training) for i in range(used_backbone_levels)
        ]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(self.fun_max(outs[-1]))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source,training=training))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](tf.nn.relu(outs[-1]), training=training))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1],training=training))
        return tuple(outs)
    
    def call_function(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if not self.use_image_resize:
                laterals[i - 1] = laterals[i-1] +  self.fun_upsample(laterals[i])
            else:
                prev_shape = laterals[i - 1].shape[-3:-1]
                up_s=tf.compat.v1.image.resize_nearest_neighbor(
                        laterals[i], [prev_shape[0], prev_shape[1]])
                
                laterals[i - 1] =  laterals[i - 1] + up_s
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):

                    outs.append(self.fun_max(outs[-1]))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](tf.nn.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
   