
from mmdet.utils.registry import Registry

CONV_LAYERS = Registry('conv layer')
NORM_LAYERS = Registry('norm layer')
ACTIVATION_LAYERS = Registry('activation layer')
PADDING_LAYERS = Registry('padding layer')
UPSAMPLE_LAYERS = Registry('upsample layer')
print("this builder in dir_will_be_delete will be delete because this is depcrept with keras.layers.*")
# PLUGIN_LAYERS = Registry('plugin layer')

# DROPOUT_LAYERS = Registry('drop out layers')
# POSITIONAL_ENCODING = Registry('position encoding')
# ATTENTION = Registry('attention')
# FEEDFORWARD_NETWORK = Registry('feed-forward Network')
# TRANSFORMER_LAYER = Registry('transformerLayer')
# TRANSFORMER_LAYER_SEQUENCE = Registry('transformer-layers sequence')