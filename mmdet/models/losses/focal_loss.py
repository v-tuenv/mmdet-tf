import tensorflow as tf
import tensorflow_addons as tfa
from ..builder import LOSSES,build_loss
from .utils import reduce_loss, weight_reduce_loss
@tf.function(experimental_relax_shapes=True)
def focal_loss_funtion(pred, target, alpha = 0.25, gamma = 2., label_smoothing = 0.):
    """y_true: shape = (batch, n_anchors, 1)
       y_pred : shape = (batch, n_anchors, num_class)
    """ 
    pred_prob = tf.nn.sigmoid(pred)
    target = tf.cast(target, pred_prob.dtype)
    p_t = (1-target )* pred_prob + target * (1 - pred_prob)
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor =  tf.pow(p_t,gamma)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=pred)
    loss_without_weights= alpha_factor * modulating_factor * ce
    return loss_without_weights

@LOSSES.register_module()
class FocalLoss(tf.keras.layers.Layer):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
    @tf.function(experimental_relax_shapes=True)
    def call(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        # print(pred.shape, target.shape, weight.shape,avg_factor,reduction_override)
        assert reduction_override in (None, 'none', 'mean', 'sum')
#         print(pred.shape,target.shape,"focal")
        print(pred.shape)
        print(target.shape)
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            num_classes = pred.shape[1]
            target = tf.one_hot(target,  depth=num_classes)
            loss_v = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True,alpha=1.,gamma=0., reduction = tf.keras.losses.Reduction.NONE,)
            # loss_cls = tf.nn.sigmoid_cross_entropy_with_logits(target, pred)
            loss_cls = focal_loss_funtion(pred, target)
            loss_cls = tf.math.reduce_sum(loss_cls,-1)
            if weight is not None:
                weight = tf.reshape(weight,(-1,))
            loss = weight_reduce_loss(loss_cls, weight, reduction, avg_factor)
        else:
            raise NotImplementedError
        return loss
