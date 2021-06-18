import tensorflow as tf

from ..builder import LOSSES,build_loss

def cls_loss(y_true, y_pred, alpha = 0.25, gamma = 1.5, label_smoothing = 0.):
    """y_true: shape = (batch, n_anchors, 1)
       y_pred : shape = (batch, n_anchors, num_class)
    """ 
#     print(y_true.shape, y_pred.shape)
    pred_prob = tf.sigmoid(y_pred)
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t)**gamma
    
    y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    
    return alpha_factor * modulating_factor * ce

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
        print(pred.shape, target.shape, weight.shape,avg_factor,reduction_override)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            # if torch.cuda.is_available() and pred.is_cuda:
            #     calculate_loss_func = sigmoid_focal_loss
            # else:
            #     num_classes = pred.size(1)
            #     target = F.one_hot(target, num_classes=num_classes + 1)
            #     target = target[:, :num_classes]
            #     calculate_loss_func = py_sigmoid_focal_loss
            tf.print("debuger focalloss")
            # loss_cls = self.loss_weight * calculate_loss_func(
            #     pred,
            #     target,
            #     weight,
            #     gamma=self.gamma,
            #     alpha=self.alpha,
            #     reduction=reduction,
            #     avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return tf.constant([0.])
