from abc import ABCMeta, abstractmethod
import tensorflow as tf

class BaseDenseHead(tf.keras.layers.Layer, metaclass=ABCMeta):
    """Base class for DenseHeads."""
    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__()

    @abstractmethod
    @tf.function(experimental_relax_shapes=True)
    def mloss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass
    
    @tf.function(experimental_relax_shapes=True)
    def forward_train(self,
                      x,
                      gt_bboxes,
                      gt_labels=None,
                      batch_size=None,
                      proposal_cfg=None,
                      ):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
#         print("trace base dense")
        outs = self(x, training=True)
        if batch_size is not None:
            gt_bboxes = tf.unstack(gt_bboxes,batch_size)
            if gt_labels is not None:
                gt_labels=tf.unstack(gt_labels, batch_size)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes,)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, )
#         print(loss_inputs)
        losses = self.mloss(*loss_inputs)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs,  cfg=proposal_cfg)
            return losses, proposal_list

    def simple_test(self, feats, rescale=False):
        """Test function without test-time augmentation.
        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        return self.simple_test_bboxes(feats, rescale=rescale)