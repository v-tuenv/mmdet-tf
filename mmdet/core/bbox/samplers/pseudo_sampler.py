from operator import ne
import tensorflow as tf

from ..builder import BBOX_SAMPLERS
from .base_sample import BaseSampler
from .sampling_result import SamplingResult

@BBOX_SAMPLERS.register_module()
class PseudoSampler(BaseSampler):
    def __init__(self,**kwargs):
        pass
    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError
    @tf.function(experimental_relax_shapes=True)
    def sample(self,assigned_gt_inds, max_overlaps,assigned_labels, bboxes, gt_bboxes, **kwargs):
        """Directly returns the positive and negative indices  of samples.
        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes
        Returns:
            :obj:`SamplingResult`: sampler results
        """
        print("trace sample")
        sh = assigned_gt_inds.shape[0]
        if sh is None:
            sh= -1
        pos_inds =tf.reshape(tf.where(assigned_gt_inds > 0,1,0),(sh,))
        neg_inds = tf.where(assigned_gt_inds==0,1,0)
        neg_inds = tf.reshape(neg_inds,[sh,])
        
        return pos_inds,neg_inds
