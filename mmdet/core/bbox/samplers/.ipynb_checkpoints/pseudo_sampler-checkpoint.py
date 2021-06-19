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
    
    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        """Directly returns the positive and negative indices  of samples.
        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes
        Returns:
            :obj:`SamplingResult`: sampler results
        """
        print("trace sample")
        print(assign_result.gt_inds.shape)
        print(bboxes.shape)
        print(gt_bboxes.shape)
        print("trace done")
        sh = assign_result.gt_inds.shape[0]
        if sh is None:
            sh= -1
        pos_inds =tf.reshape(tf.where(assign_result.gt_inds > 0,1,0),(sh,))
        # neg_inds =tf.reshape(tf.where(tf.equal(assign_result.gt_inds,0)), (sh,))
        neg_inds = tf.where(assign_result.gt_inds==0,1,0)
        print(neg_inds.shape)
        neg_inds = tf.reshape(neg_inds,[sh,])
        
        # gt_flags = tf.zeros(shape=(bboxes.shape[0],), dtype=tf.uint16)
        
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result)
        return sampling_result
