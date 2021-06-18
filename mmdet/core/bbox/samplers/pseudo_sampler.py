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
        pos_inds =tf.reshape(tf.where(assign_result.gt_inds > 0),(-1,))
        neg_inds =tf.reshape(tf.where(tf.equal(assign_result.gt_inds,0)), (-1,))

        gt_flags = tf.zeros(shape=(bboxes.shape[0],), dtype=tf.uint16)
        
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
