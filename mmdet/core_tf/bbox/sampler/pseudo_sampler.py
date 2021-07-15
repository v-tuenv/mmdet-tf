from mmdet.core_tf.builder import SAMPLER

import tensorflow as tf
@SAMPLER.register_module()
class PseudoSampler:
    def __init__(self,**kwargs):
        pass 
    
    
    def sampler(self, matches):
        '''matches: (N,)
            +> -2: ignore
            +> -1: background
            +> otherside match with gt_bboxes
            return :
                +> positive_inds : (N,1) :type=int
                +> negative_inds : (N,1) : type=int
        '''
        return matches