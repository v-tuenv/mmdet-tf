# PLAN 

## Thang 6:
- [x] builder
    [x] test-sandbox-builder
- [ ] Anchor Generator Space
    - [x] Base Anchor Generator
    - [x] SSD Anchor Generator
- [x] coder
- [x] iou_calculators
- [x] assigner 
- [x] sampler 
- [x] retinanet

## Tháng 7-Big update 

- [x] Update new OOP-with graph mode training funtion.
    - [x] New Detector design
    - [ ] convert all funtion and object to only  tf.funtion and tf.Tensor.

- [x] Rename funtion and classes 
- [ ] target training time at least 16 imgs/s-now is 10 imgs/s with eagerly mode.
    - [ ] PipeLine data tensorflow support auto_aug: [paper](https://arxiv.org/abs/2103.13886)
    - [ ] Optional add augmentation with pipeline tf.
- [ ] Implement SGD one-cycle detectron2 [paper](https://arxiv.org/abs/1708.07120)
- [ ] Implement Gradient checkpointing Redue memories training [paper](https://arxiv.org/abs/2103.13886)
# Models
- [x] OneStage detector
- [x] Retinanet FPN50
## Thang 8-Big Update
- [ ] Implement two stage detector


