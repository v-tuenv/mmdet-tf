import os,sys,json
from pathlib import Path
PATH =Path(os.getcwd()).parent
sys.path.append(str(PATH.absolute()) + "/")
PATH =Path(os.getcwd())
sys.path.append(str(PATH.absolute()) + "/")

from mmdet import core
import tensorflow as tf
tf.config.run_functions_eagerly(False)
from mmdet.models.losses.smooth_l1_loss import L1Loss,SmoothL1Loss
loss_cls = SmoothL1Loss()

pred = tf.convert_to_tensor([[10.,10,100,200,],[400,600,800,200]])
tar = tf.convert_to_tensor([[20.,20,100,200,],[400,600,800,200]])
a = loss_cls(pred, tar)
print(a)
