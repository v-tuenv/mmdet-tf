import os,sys,json
from pathlib import Path
PATH =Path(os.getcwd()).parent
sys.path.append(str(PATH.absolute()) + "/")
PATH =Path(os.getcwd())
sys.path.append(str(PATH.absolute()) + "/")
from mmdet_planing import test_builder, MODELS
# from mdet_plaining import 

config = {'strides':1,'ratios':2,'type':'ModelBuild'}

print("test-no-override")
instance = test_builder.build_anchor_generator(config)
assert (instance.k == 1 and instance.strides == 1 and instance.ratios==2), print(instance.k,instance.strides,instance.ratios)
print("pass")

print("test override")
config = {'strides':1,'ratios':2,'type':'ModelBuild','k':4}
instance = test_builder.build_anchor_generator(config)
assert (instance.k == 4 and instance.strides == 1 and instance.ratios==2)

print("pass")


