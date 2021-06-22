import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import size

def f1(*args):
    tf.print(*args)
    a  = args[1].read(0)
    a = a+1
    args[1].write(args[0],a)
    return args[0]+1,args[1]

def f2(*args):
    return args
x=tf.convert_to_tensor([[0.,0.,0.,0.],[1.,1.,1.,1.]])

tf.config.run_functions_eagerly(False) 
@tf.function
def ok(xz):
    max_seq_len=xz.shape[0]
    for i in tf.range(max_seq_len):
        tf.print(i)
        print("call {}".format(xz[i]))
# tf.
# ok(x)
# ok(x)

@tf.function
def ok2(xz):
    
    max_seq_len=xz.shape[0]
    tf.print(max_seq_len)
    concat_anchor_list = tf.TensorArray(dtype=tf.float32, size=max_seq_len, dynamic_size=True,clear_after_read=False)
    for i in tf.range(max_seq_len):
        # tf.print(xz[i])
        a = tf.math.reduce_sum(xz[i])
        tf.print(a)
        concat_anchor_list.write(i,a)
        # print('call')
    tf.print('inside ',concat_anchor_list)
    return concat_anchor_list.stack()

a = tf.convert_to_tensor([
    [0.,0.,0.,0.],
    [0.,0.,1.,1.]
])
# tf.print(ok2(a))
# tf.print(ok2(a))

x=ok2(a)

print(x)