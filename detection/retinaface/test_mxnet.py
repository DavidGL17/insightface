import mxnet as mx
import numpy as np

# create a matrix of size 1000x1000 on the GPU
a = mx.nd.random.uniform(shape=(1000, 1000), ctx=mx.gpu(0))

# perform a matrix multiplication on the GPU
b = mx.nd.dot(a, a)

# copy the result back to the CPU and print it
print(b.asnumpy())

# check what gpu is being used
print(mx.context.gpu())
