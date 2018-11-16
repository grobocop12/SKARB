#! usr/bin/python3/

import numpy as np
import theano.tensor as T
from theano import function

'''
import theano
from theano import tensor

a = tensor.dscalar()
b = tensor.dscalar()

c = a+b

f = theano.function([a,b],c)

print(f(1.5,2.5))
'''

'''
from theano import *
import theano.tensor as T
import numpy as np

A = np.asarray([[1,2], [3,4],[5,6]])
b= 2.0
print(A * b)
'''


'''
x = T.dscalar('x')
y = T.dscalar('y')

z = x+y
f = function([x,y],z)

print(f(2,3))
'''

'''
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x,y],z)

A = np.asarray([[1,2], [3,4],[5,6]])
B = np.flip(A)

print(f(A,B))
'''

a = T.vector()
b = T.dscalar()
out = a[0] **2 +b**2 + 2*a[1]*b
f = function([a,b], out)
print(f([0,1],1))
