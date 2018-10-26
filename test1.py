#! /usr/bin/python
import theano
import numpy

target_value = 20
x = theano.tensor.fvector('x')
target = theano.tensor.fscalar('target')

W = theano.shared(numpy.asarray([0.2,0.7]), 'W')
y = (x*W).sum()

cost = theano.tensor.sqr(target - y)
gradients = theano.tensor.grad(cost, [W])
W_updated = W- (0.1 * gradients[0])
updates = [(W, W_updated)]

f = theano.function([x,target],y, updates=updates)

print('Cel: %s' % target_value)
print('')

for i in range(100):
    print('i: %s' % i)
    print('Wagi: %s' % W.get_value())
    print('Wynik: %s \n' % f([1,1],target_value))
    
