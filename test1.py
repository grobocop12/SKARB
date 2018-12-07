#! /usr/bin/python
import theano
import numpy

accuracy = 0.001
target_value = 20
x = theano.tensor.fscalar('x')
target = theano.tensor.fscalar('target')

W = theano.shared(numpy.asarray([0.2,0.7,0.1]), 'W')
y = x**2 * W[0] + x*W[1] + W[2]

cost = theano.tensor.sqr(target - y)
gradients = theano.tensor.grad(cost, [W])

W_updated = W- (0.08 * gradients[0])
updates = [(W, W_updated)]

f = theano.function([x,target],y, updates=updates)

print('Cel: %s' % target_value)
print('')

i = 0

while i<100:
    print('i: %s' % i)
    print('Wagi: %s' % W.get_value())

    wynik = f([1,1],target_value)
    print('Wynik: %s \n' % f([1,1],target_value))
    
    if abs(wynik-target_value) < accuracy:
        break
