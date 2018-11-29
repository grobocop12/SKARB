import math
import matplotlib
import matplotlib.pyplot as plt
import sympy
import pandas
from pylab import *


def shoting(alfa ,beta,distance,targetheight,airresistance,windforce):
    alfa= alfa * math.pi / 180
    beta = beta * math.pi / 180
    i=0
    v0=100
    g0=10
    ontarget=0
    score =0
    height = []
    misledistance = []
    horizontal=[]
    while(1>ontarget):
        z=v0*i*math.sin(beta) - 1/2*i*i*windforce[0]
        x=v0*i*math.cos(alfa)*math.cos(beta) - 1/2*i*i*windforce[1]
        y=v0*i*math.sin(alfa)*math.cos(beta) - 1/2*i*i*g0
        i=i+0.00001
        v0=(1-airresistance)*v0
        height.append(y)
        horizontal.append(z)
        misledistance.append(x)
        if(x>distance):
            ontarget=2
            score = 1/(math.fabs(targetheight-y)+1)
        if(y<0):
            ontarget=2
            score = 1/(math.fabs(x-distance)+1+targetheight)
    return score, height, misledistance,horizontal


def shotingdrag(alfa ,beta,distance,targetheight,airresistance,windforce,v0):
    alfa= alfa * math.pi / 180
    beta = beta * math.pi / 180


    # Physical constants
    g = 9.8   # stała grawitacji
    m = 1.0   # masa
    rho = 1.0  # opór powietrza
    Cd = 1.0   # kaliber
    A = math.pi * pow(0.01, 2.0)
    alpha = rho * Cd * A / 2.0
    beta = alpha / m

    # Initial conditions
    X0 = 1.0
    Y0 = 0.0
    Z0 = 0.0
    Vx0 = math.cos(alfa)*math.cos(beta)*v0
    Vy0 = math.sin(alfa)*math.cos(beta)*v0
    Vz0 = math.sin(beta)*v0
    # Time steps
    steps = 10000
    t_HIT = 2.0 * Vy0 / g
    dt = t_HIT / steps

    # No drag
    X_ND = list()
    Y_ND = list()

    for i in range(steps + 1):
        X_ND.append(X0 + Vx0 * dt * i)
        Y_ND.append(Y0 + Vy0 * dt * i - 0.5 * g * pow(dt * i, 2.0))

    # With drag
    X_WD = list()
    Y_WD = list()
    Vx_WD = list()
    Vy_WD = list()

    for i in range(steps + 1):
        X_ND.append(X0 + Vx0 * dt * i)
        Y_ND.append(Y0 + Vy0 * dt * i - 0.5 * g * pow(dt * i, 2.0))

    # With drag
    X_WD = list()
    Y_WD = list()
    Vx_WD = list()
    Vy_WD = list()

    X_WD.append(X0)
    Y_WD.append(Y0)
    Vx_WD.append(Vx0)
    Vy_WD.append(Vy0)

    stop = 0
    for i in range(1, steps + 1):
        if stop != 1:
            speed = pow(pow(Vx_WD[i - 1], 2.0) + pow(Vy_WD[i - 1], 2.0), 0.5)
            # First calculate velocity
            Vx_WD.append(Vx_WD[i - 1] * (1.0 - beta * speed * dt))
            Vy_WD.append(Vy_WD[i - 1] + (- g - beta * Vy_WD[i - 1] * speed) * dt)

            # Now calculate position
            X_WD.append(X_WD[i - 1] + Vx_WD[i - 1] * dt)
            Y_WD.append(Y_WD[i - 1] + Vy_WD[i - 1] * dt)

            # Stop if hits ground
            if (Y_WD[i] <= targetheight)&(Y_WD[i]<Y_WD[i-1]):
                stop = 1

    return X_WD, Y_WD, X_ND, Y_ND


score, height, misledistance,horizontal = shoting(20 ,0,700,2,0.00000,[1,5])
print(height[height.__len__()-1])
print("dystans")
print(misledistance[misledistance.__len__()-1])
print("znios")
print(horizontal[horizontal.__len__()-1])
plt.plot(misledistance,height)
plt.show()
plt.plot(misledistance,horizontal)
plt.show()