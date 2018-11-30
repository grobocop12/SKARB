import math
import matplotlib
import matplotlib.pyplot as plt
import sympy
import pandas
from pylab import *

def ss(alfa ,beta,targetheight,airresistance,windforce,v0):
    podniesienie= alfa * math.pi / 180
    wyprzedzenie = beta * math.pi / 180
    winx = windforce[0] # wiatr równoległy
    winy = windforce[1] # wiatr wznoszący
    winz = windforce[2] # wiatr boczny

    # Physical constants
    g = 9.8   # stała grawitacji
    m = 1.0   # masa
    rho = airresistance  # opór powietrza
    Cd = 1.0   # kaliber
    A = math.pi * pow(0.01, 2.0)
    alpha = rho * Cd * A / 2.0
    beta = alpha / m

    # Initial conditions
    X0 = 1.0
    Y0 = 1.0
    Z0 = 0.0
    Vx0 = math.cos(podniesienie)*math.cos(wyprzedzenie)*v0
    Vy0 = math.sin(podniesienie)*v0
    Vz0 = math.sin(wyprzedzenie)*math.cos(podniesienie)*v0
    # Time steps
    steps = 100000
    t_HIT = 2.0 * Vy0 / g
    dt = t_HIT / steps

    # With drag
    X_WD = list()
    Y_WD = list()
    Z_WD = list()
    Vx_WD = list()
    Vy_WD = list()
    Vz_WD = list()

    X_WD.append(X0)
    Y_WD.append(Y0)
    Z_WD.append(Z0)
    Vx_WD.append(Vx0)
    Vy_WD.append(Vy0)
    Vz_WD.append(Vz0)

    stop = 0
    for i in range(1, steps + 1):
        if stop != 1:
            speed = (pow(pow(Vx_WD[i - 1], 2.0) + pow(Vy_WD[i - 1], 2.0), 0.5))
            speed = pow(pow(speed, 2.0) + pow(Vz_WD[i - 1], 2.0), 0.5)
            # First calculate velocity
            Vx_WD.append(Vx_WD[i - 1] + ( -winx - beta * Vx_WD[i - 1]*speed )* dt)
            Vy_WD.append(Vy_WD[i - 1] + (- winy -g - beta * Vy_WD[i - 1] * speed) * dt)
            Vz_WD.append(Vz_WD[i - 1] + ( -winz - beta * Vz_WD[i - 1]*speed )* dt)
            # Now calculate position
            X_WD.append(X_WD[i - 1] + Vx_WD[i - 1] * dt)
            Y_WD.append(Y_WD[i - 1] + Vy_WD[i - 1] * dt)
            Z_WD.append(Z_WD[i - 1] + Vz_WD[i - 1] * dt)
            # Stop if hits ground
            if (Y_WD[i] <= targetheight)&(Y_WD[i]<Y_WD[i-1]):
                stop = 1

    return X_WD, Y_WD, Z_WD


def score (a,b,h,r,w,v0,d):

    X_WD, Y_WD, Z_WD =   ss(b, a, h, r, w,v0)
    Y= Y_WD[Y_WD.__len__()-2]/(Y_WD[Y_WD.__len__()-2]-Y_WD[Y_WD.__len__()-1])
    X = Y*(X_WD[X_WD.__len__()-2]-X_WD[X_WD.__len__()-1]) + X_WD[X_WD.__len__()-1]
    Z  = Y*(Z_WD[Z_WD.__len__()-2]-Z_WD[Z_WD.__len__()-1]) + Z_WD[Z_WD.__len__()-1]
    Xscore = 1/(math.fabs(X-d)+1)
    Zscore = 1/(math.fabs(Z)+1)


    return Xscore, Zscore
def shotingaglevertical2(distance,targetheight):
    onpoint=0
    lastshot =0;
    acturalshot=0;
    startagle=0.0001;
    startwector=20;
    scorelist=[]
    pudlo="trafiony"
    wathdog=0
    while(1>onpoint):
        acturalshot =
        wathdog=wathdog+1
        if(acturalshot>lastshot):
            startagle=startagle+startwector
            lastshot = acturalshot
        if(acturalshot<lastshot):
            startwector=startwector*(-2/5)
            startagle = startagle+startwector
            lastshot = acturalshot
        if(0.999<acturalshot<1.001):
            onpoint=2
        if(wathdog>200):
            pudlo="poza zasięiem"
            onpoint=2
        scorelist.append(acturalshot)
    return startagle, acturalshot, scorelist, pudlo