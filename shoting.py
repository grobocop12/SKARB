import math

def shotingvertical(alfa,distance,targetheight):
    i=1
    v0=100
    g0=10
    ontarget=0
    score =0
    while(1>ontarget):
        x=v0*i*math.cos(alfa)
        y=v0*i*math.sin(alfa) - 1/2*i*i*g0
        i=i+1
        if(x>distance):
            ontarget=2
            score = 1/(math.fabs(targetheight-y)+1)
        if(y<0):
            ontarget=2
            score = 1/(math.fabs(x-distance)+1)
    return score