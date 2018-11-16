import math
import matplotlib
import matplotlib.pyplot as plt

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
def shotinghorizontal(beta,targethorizont):
    score=(1 / (math.fabs(beta - targethorizont) + 1))
    return score

x =shotingvertical(1/4*math.pi,100,2)
print (x)

def shotingaglevertical(distance,targetheight):
    onpoint=0
    lastshot =0;
    acturalshot=0;
    startagle=0;
    startwector=1/4*math.pi;
    scorelist=[]
    pudło="trafiony"
    wathdog=0
    while(1>onpoint):
        acturalshot=shotingvertical(startagle,distance,targetheight)
        wathdog=wathdog+1
        if(acturalshot>lastshot):
            startagle=startagle+startwector
            lastshot = acturalshot
        if(acturalshot<lastshot):
            startwector=startwector*(-1/2)
            startagle = startagle+startwector
            lastshot = acturalshot
        if(0.99999<acturalshot<1.0001):
            onpoint=2
        if(wathdog>10000):
            pudło="poza zasięiem"
            onpoint=2
        scorelist.append(acturalshot)
    return startagle, acturalshot, scorelist, pudło

x,lastshot,scorelist, pudło = shotingaglevertical(780,2)
def maxdisctance():
    range=220
    targeton= "trafiony"
    while (targeton== "trafiony"):
        x,lastshot,scorelist,pudło=shotingaglevertical(range,0)
        targeton=pudło
        range=range+1
    return range
max= maxdisctance()
plt.plot(scorelist)
plt.show()


print (x)
print (lastshot)
print (pudło)
print(max)

