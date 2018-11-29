import math
import matplotlib
import matplotlib.pyplot as plt
import sympy
import pandas


def shotingvertical(alfa,distance,targetheight):
    i=0
    v0=100
    g0=10
    ontarget=0
    score =0
    while(1>ontarget):
        x=v0*i*math.cos(alfa)
        y=v0*i*math.sin(alfa) - 1/2*i*i*g0
        i=i+0.00001
        if(x>distance):
            ontarget=2
            score = 1/(math.fabs(targetheight-y)+1)
        if(y<0):
            ontarget=2
            score = 1/(math.fabs(x-distance)+1+targetheight)
    return score
def shotinghorizontal(beta,targethorizont):
    score=(1 / (math.fabs(beta - targethorizont) + 1))
    return score

def shotingvertical2(alfa,distance,targetheight):

    i=0
    v0=100
    g0=10
    ontarget=0
    score =0
    height = []
    misledistance = []
    while(1>ontarget):
        x=v0*i*math.cos(alfa)
        y=v0*i*math.sin(alfa) - 1/2*i*i*g0
        i=i+0.0001
        height.append(y)

        misledistance.append(x)
        if(x>distance):
            ontarget=2
            score = 1/(math.fabs(targetheight-y)+1)
        if(y<0):
            ontarget=2
            score = 1/(math.fabs(x-distance)+1+targetheight)
    return score, height, misledistance

def shotingverticaldegrees(kat,distance,targetheight):
    alfa = kat*math.pi/180
    i=0
    v0=100
    g0=10
    ontarget=0
    score =0

    i = sympy.Symbol('i')
    czas = sympy.solve(v0 * i * math.sin(alfa) - (1 / 2) * i * i * g0)

    i = sympy.Symbol('i')
    czas2 = sympy.solve(v0*i*math.cos(alfa)-distance)

    if(czas[1]<czas2[0]):
        i = czas[1]
        x2 = float(v0) * float(i) * float(math.cos(alfa))
        score = float(1) / (math.fabs(x2 - float(distance)) + float(1) + float(targetheight))
    else:
        i = czas2[0]
        y2 = float(v0) * i * float(math.sin(alfa)) - float(1/2) * i * i * float(g0)
        score = float(1) / (math.fabs(float(targetheight) - y2) + float(1))

    return score

def shotingverticaldegrees2(kąt,distance,targetheight):
    alfa = kąt*math.pi/180
    i=0
    v0=100
    g0=10
    ontarget=0
    score =0
    height = []
    misledistance = []
    while(1>ontarget):
        x=v0*i*math.cos(alfa)
        y=v0*i*math.sin(alfa) - (1/2)*i*i*g0
        i=i+0.0001
        height.append(y)

        misledistance.append(x)
        if(x>distance):
            ontarget=2
            score = 1/(math.fabs(targetheight-y)+1)
        if(y<0):
            ontarget=2
            score = 1/(math.fabs(x-distance)+1+targetheight)
    i = sympy.Symbol('i')
    czas = sympy.solve(v0 * i * math.sin(alfa) - (1 / 2) * i * i * g0)
    print(czas)
    i = sympy.Symbol('i')
    czas2 = sympy.solve(v0*i*math.cos(alfa)-distance)
    print(czas2)
    if(czas[1]<czas2[0]):
        i = czas[1]
        x2 = float(v0) * float(i) * float(math.cos(alfa))
        score = float(1) / (math.fabs(x2 - float(distance)) + float(1) + float(targetheight))
    else:
        i = czas2[0]
        y2 = float(v0) * i * float(math.sin(alfa)) - float(1/2) * i * i * float(g0)
        score = float(1) / (math.fabs(float(targetheight) - y2) + float(1))

    return score, height, misledistance

def shotinghorizontal(beta,targethorizont):
    score=(1 / (math.fabs(beta - targethorizont) + 1))
    return score


def shotingaglevertical(distance,targetheight):
    onpoint=0
    lastshot =0;
    acturalshot=0;
    startagle=0;
    startwector=1/4*math.pi;
    scorelist=[]
    pudlo="trafiony"
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
        if(0.999<acturalshot<1.001):
            onpoint=2
        if(wathdog>10000):
            pudlo="poza zasięiem"
            onpoint=2
        scorelist.append(acturalshot)
    return startagle, acturalshot, scorelist, pudlo

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
        acturalshot = shotingverticaldegrees(startagle,distance,targetheight)
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

x,lastshot,scorelist, pudlo = shotingaglevertical(660,2)

def maxdisctance():
    range=800
    targeton= "trafiony"
    while (targeton== "trafiony"):
        x,lastshot,scorelist,pudlo=shotingaglevertical2(range,0)
        targeton=pudlo
        range=range+1
    return range

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

x, wysokosci, dystans = shotingvertical2(x,660,2)
plt.plot(dystans,wysokosci)
plt.show()
print(wysokosci)
print(dystans)

agle,lastshot,scorelist, pudlo = shotingaglevertical2(460,10)
x, wysokosci, dystans = shotingverticaldegrees2(agle,460,10)
print(x)
print (pudlo)
print(agle)
plt.plot(dystans,wysokosci)
plt.show()
plt.plot(scorelist)
plt.show()
score, height, misledistance,horizontal = shoting(20 ,0,700,2,0.00000,[1,0])
plt.plot(misledistance,height)
plt.show()
plt.plot(misledistance,horizontal)
plt.show()