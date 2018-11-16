import math
import matplotlib

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

    while(1>onpoint):
        acturalshot=shotingvertical(startagle,distance,targetheight)

        if(acturalshot>lastshot):
            startagle=startagle+startwector
            lastshot = acturalshot
        if(acturalshot<lastshot):
            startwector=startwector*(-1/2)
            startagle = startagle+startwector
            lastshot = acturalshot
        if(acturalshot==1):
            onpoint=2
        scorelist.append(acturalshot)
    return startagle, acturalshot, scorelist

x,lastshot,scorelist = shotingaglevertical(120,2)
import matplotlib.pyplot as plt
plt.plot(scorelist)
plt.show()

print (x)


