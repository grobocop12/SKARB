import math
import matplotlib
import matplotlib.pyplot as plt
import sympy
import pandas
from pylab import *
import csv

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

def precision(alfa ,beta,targetheight,airresistance,windforce,v0):
    X_WD, Y_WD, Z_WD = ss(alfa ,beta,targetheight,airresistance,windforce,v0)
    Y = (Y_WD[Y_WD.__len__() - 2] - targetheight) / ((Y_WD[Y_WD.__len__() - 2]-targetheight) + math.fabs((Y_WD[Y_WD.__len__() - 1])-targetheight))
    Xdis=math.fabs(X_WD[X_WD.__len__() - 2] - X_WD[X_WD.__len__() - 1])
    Zdis=math.fabs(Z_WD[Z_WD.__len__() - 2] - Z_WD[Z_WD.__len__() - 1])
    X = Y * (Xdis) + X_WD[X_WD.__len__() - 2]

    Z = Y * (Zdis) + Z_WD[Z_WD.__len__() - 2]
    return X,targetheight,Z

def csvgen(liczba):
    print ('Start')
    with open('genfile_file_seed145_winforcezyx_big.csv', mode='a+') as genfile_file:
        genfile_writer = csv.writer(genfile_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        genfile_writer.writerow(['kat podniesienia', 'kat boczny', 'wysokosc celu', 'opor powietrza', 'wiatr x', 'wiatr y', 'wiatr z' ,'wylotowa', 'X', 'Z'])
        sed =seed(145)
        np.random.seed(145)
        iterator=0
        while(iterator<liczba):
            iterator = iterator + 1

            alfa=90*np.random.random_sample()
            beta=180*np.random.random_sample()-90
            targetheight = 2
            airresistance = 0.1
            windforce = [np.random.random_sample()-0.5,0,np.random.random_sample()-0.5]
            v0 = 1100
            winx = windforce[0]  # wiatr równoległy
            winy = windforce[1]  # wiatr wznoszący
            winz = windforce[2]

            X,Y,Z = precision(alfa, beta, targetheight,airresistance , windforce, v0)


            genfile_writer.writerow([alfa,beta,targetheight,airresistance,winx,winy,winz,v0,X,Z])
            print(iterator)
        wynik = 'Wygenerowano następującą liczbę strzałów:'+ str(liczba)

        print(wynik)
        return wynik


def csvgen2(liczba):
    print('Start')
    with open('genfile_file_seed145_onlya.csv', mode='w') as genfile_file:
        genfile_writer = csv.writer(genfile_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        genfile_writer.writerow(
            ['kąt podniesienia', 'kat boczny', 'wysokosc celu', 'opor powietrza', 'wiatr x', 'wiatr y', 'wiatr z','wylotowa', 'X', 'Z'])
        sed = seed(145)
        iterator = 0
        np.random.seed(145)
        while (iterator < liczba):
            iterator = iterator + 1

            alfa = 90 * np.random.random_sample()
            beta = 0
            targetheight = 0
            airresistance = 0.1
            windforce = [0,0,0]
            v0 = 1100
            winx = windforce[0]  # wiatr równoległy
            winy = windforce[1]  # wiatr wznoszący
            winz = windforce[2]

            X, Y, Z = precision(alfa, beta, targetheight, airresistance, windforce, v0)

            genfile_writer.writerow([alfa, beta, targetheight, airresistance, winx, winy, winz, v0, X, Z])
            print(iterator)
        wynik = 'Wygenerowano następującą liczbę strzałów:' + str(liczba)

        print(wynik)
        return wynik


def csvgen3(liczba):
    print('Start')
    with open('genfile_file_seed145_winforcez.csv', mode='w') as genfile_file:
        genfile_writer = csv.writer(genfile_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        genfile_writer.writerow(
            ['kąt podniesienia', 'kat boczny', 'wysokosc celu', 'opor powietrza', 'wiatr X', 'wiatr y', 'wiatr z',
             'wylotowa', 'X', 'Z'])
        sed = seed(145)
        np.random.seed(145)
        iterator = 0
        while (iterator < liczba):
            iterator = iterator + 1

            alfa = 90 * np.random.random_sample()
            beta = 180 * np.random.random_sample() - 90
            targetheight = 0
            airresistance = 0.1
            windforce = [0,0,np.random.random_sample()]
            v0 = 1100
            winx = windforce[0]  # wiatr równoległy
            winy = windforce[1]  # wiatr wznoszący
            winz = windforce[2]

            X, Y, Z = precision(alfa, beta, targetheight, airresistance, windforce, v0)

            genfile_writer.writerow([alfa, beta, targetheight, airresistance, winx, winy, winz, v0, X, Z])
            print(iterator)
        wynik = 'Wygenerowano następującą liczbę strzałów:' + str(liczba)

        print(wynik)
        return wynik


def gentab(iteracje):
    licznik =0
    kat = []
    zasieg = []
    while(licznik<iteracje):
        alfa = 90 * np.random.random_sample()
        beta = 0
        targetheight = 0
        airresistance = 0.2
        windforce = [0, 0, 0]
        v0 = 600

        X, Y, Z = precision(alfa -1*np.random.normal() + np.random.normal(), beta, targetheight, airresistance, windforce, v0)
        kat.append(alfa)
        zasieg.append(X)
        licznik = licznik +1
        print(licznik)


    kat = transpose(kat)
    return kat, zasieg


'''X,Y ,Z= precision(20,0.0,2,0.1,[1,0,0],1100)
print(X,Y,Z)
X,Y,Z = ss(20,0.0,2,0.1,[1,0,0],1100)
X1 =X[X.__len__() - 1]
Y1=Y[Y.__len__() - 1]
Z1= Z[Z.__len__() - 1]

print( X1,Y1,Z1)
X2 =X[X.__len__() - 2]
Y2=Y[Y.__len__() - 2]
Z2= Z[Z.__len__() - 2]
print( X2,Y2,Z2)
'''
'''
z = csvgen(10000)
f = csvgen2(10000)
g = csvgen3(10000)

'''



'''
X20, h20, Z20 = precision(20 ,20,0,0.1,[1,0,1],1100)
print(X20)
print(Z20)
'''
z = csvgen(50000)