from math import log, sin
from math import exp
import random
import matplotlib.pyplot as plt
import numpy as np
import time

w = 5000000
g0 = -40
theta = 4
d0 = 1
d = 100
N0 = -174
FRQ = 1e9
power = 5

class Task :
    def __init__(self, D, C) : #D: data size C: cpu cycles
        self.datasize = D
        self.cpucycles = C
    
    def TransmissionTime(self, p) :
        return self.datasize / TransmissionRate(p) 
        
    def Disposetime(self) :
        return self.datasize * self.cpucycles / FRQ

class instance :
    def __init__(self, dir) :
        self.tasklist = list()
        with open(dir) as f :
            #datasize,cycle
            lines=f.readlines()
            dim = int(lines[0])
            i = 1
            while(i < dim - 1) :
                curline = lines[i].split()
                self.tasklist.append(Task(int(curline[1]), int(curline[2])))  
                i += 1

def TransmissionRate(p) :
    # return w * log(1 + g0 * (d0 / d) ** theta * p / (w * N0))
    return w * log(1 + 1e-12 * p  / (w * 3.981071705534985E-18)) / 0.6931471805599453

def makespan(seq, tasklist) : #seq records index
    readytime = list()
    completetime = list()

    isfirst = True
    for i in seq :
        if isfirst == True :       
            readytime.append(tasklist[i].TransmissionTime(power))
            isfirst = False
            
        else:
            readytime.append(readytime[-1] + tasklist[i].TransmissionTime(power))

    isfirst = True
    for i in seq :
        if isfirst == True :
            completetime.append(readytime[i] + tasklist[i].Disposetime())
            isfirst = False
        else :
            completetime.append(max(readytime[i], completetime[-1]) + tasklist[i].Disposetime())

    return completetime[-1]


def subtraction(position1, position2) :
    velocity = list()
    N = len(position1)
    currentposition = position2.copy()
    for i in range(N) :
        idx = currentposition.index(position1[i])
        if idx != i :
            velocity.append((i, idx))
            temp = currentposition[i]
            currentposition[i] = currentposition[idx]
            currentposition[idx] = temp
    return velocity

def addition_p_v(position, velocity) :
    newposition = position.copy()
    for item in velocity :
        temp = newposition[item[0]]
        newposition[item[0]] = newposition[item[1]]
        newposition[item[1]] = temp
    return newposition

def addition_v_v(velocity1, velocity2) :
    return velocity1 + velocity2

def multiplication(c, velocity) :
    L = len(velocity)
    return velocity[0 : int(c * L)]


def footrule(a, b) :
    distance = 0
    for i in range(len(a)) :
        if a[i] != b[i] :
            distance += 1
    return distance


class particle :
    def __init__(self, tasklist) :
        self.tasklist = tasklist
        self.dim = len(tasklist)
        self.v = list() 
        L = random.randrange(0, self.dim)

        for i in range(L) :
            x0 = random.randrange(0, self.dim)
            x1 = x0
            while(x1 == x0) :
                x1 = random.randrange(0, self.dim)
            self.v.append((x0, x1))

        self.position = [i for i in range(self.dim)]
        random.shuffle(self.position)
        self.best = self.position.copy()
        self.fit = self.fitness()

    def fitness(self) :
        return makespan(self.position, self.tasklist)
        
    def updatevelocity(self, c1, c2, c3, Gbest) :
        Pcurrent = self.position
        Pbest = self.best

        self.v = list() 
        L = random.randrange(0, self.dim)

        for i in range(L) :
            x0 = random.randrange(0, self.dim)
            x1 = x0
            while(x1 == x0) :
                x1 = random.randrange(0, self.dim)
            self.v.append((x0, x1))


        v0 = multiplication(c1, self.v)
        v1 = multiplication(c2, subtraction(Pbest,  Pcurrent))
        v2 = multiplication(c3, subtraction(Gbest, Pcurrent))
        v = addition_v_v(v0, addition_v_v(v1, v2))
       
        
        self.v = v
    
    def updateposition(self, c1, c2, c3, Gbest) :
        oldfitness = self.fit
        self.updatevelocity(c1, c2, c3, Gbest)
        self.position = addition_p_v(self.position, self.v)
        self.fit = self.fitness()
        if self.fit < oldfitness :
            self.best = self.position.copy()

class pso :
    def __init__(self, tasklist, population, max_iter) :
        self.swarms = list()
        self.tasklist = tasklist
        self.dim = len(tasklist)
        self.max_iter = max_iter
        self.bestvalues = list()
        self.besttrace = list()
        self.population = population

        for i in range(population) :
            self.swarms.append(particle(tasklist))
    
    def optimize(self, c1, c2, c3) :
        self.bestvalues.clear()
        self.besttrace.clear()

        for iter in range(self.max_iter) :
            print(iter)
            self.swarms = sorted(self.swarms, key = lambda item : item.fit)
            if self.bestvalues != [] :
                if self.swarms[0].fit < self.bestvalues[-1] :
                    self.bestvalues.append(self.swarms[0].fit)
                    self.besttrace.append(self.swarms[0].position)
                else :
                    self.bestvalues.append(self.bestvalues[-1])
                    self.besttrace.append(self.besttrace[-1])
            else :
                self.bestvalues.append(self.swarms[0].fit)
                self.besttrace.append(self.swarms[0].position)

            Gbest = self.besttrace[-1]
            for i in self.swarms :
                i.updateposition(c1, c2, c3, Gbest)

if __name__ == '__main__' :
    ins = instance('./TestingInstances/100/100_1.txt')
    tasks = ins.tasklist

    iterations = 1500
    population = 100

    optimizer = pso(tasks, population, iterations)

    time_start = time.time() 
    optimizer.optimize(0.75, 0.25, 0.25)
    time_end = time.time()
    time_c1= time_end - time_start
    x1 = range(iterations)
    y1 = optimizer.bestvalues

    fig = plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(x1, y1, label = 'best_value')
    ax1.set_xlabel('iterations')
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    ax1.legend()
    
    ax1.set_title('PSO: Time Consumed: ' + str(time_c1) + 
    's\n best value: ' + str(optimizer.bestvalues[-1]))

    ax2 = plt.subplot(2, 1, 2)
    x2 = range(iterations)
    y2 = list()
    for i in x2 :
        y2.append(footrule(optimizer.besttrace[-1], optimizer.besttrace[i]))
    ax2.plot(x2, y2, label = 'distance')
    ax2.set_xlabel('iterations')
    ax2.legend()

    plt.show()









        

        



    



        

        

