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

class c_wolf :
    def __init__(self, tasklist, minx, maxx) :
        # 有无静态成员变量
        self.tasklist = tasklist
        self.rnd = random.Random()
        self.dim = len(tasklist)
        self.position = [0.0] * self.dim

        for i in range(self.dim):
            self.position[i] = (maxx - minx) * self.rnd.random()
        self.position = np.array(self.position)
        # fitness
        self.fit = self.fitness()

    def fitness(self) :
        pairs = list()
        for i in range(self.dim) :
            pairs.append((i, self.position[i]))
        pairs = sorted(pairs, key = lambda item : item[1])

        seq = list()
        for i in range(self.dim) :
            seq.append(pairs[i][0])

        return makespan(seq, self.tasklist)

    def update(self, alpha_position, beta_position, sigma_position, a) :
        rnd = random.Random()
        A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
              2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
        C1, C2, C3 = rnd.random(), rnd.random(), rnd.random()

        D1 = abs(C1 * alpha_position - self.position)
        D2 = abs(C2 * beta_position - self.position)
        D3 = abs(C3 * sigma_position - self.position)

        X1 = alpha_position - A1 * D1
        X2 = beta_position - A2 * D2
        X3 = sigma_position - A3 * D3

        self.position = (X1 + X2 + X3) / 3.0
        # greedy selection?
        self.fit = self.fitness()


class c_gwo :
    def __init__(self, tasklist, minx, maxx, population, max_iter) :
        self.wolves = list()
        self.tasklist = tasklist
        self.dim = len(tasklist)
        self.max_iter = max_iter
        self.alphavalues = list()

        for i in range(population) :
            self.wolves.append(c_wolf(tasklist, minx, maxx))
    
    def optimize(self) :
        self.alphavalues.clear()

        for iter in range(self.max_iter) :
            a = 2 * (1 - iter / self.max_iter)

            self.wolves = sorted(self.wolves, key = lambda item : item.fit)
            self.alphavalues.append(self.wolves[0].fit)

            alpha_position = self.wolves[0].position
            beta_position = self.wolves[1].position
            sigma_position = self.wolves[2].position
            
            for individual in self.wolves :
                individual.update(alpha_position, beta_position, sigma_position, a)


class d_wolf :
    def __init__(self, tasklist) :
        self.tasklist = tasklist
        self.dim = len(tasklist)
        self.seq = [i for i in range(self.dim)]
        random.shuffle(self.seq)
        self.fit = self.fitness()

    def fitness(self) :
        return makespan(self.seq, self.tasklist)

    def update(self, alpha_seq, beta_seq, sigma_seq, a) :
        dist = self.dim * a
        dist = round(dist + np.random.normal(loc=0.0, scale=2, size=None))
        ca = random.uniform(0, 1)
        if ca < 1 / 3 :   
            self.seq = randseq(alpha_seq, dist)

        elif ca < 2 / 3 :
            self.seq = randseq(beta_seq, dist)        

        else :
            self.seq = randseq(sigma_seq, dist)

        self.fit = self.fitness()
            
            
        
class d_gwo :
    def __init__(self, tasklist, population, max_iter) :
        self.wolves = list()
        self.tasklist = tasklist
        self.dim = len(tasklist)
        self.max_iter = max_iter
        self.alphavalues = list()
        self.bestvalues = list()
        self.besttrace = list()
        

        for i in range(population) :
            self.wolves.append(d_wolf(tasklist))

    def optimize(self) :
        self.alphavalues.clear()
        self.besttrace.clear()
        self.bestvalues.clear()

        for iter in range(self.max_iter) :
            print(iter)
            x = iter / self.max_iter
            a = updatefunction(x)

            # # sigmoid
            # x = 10 * iter / self.max_iter
            # a = 0.5 / (1 + exp(x - 5))

            self.wolves = sorted(self.wolves, key = lambda item : item.fit)
            self.alphavalues.append(self.wolves[0].fit)
            if self.bestvalues != [] :
                if self.wolves[0].fit < self.bestvalues[-1] :
                    self.bestvalues.append(self.wolves[0].fit)
                    self.besttrace.append(self.wolves[0].seq)
                else :
                    self.bestvalues.append(self.bestvalues[-1])
                    self.besttrace.append(self.besttrace[-1])
            else :
                self.bestvalues.append(self.wolves[0].fit)
                self.besttrace.append(self.wolves[0].seq)



            alpha_seq = self.wolves[0].seq
            beta_seq = self.wolves[1].seq
            sigma_seq = self.wolves[2].seq

            for individual in self.wolves :
                individual.update(alpha_seq, beta_seq, sigma_seq, a)


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

def Dn(n) :
    d_n = - (n % 2) + (n + 1) % 2
    sum = 0
    u = n
    d_u = d_n
    while(u > 0) :
        sum += d_u
        d_u = - d_u * u
        u -= 1
    sum += d_u

    return sum

def Prob1(u) :
    return (u - 1) * Dn(u - 2) / Dn(u) 

def Prob2(u) :
    return 1 / u

def Prob(u) :
    if(u > 10) :
        return Prob2(u)
    else :
        return Prob1(u)


def footrule(a, b) :
    distance = 0
    for i in range(len(a)) :
        if a[i] != b[i] :
            distance += 1
    return distance

def randseq(seq, distance) :
    newseq = [i for i in seq]
    length = len(seq)
    if distance > length :
        distance = length

    if distance <= 0 :
        distance = 1
    
    distance = int(distance)

    indexes = [i for i in range(length)]
    pickedindex = list()
    que = list()

    for i in range(distance) :
        curindex = length - 1 - i
        swapindex = random.randint(0, curindex)
        tempval = indexes[curindex]
        indexes[curindex] = indexes[swapindex]
        indexes[swapindex] = tempval
        pickedindex.append(indexes[curindex])
        que.append(seq[indexes[curindex]])

    marked = [0] * distance
    for i in range(distance - 1) :
        curindex = distance - 1 - i
        if marked[curindex] == 1 :
            continue
        
        while(True) :
            swapindex = random.randrange(0, curindex)
            if marked[swapindex] == 0:
                break

            firstavailable = distance
            for i in range(distance) :
                if marked[i] == 0 :
                    firstavailable = i

            if firstavailable >= curindex :
                break
            
        tempval = que[curindex]
        que[curindex] = que[swapindex]
        que[swapindex] = tempval

        p = random.uniform(0, 1)
        if p < Prob(curindex + 1) :
            marked[swapindex] = 1

    for i in range(distance) :
        newseq[pickedindex[i]] = que[i]

    return newseq

def updatefunction(x) :
    return 0.5 * (1 - x) * (1 + 0.3 * np.sin(25 * x))

if __name__ == '__main__' :
    ins = instance('./TestingInstances/100/100_1.txt')
    tasks = ins.tasklist

    iterations = 1500
    population = 100

    optimizer1 = d_gwo(tasks, population, iterations)
    time_start = time.time() 
    optimizer1.optimize()
    time_end = time.time()
    time_c1= time_end - time_start
    x1 = range(iterations)
    y1 = optimizer1.bestvalues


    fig = plt.figure()
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(x1, y1, label = 'best_value')
    ax1.set_xlabel('iterations')
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    ax1.legend()
    
    ax1.set_title('Discrete GWO\n Time Consumed: ' + str(time_c1) + 
    's\n best value: ' + str(optimizer1.alphavalues[-1]))

    ax2 = plt.subplot(3, 1, 2)
    x2 = range(iterations)
    y2 = list()
    for i in x2 :
        y2.append(footrule(optimizer1.besttrace[-1], optimizer1.besttrace[i]))
    ax2.plot(x2, y2, label = 'distance')
    ax2.set_xlabel('iterations')

    ax2.legend()

    ax3 = plt.subplot(3, 1, 3)
    x3 = np.linspace(0, iterations)
    y3 = updatefunction(x3 / iterations)
    ax3.plot(x3, y3)
    ax3.set_xlabel('iterations')

    plt.show()