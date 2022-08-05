from math import log
import random
import numpy as np
import time
import csv
import os

w = 5000000
g0 = -40
theta = 4
d0 = 1
d = 100
N0 = -174
FRQ = 1e9
power = 70

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
            while(i < dim) :
                curline = lines[i].split()
                self.tasklist.append(Task(int(curline[0]), int(curline[1])))  
                i += 1


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
       
                else :
                    self.bestvalues.append(self.bestvalues[-1])
              
            else :
                self.bestvalues.append(self.wolves[0].fit)
          



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


iters = [200, 400, 600, 800, 1000, 1200, 1400, 1600]


if __name__ == '__main__' :
    for sizes in range(3, 11) :
        dir = './TestingInstances3/' + str(sizes * 10) 
        for No in range(0, 10) :
            url = dir + '/' + str(sizes * 10) + '_' + str(No) + '.txt'
            for i in range(5) :
                ins = instance(url)
                
                tasks = ins.tasklist
                iterations = iters[sizes - 3]
                population = 100
                optimizer1 = d_gwo(tasks, population, iterations)
                time_start = time.time() 
                optimizer1.optimize()
                time_end = time.time()
                time_c1= time_end - time_start
                print(url + ' ' +str(i) +' ' + str(time_c1))
                y1 = optimizer1.bestvalues
                data = ['D_GWO', sizes * 10, No, iterations, population, y1[-1], time_c1]
                with open('./performance/' + str(sizes * 10) + '.csv', 'a+') as f:
                    writer = csv.writer(f)
                    if os.path.getsize('./performance/' + str(sizes * 10) + '.csv') == 0:
                        writer.writerow(['Algorithm', 'Scale', 'Instance_no', 'Max_iter', 'Population', 'Target', 'Time'])
                    writer.writerow(data)


