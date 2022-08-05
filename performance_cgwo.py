from math import log
import random
import numpy as np
import csv
import time
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
            while(i < dim - 1) :
                curline = lines[i].split()
                self.tasklist.append(Task(int(curline[0]), int(curline[1])))  
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
        self.bestvalues = list()

        for i in range(population) :
            self.wolves.append(c_wolf(tasklist, minx, maxx))
    
    def optimize(self) :
        self.bestvalues.clear()

        for iter in range(self.max_iter) :
            a = 2 * (1 - iter / self.max_iter)

            self.wolves = sorted(self.wolves, key = lambda item : item.fit)
            self.bestvalues.append(self.wolves[0].fit)

            alpha_position = self.wolves[0].position
            beta_position = self.wolves[1].position
            sigma_position = self.wolves[2].position
            
            for individual in self.wolves :
                individual.update(alpha_position, beta_position, sigma_position, a)


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
                optimizer1 = c_gwo(tasks, 0, 4, population, iterations)
                time_start = time.time() 
                optimizer1.optimize()
                time_end = time.time()
                time_c1= time_end - time_start
                y1 = optimizer1.bestvalues
                print(url + ' ' +str(i) +' ' + str(time_c1))
    
                data = ['C_GWO', sizes * 10, No, iterations, population, y1[-1], time_c1]
                with open('./performance/' + str(sizes * 10) + '.csv', 'a+') as f:
                    writer = csv.writer(f)
                    if os.path.getsize('./performance/' + str(sizes * 10) + '.csv') == 0:
                        writer.writerow(['Algorithm', 'Scale', 'Instance_no', 'Max_iter', 'Population', 'Target', 'Time'])
                    writer.writerow(data)
