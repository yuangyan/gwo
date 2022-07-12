from math import log
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging

w = 5000000
g0 = -40
theta = 4
d0 = 1
d = 100
N0 = -174
FRQ = 1e9
power = 5

current_time = str(datetime.datetime.now()).replace(' ', '-')
current_time = current_time.replace(':', '-')
logger = logging.getLogger('test_logger')
logger.setLevel(logging.INFO)
test_log = logging.FileHandler(current_time + '.log', 'a', encoding='utf-8')
test_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s ')
test_log.setFormatter(formatter)
logger.addHandler(test_log)

class Task :
    def __init__(self, D, C) : #D: data size C: cpu cycles
        self.datasize = D
        self.cpucycles = C
    
    def TransmissionTime(self, p) :
        return self.datasize / TransmissionRate(p) 
        
    def Disposetime(self) :
        return self.datasize * self.cpucycles / FRQ 

class wolf :
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
        C1, C2, C3 = 2 * rnd.random(), 2*rnd.random(), 2*rnd.random()

        D1 = abs(C1 * alpha_position - self.position)
        D2 = abs(C2 * beta_position - self.position)
        D3 = abs(C3 * sigma_position - self.position)

        X1 = alpha_position - A1 * D1
        X2 = beta_position - A2 * D2
        X3 = sigma_position - A3 * D3

        self.position = (X1 + X2 + X3) / 3.0
        # greedy selection?
        self.fit = self.fitness()

class gwo :
    def __init__(self, tasklist, minx, maxx, population, max_iter) :
        self.wolves = list()
        self.tasklist = tasklist
        self.dim = len(tasklist)
        self.max_iter = max_iter
        self.bestvalues = list()

        for i in range(population) :
            self.wolves.append(wolf(tasklist, minx, maxx))
    
    def optimize(self) :
        self.bestvalues.clear()

        for iter in range(self.max_iter) :
            print("iterations=" + str(iter))
            logger.info("iterations=" + str(iter))
            a = 2 * (1 - iter / self.max_iter)

            self.wolves = sorted(self.wolves, key = lambda item : item.fit)
            self.bestvalues.append(self.wolves[0].fit)

            alpha_position = self.wolves[0].position
            beta_position = self.wolves[1].position
            sigma_position = self.wolves[2].position
            
            print("alpha fitness: " + str(self.wolves[0].fit))
            logger.info("alpha fitness: " + str(self.wolves[0].fit))

            wolfno = 0
            for individual in self.wolves :
                print("Wolf No." + str(wolfno))
                logger.info("Wolf No." + str(wolfno))
                print("current fitness: " + str(individual.fit))
                logger.info("current fitness: " + str(individual.fit))
                individual.update(alpha_position, beta_position, sigma_position, a)
                print("new fitness: " + str(individual.fit))
                logger.info("new fitness: " + str(individual.fit))

                wolfno += 1

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


if __name__ == '__main__' :
    ins = instance('./TestingInstances/100/100_1.txt')
    tasks = ins.tasklist
    iterations = 20
    optimizer1 = gwo(tasks, 0, 4, 200, iterations)
    optimizer1.optimize()
    x = range(iterations)
    y = optimizer1.bestvalues
    s = "bestvalue: " + str(y[-1])
    print("bestvalue: " + str(y[-1]))
    logger.info("bestvalue: " + str(y[-1]))
    plt.plot(x, y)
    plt.show()
    