from math import log
import random
import matplotlib.pyplot as plt

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

class wolf :
    def __init__(self, tasklist) :
        self.tasklist = tasklist
        self.dim = len(tasklist)
        self.seq = [i for i in range(self.dim)]
        random.shuffle(self.seq)
        self.fit = self.fitness()

    def fitness(self) :
        return makespan(self.seq, self.tasklist)

    def update(self, alpha_seq, beta_seq, sigma_seq, a) :
        pass
        
class gwo :
    def __init__(self, tasklist, population, max_iter) :
        self.wolves = list()
        self.tasklist = tasklist
        self.dim = len(tasklist)
        self.max_iter = max_iter
        self.bestvalues = list()

        for i in range(population) :
            self.wolves.append(wolf(tasklist))

    def optimize(self) :
        self.bestvalues.clear()

        for iter in range(self.max_iter) :
            a = 2 * (1 - iter / self.max_iter)
            self.wolves = sorted(self.wolves, key = lambda item : item.fit)
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