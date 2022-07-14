from math import log
import random
import matplotlib.pyplot as plt
import numpy as np

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
        dist = self.dim / 2 * a
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
        self.bestvalues = list()

        for i in range(population) :
            self.wolves.append(d_wolf(tasklist))

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


if __name__ == '__main__' :
    ins = instance('./TestingInstances/100/100_1.txt')
    tasks = ins.tasklist
    iterations = 50
    optimizer1 = d_gwo(tasks, 50, iterations)
    optimizer1.optimize()
    x = range(iterations)
    y = optimizer1.bestvalues
    s = "bestvalue: " + str(y[-1])
    print("bestvalue: " + str(y[-1]))
    plt.plot(x, y)
    plt.show()
       