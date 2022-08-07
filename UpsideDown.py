import os
import csv
from math import log
import time
import random

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


class instance :
    def __init__(self, dir) :
        self.tasklist = list()
        with open(dir) as f :
            #datasize,cycle
            lines=f.readlines()
            dim = int(lines[0])
            i = 1
            while(i < dim + 1) :
                curline = lines[i].split()
                self.tasklist.append(Task(int(curline[0]), int(curline[1])))  
                i += 1

if __name__ == '__main__' :
    for sizes in range(10, 11) :
        dir = './TestingInstances3/' + str(sizes * 10) 
        for No in range(0, 1) :
            url = dir + '/' + str(sizes * 10) + '_' + str(No) + '.txt'
            for i in range(1) :
                ins = instance(url)
                tasks = ins.tasklist
                time_start = time.time() 

                N = len(tasks)
                seq = [N - 1 - i for i in range(N)]
                random.shuffle(seq)
                res = makespan(seq, tasks)

                print(res)