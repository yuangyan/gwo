from math import log
import random
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


def footrule(a, b) :
    distance = 0
    for i in range(len(a)) :
        if a[i] != b[i] :
            distance += 1
    return distance

def order1(chromosome1, chromosome2) :
    N = len(chromosome1)
    child = list()
    idx0 = random.randrange(0, N)
    idx1 = random.randrange(0, N)
    
    lo, hi = 0, 0
    if idx0 < idx1 :
        lo = idx0
        hi = idx1
    else :
        lo = idx1
        hi = idx0

    fragment = chromosome1[lo : hi + 1]

    for i in range(N) :
        if chromosome2[i] not in fragment :
            child.append(chromosome2[i])

    insertion = random.randint(0, len(child))
    child = child[:insertion] + fragment + child[insertion:]
    return child


def PMX(chromosome1, chromosome2) :
    N = len(chromosome1)
    child = chromosome2.copy()
    idx0 = random.randrange(0, N)
    idx1 = random.randrange(0, N)
    
    lo, hi = 0, 0
    if idx0 < idx1 :
        lo = idx0
        hi = idx1
    else :
        lo = idx1
        hi = idx0
    
    child[lo : hi + 1] = chromosome1[lo : hi + 1]
    for i in range(hi - lo + 1) :
        if chromosome2[lo + i] not in child[lo : hi + 1] :
            value = chromosome1[lo + i]
            position = chromosome2.index(value)
            while(position >= lo and position <= hi) :
                value = chromosome1[position]
                position = chromosome2.index(value)
            child[position] = chromosome2[lo + i]
    return child


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

class individual :
    def __init__(self, tasklist) :
        self.tasklist = tasklist
        self.dim = len(tasklist)
        self.chromosome = [i for i in range(self.dim)]
        random.shuffle(self.chromosome)
        self.fit = self.fitness()

    def fitness(self) :
        return makespan(self.chromosome, self.tasklist)

    def setchromosome(self, chromosome) :
        self.chromosome = chromosome
        self.fit = self.fitness()

class ga :
    def __init__(self, tasklist, population, max_iter) :
        self.individuals = list()
        self.tasklist = tasklist
        self.dim = len(tasklist)
        self.max_iter = max_iter
        self.currentbest = list()
        self.population = population
        self.bestvalues = list()
        self.besttrace = list()

        for i in range(population) :
            self.individuals.append(individual(tasklist))
    
    def optimize(self) :
        self.currentbest.clear()
        self.bestvalues.clear()
        self.besttrace.clear()

        for iter in range(self.max_iter) :
            nextgen = list()
            # Roulette Wheel selection, retain 10% best individuals
            self.individuals = sorted(self.individuals, key = lambda item : item.fit)
            self.currentbest.append(self.individuals[0].fit)

            if self.bestvalues != [] :
                if self.individuals[0].fit < self.bestvalues[-1] :
                    self.bestvalues.append(self.individuals[0].fit)
              
                else :
                    self.bestvalues.append(self.bestvalues[-1])
               
            else :
                self.bestvalues.append(self.individuals[0].fit)
               

            for i in range(int(self.population * 0.1)) :
                nextgen.append(self.individuals[i])
            
            while(len(nextgen) < self.population) :
                parent1 = random.randrange(0, self.population)
                parent2 = parent1
                while(parent2 == parent1) :
                    parent2 = random.randrange(0, self.population)
                
                childchromosome = order1(self.individuals[parent1].chromosome, 
                                    self.individuals[parent2].chromosome)

                child = individual(self.tasklist)
                child.setchromosome(childchromosome)
                nextgen.append(child)
            self.individuals = nextgen

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
                optimizer1 = ga(tasks, population, iterations)
                time_start = time.time() 
                optimizer1.optimize()
                time_end = time.time()
                time_c1= time_end - time_start
                y1 = optimizer1.bestvalues
                print(url + ' ' +str(i) +' ' + str(time_c1))
    
                data = ['GA', sizes * 10, No, iterations, population, y1[-1], time_c1]
                with open('./performance/' + str(sizes * 10) + '.csv', 'a+') as f:
                    writer = csv.writer(f)
                    if os.path.getsize('./performance/' + str(sizes * 10) + '.csv') == 0:
                        writer.writerow(['Algorithm', 'Scale', 'Instance_no', 'Max_iter', 'Population', 'Target', 'Time'])
                    writer.writerow(data)
