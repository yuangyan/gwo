import random
import matplotlib.pyplot as plt

import time
import datetime
import logging

current_time = str(datetime.datetime.now()).replace(' ', '-')
current_time = current_time.replace(':', '-')
logger = logging.getLogger('test_logger')
logger.setLevel(logging.INFO)
test_log = logging.FileHandler(current_time + '.log', 'a', encoding='utf-8')
test_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s ')
test_log.setFormatter(formatter)
logger.addHandler(test_log)

def footrule(a, b) :
    distance = 0
    for i in range(len(a)) :

        if a[i] != b[i] :
            distance += 1
    return distance

def genseq(seq, distance) :
    newseq = [i for i in seq]
    length = len(seq)
    if distance > length :
        distance = length

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


    for i in range(distance - 1) :
        curindex = distance - 1 - i
        swapindex = random.randrange(0, curindex)
        tempval = que[curindex]
        que[curindex] = que[swapindex]
        que[swapindex] = tempval

    for i in range(distance) :
        newseq[pickedindex[i]] = que[i]

    return newseq

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

def randseq(seq, distance) :
    newseq = [i for i in seq]
    length = len(seq)
    if distance > length :
        distance = length
    
    if distance < 0 :
        distance = 0

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
                
            # logger.info('curindex' + str(curindex))
            # logger.info('j is marked' + str(swapindex))

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

length = 100
a = [i for i in range(length)]
# distance = random.randint(0, length)
distance = 99
# logger.info(distance)
print(distance)
random.shuffle(a)


time_start = time.time() 
b = randseq(a, distance)
time_end = time.time()
time_c= time_end - time_start

print('randseq' + str(time_c))

# time_start = time.time() 
# b = genseq(a, distance)
# time_end = time.time()
# time_c= time_end - time_start
# print('sattolo' + str(time_c))

print(footrule(a, b))


        

    
