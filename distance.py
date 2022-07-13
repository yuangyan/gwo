def footrule(a, b) :
    distance = 0
    for i in range(len(a)) :
        if a[i] != b[i] :
            distance += 1
    return distance

def genseq(seq, distance) :
    if distance > len(seq) :
        distance = len(seq)
    
