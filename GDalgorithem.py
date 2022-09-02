import numpy
import logging
from settings import Setting
from objective_function import theano_expression

# Revised the algorithm so that it can be applied when P is a matrix
# verifyGDvalue = []


def extract(vec, index):
    if type(vec).__module__ == numpy.__name__:
        temp = vec
    else:
        temp = numpy.array(vec)
    size = temp.shape
    if len(size) == 1:  # vec is a vector
        return temp[index]
    else:  # vec is a matrix
        return temp[:, index]


# Update the best solution
def updateGdbest(currentBest, newValue):
    if currentBest is None:
        return newValue
    elif currentBest > newValue:
        return newValue
    else:
        return currentBest


def Batch_RMSprop(func, setting, candidate):
    batch_size = 5
    fitness = []
    for j in range(batch_size):
        fitness.append(RMSprop(func, setting, candidate))
    return sum(fitness) * 1.0 / batch_size


# Gradient descent using RMSprop
def RMSprop(func, setting, candidate):
    learningRateMin = 0.01 #0.0000001  # 0.000001
    learningRateMax = 0.06 #0.0000007  #0.00001
    iterationNum = 10  #30
    gdbest = None

    # initializing the parameters and make sure it sums to 1
    nonzeroIndex = numpy.nonzero(candidate)[0]
    a = extract(setting.capacity, nonzeroIndex)
    D = extract(setting.latency, nonzeroIndex)
    beta = extract(setting.decayFactor, nonzeroIndex)
    paraLen = len(nonzeroIndex)  # the number of selected controllers
    # print("extracted capacity: %s\nextracted latency: %s\nextracted decay_factor: %s \n" % (a, D, beta))
    # print("selected capacity: %s, arrival_rate: %s" % (sum(a), sum(setting.arrivalRate)))
    if paraLen == 1:  # only one controller is selected
        para = [[]]
        cost = func(setting, para, a, D, beta, False)
        logging.warning("individual %s" % candidate)
        # print "Complete iteration cost: %s, probability: 1" % cost
        return cost

    # temp = numpy.random.random(paraLen)
    temp = list(a)  # initialize the probability based on the capacity
    para_row = [float(element) / sum(temp) for element in temp]
    sw_num, _ = D.shape

    para = [0]*sw_num
    for i in range(sw_num):
        para[i] = list(para_row) # para = [para_row]* sw_num # this is a shallow copy, change one element will change all elements with the same index
    para = numpy.array(para)
    para = para[:, :-1]
    # print("initialized probability %s" % para[:2])

    # print "start gradient evaluation"
    cost = func(setting, para, a, D, beta, False)
    gdbest = updateGdbest(gdbest, cost)
    # verifyGDvalue.append(cost)

    g = func(setting, para, a, D, beta, True)
    paraOld = para

    for i in range(iterationNum):
        g = numpy.array(g)
        learningRate = learningRateMax - (i - 1) * (learningRateMax - learningRateMin) / iterationNum
        para = paraOld - learningRate * g  # update solution

        for rownum in range(sw_num):
            para[0][rownum] = para[0][rownum].clip(0)
            while sum(para[0][rownum]) > 1:
                diff = sum(para[0][rownum])-1
                weight = para[0][rownum]
                weightedDiff = [diff * element / sum(weight) for element in weight]
                para[0][rownum] -= weightedDiff

        para = numpy.ndarray.tolist(para)[0]

        cost = func(setting, para, a, D, beta, False)

        # verifyGDvalue.append(cost)
        # print i, cost

        gdbest = updateGdbest(gdbest, cost)
        g = func(setting, para, a, D, beta, True)
        paraOld = para
        
    return gdbest


def main():
    
    latency_matrix = [[0, 2.8300775173781934, 1.3895379545709188, 2.9586777153181227, 3.3631162017063057], [2.8300775173781934, 0, 1.4405395628072746, 3.0096793235544785, 0.5330386843281122], [1.3895379545709188, 1.4405395628072746, 0, 1.5691397607472042, 1.9735782471353867], [2.9586777153181227, 0.5330386843281122, 1.5691397607472042, 0, 0.0], [3.3631162017063057, 0.5330386843281122, 1.9735782471353867, 0.0, 0]]
    
    setting = Setting(latency_matrix)
    print(RMSprop(theano_expression, setting, [1] * setting.ctlNum))


if __name__ == '__main__' :
    main()


