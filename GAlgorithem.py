import random
from deap import creator, base, tools
import multiprocessing
import logging
# import matplotlib.pyplot as plt
from objective_function import theano_expression
from settings import Setting
from GDalgorithem import RMSprop

latency_matrix = [[0, 2.8300775173781934, 1.3895379545709188, 2.9586777153181227, 3.3631162017063057], [2.8300775173781934, 0, 1.4405395628072746, 3.0096793235544785, 0.5330386843281122], [1.3895379545709188, 1.4405395628072746, 0, 1.5691397607472042, 1.9735782471353867], [2.9586777153181227, 0.5330386843281122, 1.5691397607472042, 0, 0.0], [3.3631162017063057, 0.5330386843281122, 1.9735782471353867, 0.0, 0]]
setting = Setting(latency_matrix)
IND_SIZE = setting.ctlNum
POP_SIZE = IND_SIZE * 10# 820
# CXPB  is the probability with which two individuals are crossed
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 1, 0.1


def evalMin(individual):
    logging.info("Evaulating %s " % individual)
    if max(individual) == 0:
        return 100000000,
    temp = [a*b for a,b in zip(setting.capacity, individual)]
    # if sum(temp) < setting.arrivalRate:
    # if sum(temp) * setting.decayFactor[0] < sum(setting.arrivalRate) - setting.beta * sum(individual) * sum(individual):  # n
    if sum(temp) * setting.decayFactor[0] < sum(setting.arrivalRate) - setting.beta*sum(individual)*sum(individual)*sum(individual):  # n^2
    # if sum(temp) * setting.decayFactor[0] < sum(setting.arrivalRate) - setting.beta * math.log(sum(individual)) * sum(individual):  # log(n)
        return 100000000,
    fitness = RMSprop(theano_expression, setting, individual)
    return fitness[0],

# ----------
# Operator registration
# ----------
# register the goal / fitness function


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of IND_SIZE 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, IND_SIZE)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalMin)

# register the crossover operator
toolbox.register("mate", tools.cxOnePoint)

# register a mutation operator with a probability to flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=1)

# operator for selecting individuals for breeding the next generation:
toolbox.register("selectParent", tools.selRandom)

toolbox.register("selectGeneration", tools.selBest)


def main():
    # random.seed(103)

    pool = multiprocessing.Pool(1)

    toolbox.register("map", pool.map)
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=POP_SIZE)

    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))

    # Variable keeping track of the number of generations
    g = 0
    elitist_ind = []
    elitist_fit = []


    # Begin the evolution
    while g < 1000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next len(pop) generation individuals
        offspring = toolbox.selectParent(pop, POP_SIZE)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced
        pop[:] = toolbox.selectGeneration(pop + offspring, POP_SIZE)

        print( "New gener has %s " % (len(pop)))
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        best_ind = tools.selBest(pop, 1)[0]
        elitist_ind.append(best_ind)
        elitist_fit.append(best_ind.fitness.values)
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(elitist_ind)
    # plt.subplot(212)
    # plt.plot(elitist_fit)
    # plt.show()


if __name__ == "__main__":
    main()