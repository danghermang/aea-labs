import os
import sys
import json
import time
import random

import numpy as np
from deap import creator
from deap import base
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def read_data():
    global N_CITIES, CITIES_MATRIX
    N_CITIES = None
    CITIES_MATRIX = None
    if len(sys.argv) == 1:
        print('Specifiy an input file.\ne.g.: %s inputs/example.json' % os.path.basename(sys.argv[0]))
        exit(1)

    input_fp = sys.argv[1]
    with open(input_fp, 'rt') as f:
        input_data = json.load(f)
    N_CITIES = input_data["n"]
    CITIES_MATRIX = np.array(input_data['matrix'])


def tspd_crossover(ind1, ind2):
    tools.cxPartialyMatched(ind1[0], ind2[0])
    tools.cxTwoPoint(ind1[1], ind2[1])
    return ind1, ind2


def tspd_mutation(ind, indpb_cities, indpb_drone_visiting):
    tools.mutShuffleIndexes(ind[0], indpb_cities)
    tools.mutFlipBit(ind[1], indpb_drone_visiting)
    return ind,


def tspd_evaluate(ind):
    is_drone_at_a_customer = False
    last_city_visited_by_truck = 0
    last_city_visited_by_drone = 0
    truck_distance = 0
    drone_distance = 0

    # print('cities:', [city + 1 for city in ind[0]])
    # i = 0

    total_time_taken = 0
    for city, visited_by_drone in zip(ind[0], ind[1]):
        city += 1
        # print('i:', i, ', city:', city)
        # i += 1

        if not visited_by_drone:
            if is_drone_at_a_customer:
                truck_distance += CITIES_MATRIX[last_city_visited_by_truck, city]
                last_city_visited_by_truck = city
            else:
                total_time_taken += CITIES_MATRIX[last_city_visited_by_truck, city]
                last_city_visited_by_truck = city
                # print('totaltime:', total_time_taken)
                # print()
        else:
            if not is_drone_at_a_customer:
                is_drone_at_a_customer = True
                drone_distance += CITIES_MATRIX[last_city_visited_by_truck, city]
                last_city_visited_by_drone = city
            else:
                is_drone_at_a_customer = False
                drone_distance += CITIES_MATRIX[last_city_visited_by_drone, city]
                truck_distance += CITIES_MATRIX[last_city_visited_by_truck, city]
                last_city_visited_by_truck = city
                last_city_visited_by_drone = city
                total_time_taken += max(drone_distance, truck_distance)
                # print('totaltime:', total_time_taken)
                # print()
                drone_distance = 0
                truck_distance = 0

    if is_drone_at_a_customer:
        drone_distance += CITIES_MATRIX[last_city_visited_by_drone, 0]
        truck_distance += CITIES_MATRIX[last_city_visited_by_truck, 0]
        total_time_taken += max(drone_distance, truck_distance)
        # print('totaltime:', total_time_taken)

    # print('finaltotaltime:', total_time_taken)
    # exit(1)

    return total_time_taken,


def main():
    global POP_SIZE
    read_data()
    print('data finished to read')

    POP_SIZE = 200
    N_GENERATIONS = N_CITIES * 20
    js = 19  # statistics justify size: used for space padding

    creator.create("FitnessTSPD", base.Fitness, weights=(-1.0,))
    creator.create("Individual", tuple, fitness=creator.FitnessTSPD)

    toolbox = base.Toolbox()
    # individual and population
    toolbox.register("aux_indices", random.sample, range(0, N_CITIES - 1), N_CITIES - 1)
    toolbox.register("attr_cities", tools.initIterate, list, toolbox.aux_indices)

    toolbox.register("aux_bool", lambda x, y: 0 if random.randint(x, y) <= 50 else 1, 1, 100)
    toolbox.register("attr_drone_visiting", tools.initRepeat, list, toolbox.aux_bool, n=N_CITIES - 1)

    toolbox.register("individual",
                     tools.initCycle,
                     creator.Individual,
                     (toolbox.attr_cities, toolbox.attr_drone_visiting),
                     n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # operators
    toolbox.register("evaluate", tspd_evaluate)
    toolbox.register("mate", tspd_crossover)
    toolbox.register("mutate", tspd_mutation, indpb_cities=0.01, indpb_drone_visiting=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # run the algorithm
    start = time.time()
    population = toolbox.population(n=POP_SIZE)
    print('%s %s %s %s %s' % ('gen'.ljust(js), 'avg'.ljust(js), 'std'.ljust(js), 'min'.ljust(js), 'max'.ljust(js)))
    bests = []
    best = np.sum(CITIES_MATRIX)
    for gen in range(N_GENERATIONS):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.2, mutpb=0.05)
        fits = toolbox.map(toolbox.evaluate, offspring)
        fits_list = []
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
            fits_list.append(fit)
        population = toolbox.select(offspring, k=POP_SIZE)

        # print statistics
        pavg = np.mean(fits_list)
        pstd = np.std(fits_list)
        pmin = np.min(fits_list)
        pmax = np.max(fits_list)
        print('%s %s %s %s %s' % (str(gen).ljust(js),
                                  str(pavg).ljust(js),
                                  str(pstd).ljust(js),
                                  str(pmin).ljust(js),
                                  str(pmax).ljust(js)))
        bests.append(pmin)
        if bests[-1] < best:
            best = bests[-1]
            best_idx = gen
            best_ind = offspring[np.argmin(fits_list)]
    print('%s %s %s %s %s' % ('gen'.ljust(js), 'avg'.ljust(js), 'std'.ljust(js), 'min'.ljust(js), 'max'.ljust(js)))

    # plot statistics
    end = time.time()
    run_time = end - start
    print('Best solution: c=%s, v=%s' % (best, best_ind))
    print('Time to finish run: %ss. Average time per generation: %ss' % (str(run_time), str(run_time / N_GENERATIONS)))
    plt.plot(range(N_GENERATIONS), bests, 'bo', [best_idx], [best], 'r+')
    plt.legend(('Best for each generation', 'Global best'), shadow=True)
    plt.xlabel('Generation')
    plt.ylabel('Time taken')
    plt.show()


if __name__ == '__main__':
    main()
