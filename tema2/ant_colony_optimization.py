import os
import sys
import json
import random
from copy import deepcopy
import numpy as np


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


def dist(a, b):
    dist = CITIES_MATRIX[a, b]
    if dist < 1e-10:
        return 1e-10
    return dist


def distance(last_city_visited_by_truck, city, last_drone_visit, drone_visit, drone_distance, truck_distance,
             last_city_visited_by_drone, cities_involved):
    # cities_involved = 1
    if not last_drone_visit and not drone_visit:
        return (max(drone_distance, truck_distance) + dist(last_city_visited_by_truck, city)) / cities_involved
    if not last_drone_visit and drone_visit:
        return max(drone_distance + dist(last_city_visited_by_drone, city), truck_distance) / cities_involved
    if last_drone_visit and not drone_visit:
        return max(drone_distance, truck_distance + dist(last_city_visited_by_truck, city)) / cities_involved
    if last_drone_visit and drone_visit:
        return max(drone_distance + dist(last_city_visited_by_drone, city),
                   truck_distance + dist(last_city_visited_by_truck, city)) / cities_involved


def cost(permutation, drone_visits):
    is_drone_at_a_customer = False
    last_city_visited_by_truck = 0
    last_city_visited_by_drone = 0
    truck_distance = 0
    drone_distance = 0

    # print('cities:', [city + 1 for city in permutation])
    # i = 0

    total_time_taken = 0
    for city, visited_by_drone in zip(permutation, drone_visits):
        # print('i:', i, ', city:', city)
        # i += 1

        if not visited_by_drone:
            if is_drone_at_a_customer:
                truck_distance += dist(last_city_visited_by_truck, city)
                last_city_visited_by_truck = city
            else:
                total_time_taken += dist(last_city_visited_by_truck, city)
                # print('totaltime:', total_time_taken)
                # print()
                last_city_visited_by_truck = city
        else:
            if not is_drone_at_a_customer:
                is_drone_at_a_customer = True
                drone_distance += dist(last_city_visited_by_truck, city)
                last_city_visited_by_drone = city
            else:
                is_drone_at_a_customer = False
                drone_distance += dist(last_city_visited_by_drone, city)
                truck_distance += dist(last_city_visited_by_truck, city)
                last_city_visited_by_truck = city
                last_city_visited_by_drone = city
                total_time_taken += max(drone_distance, truck_distance)
                # print('totaltime:', total_time_taken)
                # print()
                drone_distance = 0
                truck_distance = 0

    if is_drone_at_a_customer:
        drone_distance += dist(last_city_visited_by_drone, 0)
        truck_distance += dist(last_city_visited_by_truck, 0)
        total_time_taken += max(drone_distance, truck_distance)
        # print('totaltime:', total_time_taken)

    # print('finaltotaltime:', total_time_taken)
    # exit(1)

    return total_time_taken


def random_permutation(n):
    cities = list(range(1, n))
    random.shuffle(cities)
    cities = [0] + cities
    return cities


def random_bool_list(n):
    return [0 if random.randint(1, 100) <= 50 else 1 for i in range(1, n)]


def initialise_pheromone_matrix(init_pher):
    return [[init_pher for _ in range(N_CITIES)] for _ in range(N_CITIES)]


def calculate_choices(last_city_visited_by_truck, last_drone_visit, last_city_visited_by_drone, drone_distance,
                      truck_distance, cities_involved, exclude, pheromone, c_heur, c_hist):
    choices = []
    for city in range(1, N_CITIES):
        for drone_visit in range(0, 2):
            if city in exclude:
                continue
            prob = {"city": city,
                    "drone_visit": drone_visit,
                    "history": pheromone[last_city_visited_by_truck][city] ** c_hist,
                    "distance": distance(last_city_visited_by_truck, city, last_drone_visit, drone_visit,
                                         drone_distance, truck_distance, last_city_visited_by_drone, cities_involved)
                    }
            prob["heuristic"] = (1.0 / prob["distance"]) ** c_heur
            prob["prob"] = prob["history"] * prob["heuristic"]
            choices.append(prob)
    return choices


def prob_select(choices):
    sum_all = 0
    for choice in choices:
        sum_all += choice["prob"]
    if sum_all == 0:
        return choices[random.randint(1, len(choices) - 1)]

    v = random.random()
    for index, choice in enumerate(choices):
        v -= choice["prob"] / sum_all
        if v <= 0.0:
            return choice

    return choices[-1]


def greedy_select(choices):
    return max(choices, key=lambda e: e["prob"])


def stepwise_const(phero, c_heur, c_greed):
    perm = [0]
    drone_visits = [0]
    drone_distance = 0
    truck_distance = 0
    last_drone_visit = 0
    last_city_visited_by_truck = 0
    last_city_visited_by_drone = 0
    cities_involved = 1

    while True:
        choices = calculate_choices(
            last_city_visited_by_truck=last_city_visited_by_truck,
            last_drone_visit=last_drone_visit,
            last_city_visited_by_drone=last_city_visited_by_drone,
            drone_distance=drone_distance,
            truck_distance=truck_distance,
            cities_involved=cities_involved,
            exclude=perm,
            pheromone=phero,
            c_heur=c_heur,
            c_hist=1.0
        )

        greedy = random.random() <= c_greed
        if greedy:
            next_choice = greedy_select(choices)
        else:
            next_choice = prob_select(choices)

        if next_choice["drone_visit"] == last_drone_visit:
            drone_distance = 0
            truck_distance = 0
            last_city_visited_by_truck = next_choice["city"]
            last_city_visited_by_drone = next_choice["city"],
            last_drone_visit = 0
            cities_involved = 1
        elif next_choice["drone_visit"]:
            drone_distance = dist(last_city_visited_by_drone, next_choice["city"])
            last_drone_visit = 1
            last_city_visited_by_drone = next_choice["city"]
            cities_involved += 1
        else:
            truck_distance += dist(last_city_visited_by_truck, next_choice["city"])
            last_city_visited_by_truck = next_choice["city"]
            cities_involved += 1

        perm.append(next_choice["city"])
        drone_visits.append(next_choice["drone_visit"])
        if len(perm) == N_CITIES:
            break
    return perm, drone_visits


def global_update_pheromone(phero, cand, decay):
    for index, x in enumerate(cand["cities"]):
        if index == len(cand["cities"]) - 1:
            y = cand["cities"][0]
        else:
            y = cand["cities"][index + 1]
        value = ((1.0 - decay) * phero[x][y] + decay * (1.0 / cand["cost"]))
        phero[x][y] = value
        phero[y][x] = value


def local_update_pheromone(pheromone, cand, c_local_phero, init_phero):
    for index, x in enumerate(cand["cities"]):
        if index == len(cand["cities"]) - 1:
            y = cand["cities"][0]
        else:
            y = cand["cities"][index + 1]
        value = ((1.0 - c_local_phero) * pheromone[x][y]) + (c_local_phero * init_phero)
        pheromone[x][y] = value
        pheromone[y][x] = value


def search(max_it, num_ants, decay, c_heur, c_local_phero, c_greed):
    best = {"cities": random_permutation(N_CITIES), "drone_visits": random_bool_list(N_CITIES)}
    best["cost"] = cost(best["cities"], best["drone_visits"])
    init_pheromone = 1.0 / (N_CITIES * best["cost"])
    pheromone = initialise_pheromone_matrix(init_pheromone)

    solutions = [deepcopy(best)]
    for iteration in range(max_it):
        for ant_number in range(num_ants):
            cand = {}
            cand["cities"], cand["drone_visits"] = stepwise_const(pheromone, c_heur, c_greed)
            cand["cost"] = cost(cand["cities"], cand["drone_visits"])
            if cand["cost"] < best["cost"]:
                best = cand
                solutions.append(deepcopy(best))
            local_update_pheromone(pheromone, cand, c_local_phero, init_pheromone)
        global_update_pheromone(pheromone, best, decay)
        print("Iteration %d: best=%f" % (iteration, best["cost"]))
        c_greed *= 0.99
    return best, solutions


def main():
    berlin52 = [(565, 575), (25, 185), (345, 750), (945, 685), (845, 655),
                (880, 660), (25, 230), (525, 1000), (580, 1175), (650, 1130), (1605, 620),
                (1220, 580), (1465, 200), (1530, 5), (845, 680), (725, 370), (145, 665),
                (415, 635), (510, 875), (560, 365), (300, 465), (520, 585), (480, 415),
                (835, 625), (975, 580), (1215, 245), (1320, 315), (1250, 400), (660, 180),
                (410, 250), (420, 555), (575, 665), (1150, 1160), (700, 580), (685, 595),
                (685, 610), (770, 610), (795, 645), (720, 635), (760, 650), (475, 960),
                (95, 260), (875, 920), (700, 500), (555, 815), (830, 485), (1170, 65),
                (830, 610), (605, 625), (595, 360), (1340, 725), (1740, 245)]
    read_data()

    max_it = 200
    num_ants = 10
    decay = 0.1
    c_heur = 2.5
    c_local_phero = 0.1
    c_greed = 0.70
    best, solutions = search(max_it, num_ants, decay, c_heur, c_local_phero, c_greed)

    print(solutions)
    print("Best solution: c=%f, v=(%s, %s)" % (best["cost"], str(best["cities"]), str(best["drone_visits"])))


if __name__ == '__main__':
    main()
