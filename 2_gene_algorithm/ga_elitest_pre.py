import random
import pickle
import matplotlib.pyplot as plt

def initialize_population(population_size, num_nodes, num_subsets):
    return [[random.randint(0, num_subsets-1) for _ in range(num_nodes)] for _ in range(population_size)]

def fitness(individual, graph):
    total_weight = 0
    num_nodes = len(individual)
    
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if individual[i] != individual[j]:
                total_weight += graph[i][j]

    return total_weight

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1)-1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate, num_subsets):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, num_subsets-1)
    return individual

def elitist_preserving_selection(population, graph):
    population = sorted(population, key=lambda x: fitness(x, graph))
    elite_size = int(0.1 * len(population))  # Preserve top 10% as elite
    elite = population[:elite_size]
    non_elite = population[elite_size:]
    return elite + random.sample(non_elite, len(population)-elite_size)

def genetic_algorithm(graph, num_subsets, k, population_size, generations, crossover_rate, mutation_rate):
    population = initialize_population(population_size, len(graph), num_subsets)

    for generation in range(generations):
        population = elitist_preserving_selection(population, graph)

        new_population = []

        fitness_list=[]
        for i in range(0,population_size):
            fitness_list.append(fitness(population[i],graph))
        best_fitness=max(fitness_list)
        for i in range(0, population_size, 2):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            child1 = mutate(child1, mutation_rate, num_subsets)
            child2 = mutate(child2, mutation_rate, num_subsets)
            
            new_population.extend([child1, child2])
        best_fitness_values.append(best_fitness)
        population = new_population

    best_solution = min(population, key=lambda x: fitness(x, graph))
    best_fitness = fitness(best_solution, graph)

    return best_solution, best_fitness
best_fitness_values = []

def load_graph_from_file(filename):
    with open(filename, 'rb') as file:
        graph = pickle.load(file)
    return graph
g = load_graph_from_file('graph_data.pkl')

num_subsets = int(input('子集数-'))
k = int(input('均衡约束-'))
population_size = int(input('初始的种群大小-'))
generations = int(input('generation number-'))
crossover_rate = float(input('交叉率-'))
mutation_rate = float(input('变异率-'))
best_solution, best_fitness = genetic_algorithm(g, num_subsets, k, population_size, generations, crossover_rate, mutation_rate)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)


plt.plot(range(generations), best_fitness_values, label='Total Fitness')
plt.xlabel('Generation')
plt.ylabel('Total Fitness')
plt.title('Genetic Algorithm: Total Fitness Curve')
plt.legend()
plt.show()

