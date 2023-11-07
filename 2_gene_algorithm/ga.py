#用遗传算法实现图划分
import random
import pickle
import matplotlib.pyplot as plt

def initialize_population(population_size, num_nodes, num_subsets):
    return [[random.randint(0, num_subsets-1) for _ in range(num_nodes)] for _ in range(population_size)]

def fitness(individual, graph, k):
    #graph是邻接矩阵，individual是表示分组的列表
    total_weight = 0
    num_nodes = len(individual)
    num_subsets = max(individual) + 1#最大的individual元素值，比如0，1，2，那2+1=3个分组

    for subset in range(num_subsets):
        subset_nodes = [i for i, s in enumerate(individual) if s == subset]
        
        if len(subset_nodes) < k:
            # Penalize solutions with subsets having fewer than k nodes
            total_weight += (k - len(subset_nodes)) * 1000  # Adjust penalty as needed
        else:
            for i in range(len(subset_nodes)):
                for j in range(i+1, len(subset_nodes)):
                    total_weight += graph[subset_nodes[i]][subset_nodes[j]]

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

def select(population, fitness_values):
    
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]

    selected_index = roulette_wheel(probabilities)
    selected_individual = population[selected_index]

    return selected_individual

def roulette_wheel(probabilities):
    r = random.random()
    cumulative_probability = 0

    for i, prob in enumerate(probabilities):
        cumulative_probability += prob
        if r <= cumulative_probability:
            return i

    return len(probabilities) - 1

def genetic_algorithm(graph, num_subsets, k1, population_size, generations, crossover_rate, mutation_rate):
    population = initialize_population(population_size, len(graph), num_subsets)
    
    for generation in range(generations):
        population = sorted(population, key=lambda x: fitness(x, graph,k1))
        new_population = []#population是individual的列表
        best_solution = population[0]#选第一个个体做最佳
        
        total_fitness = sum(fitness(ind, graph, k1) for ind in population)
        for i in range(0, population_size, 2):
            parent1 = select(population, [fitness(ind, graph, k1) for ind in population])
            parent2 = select(population, [fitness(ind, graph, k1) for ind in population])
            
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            child1 = mutate(child1, mutation_rate, num_subsets)
            child2 = mutate(child2, mutation_rate, num_subsets)
            
            new_population.extend([child1, child2])

        population = new_population

        best_solution = min(population, key=lambda x: fitness(x, graph,k1))
        best_fitness = fitness(best_solution, graph, k1)
        best_fitness_values.append(best_fitness)
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

