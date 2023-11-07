import pickle
import matplotlib.pyplot as plt
from ga import genetic_algorithm, best_fitness_values
#from sa import simulated_annealing

#加载图
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



