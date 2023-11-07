#用模拟退火算法实现图划分
import networkx as nx
import random
import math

def initialize_partition(graph, num_partitions):
    partition = {}
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    partition_size = len(nodes) // num_partitions

    for i in range(num_partitions - 1):
        partition.update({node: i for node in nodes[i * partition_size: (i + 1) * partition_size]})

    partition.update({node: num_partitions - 1 for node in nodes[(num_partitions - 1) * partition_size:]})

    return partition

def calculate_cut_size(graph, partition):
    cut_size = 0
    for edge in graph.edges():
        if partition[edge[0]] != partition[edge[1]]:
            cut_size += graph[edge[0]][edge[1]].get("weight", 1)  # 默认权重为1
    return cut_size

def simulated_annealing(graph, num_partitions, temperature, cooling_rate, num_iterations):
    current_partition = initialize_partition(graph, num_partitions)
    current_cut_size = calculate_cut_size(graph, current_partition)

    best_partition = current_partition
    best_cut_size = current_cut_size

    for iteration in range(num_iterations):
        # Generate a neighboring solution
        neighbor_partition = current_partition.copy()
        node_to_move = random.choice(list(graph.nodes()))
        current_partition[node_to_move] = random.randint(0, num_partitions - 1)

        # Calculate the cut size for the neighbor
        neighbor_cut_size = calculate_cut_size(graph, neighbor_partition)

        # Decide whether to accept the neighbor
        if neighbor_cut_size < current_cut_size or random.uniform(0, 1) < math.exp((current_cut_size - neighbor_cut_size) / temperature):
            current_partition = neighbor_partition
            current_cut_size = neighbor_cut_size

        # Update the best solution if needed
        if current_cut_size < best_cut_size:
            best_partition = current_partition
            best_cut_size = current_cut_size

        # Cool the temperature
        temperature *= cooling_rate

    return best_partition, best_cut_size

# 使用 NetworkX 创建一个带权重的图
G = nx.Graph()
G.add_edges_from([(0, 1, {"weight": 2}), (0, 2, {"weight": 1}), (1, 2, {"weight": 3}), (1, 3, {"weight": 4})])

# 设定算法参数
num_partitions = 2
initial_temperature = 100.0
cooling_rate = 0.99
num_iterations = 1000

# 运行模拟退火算法
best_partition, best_cut_size = simulated_annealing(G, num_partitions, initial_temperature, cooling_rate, num_iterations)

print("Best Partition:", best_partition)
print("Best Cut Size:", best_cut_size)
