import pickle
import random
def generate_random_graph(num_nodes, num_edges):
    # Generate a graph with random edges and weights
    graph = [[0] * num_nodes for _ in range(num_nodes)]

    # Randomly add edges with random weights
    for _ in range(num_edges):
        node1, node2 = random.sample(range(num_nodes), 2)
        weight = random.randint(1, 10)  # Adjust the range of weights as needed
        graph[node1][node2] = weight
        graph[node2][node1] = weight  # Assuming an undirected graph

    return graph

def print_graph(graph):
    # Print the generated graph
    for row in graph:
        print(row)

def save_graph_to_file(graph, filename):
    with open(filename, 'wb') as file:
        pickle.dump(graph, file)

def load_graph_from_file(filename):
    with open(filename, 'rb') as file:
        graph = pickle.load(file)
    return graph

print('输入节点，边数：')
p1=input()
p1=int(p1)
p2=input()
p2=int(p2)
g=generate_random_graph(p1,p2)
print('已生成图:')
print_graph(g)

# 保存
save_graph_to_file(g, 'graph_data.pkl')