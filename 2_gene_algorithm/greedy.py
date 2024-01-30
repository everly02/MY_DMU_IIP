#用贪心算法进行图划分
import pickle
import numpy as np
def greedy_partition(adj_matrix, num_partitions, balance_constraint):
    num_nodes = len(adj_matrix)
    
    # 初始化划分结果和权重
    partition_result = [-1] * num_nodes
    partition_weights = [0] * num_partitions
    
    # 按照节点度量进行排序
    nodes_sorted_by_degree = np.argsort(np.sum(adj_matrix, axis=0))
    
    for node in nodes_sorted_by_degree:
        # 找到当前最轻的划分
        min_partition = np.argmin(partition_weights)
        
        # 将节点分配到最轻的划分
        partition_result[node] = min_partition
        partition_weights[min_partition] += np.sum(adj_matrix[node])
        
        # 检查平衡约束是否满足
        if max(partition_weights) - min(partition_weights) > balance_constraint:
            # 如果不满足，回退操作并尝试下一个划分
            partition_weights[min_partition] -= np.sum(adj_matrix[node])
            min_partition = np.argmin(partition_weights)
            partition_result[node] = min_partition
            partition_weights[min_partition] += np.sum(adj_matrix[node])
    
    return partition_result, max(partition_weights) - min(partition_weights)

def load_graph_from_file(filename):
    with open(filename, 'rb') as file:
        graph = pickle.load(file)
    return graph
g = load_graph_from_file('graph_data.pkl')
# 示例用法
#N = len(g)  # 节点数

M = int(input('子集数-')) # 子集数
K = int(input('均衡约束-')) # 每个子集的节点数至少为K

result, total_weight = greedy_partition(g,M,K)
print("best result:", result)
print("fitness", total_weight)