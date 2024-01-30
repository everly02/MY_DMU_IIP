import random
import string

def generate_knapsack_data(num_items, capacity):
    # 生成随机物品数据
    items = []
    for _ in range(num_items):
        item_name = ''.join(random.choices(string.ascii_uppercase, k=1))
        item_weight = random.randint(10, 60)
        item_value = random.randint(20, 200)
        items.append((item_name, item_weight, item_value))

    # 将物品按照字母排序
    items.sort()

    # 将数据写入文件
    with open("data.txt", "w") as file:
        file.write(str(capacity) + "\n")
        for item in items:
            file.write(f"{item[0]} {item[1]} {item[2]}\n")

# 设置物品数量和背包容量
num_items = 100
knapsack_capacity = 250

# 调用生成器函数
generate_knapsack_data(num_items, knapsack_capacity)
