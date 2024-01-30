def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    selected_items = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                if values[i - 1] + dp[i - 1][w - weights[i - 1]] > dp[i - 1][w]:
                    dp[i][w] = values[i - 1] + dp[i - 1][w - weights[i - 1]]
                    selected_items[i][w] = 1
                else:
                    dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = dp[i - 1][w]

    # 回溯找出选择的物品
    selected_items_list = []
    i, w = n, capacity
    while i > 0 and w > 0:
        if selected_items[i][w] == 1:
            selected_items_list.append(i - 1)
            w -= weights[i - 1]
        i -= 1

    return selected_items_list, dp[n][capacity], sum(weights[i] for i in selected_items_list), sum(values[i] for i in selected_items_list)
names = []
weights = []
values = []
capacity=0 

file_name = input("enter file path: ")
print()

with open(file_name) as file:
    
    lines = file.readlines()
    capacity = int(lines[0].strip())
    for line in lines[1:]:
        name, weight, value = map(str.strip, line.split(' '))
        names.append(name)
        weights.append(int(weight))
        values.append(int(value))
    selected_items , total_value, total_weight, _= knapsack(weights, values, capacity)
    for item in range(0,len(names)):
        print(f"{names[item]}", end=" ")
        if (item) in selected_items:
            print(1)
        else :
            print(0)
    print(f" profit:{total_value}")
    print(f" totel_weight:{total_weight}")

            