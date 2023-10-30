import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # 初始化权重和偏置
        self.weights, self.biases = self.initialize_weights_and_biases()

        # 存储中间结果，用于反向传播
        self.inputs = []
        self.outputs = []
        self.activations = []

    def initialize_weights_and_biases(self):
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        weights = [np.random.randn(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        biases = [np.zeros((1, sizes[i + 1])) for i in range(len(sizes) - 1)]
        return weights, biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, input_data):
        self.inputs = []
        self.outputs = []
        self.activations = []

        current_input = input_data
        for i in range(len(self.weights)):
            self.inputs.append(current_input)
            current_output = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.outputs.append(current_output)
            current_activation = self.sigmoid(current_output)
            self.activations.append(current_activation)
            current_input = current_activation

        return current_input

    def backward(self, target, learning_rate):
        # 反向传播
        errors = [target - self.activations[-1]]
        deltas = [errors[-1] * self.sigmoid_derivative(self.activations[-1])]

        for i in range(len(self.weights) - 1, 0, -1):
            errors.append(deltas[-1].dot(self.weights[i].T))
            deltas.append(errors[-1] * self.sigmoid_derivative(self.activations[i - 1]))

        # 更新权重和偏置
        for i in range(len(self.weights) - 1, -1, -1):
            self.weights[i] += learning_rate * self.inputs[i].T.dot(deltas[len(self.weights) - 1 - i])
            self.biases[i] += learning_rate * np.sum(deltas[len(self.weights) - 1 - i], axis=0, keepdims=True)

    def train(self, input_data, target, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(input_data)
            self.backward(target, learning_rate)

            if epoch % 1000 == 0:
                loss = np.mean(np.square(target - output))
                print(f"Epoch {epoch}, Loss: {loss}")

# 示例用法
if __name__ == "__main__":
    # 输入大小为2，隐藏层大小为3，输出大小为1
    nn = NeuralNetwork(input_size=2, hidden_sizes=[3], output_size=1)

    # 训练数据
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target = np.array([[0], [1], [1], [0]])

    # 训练神经网络
    nn.train(input_data, target, epochs=10000, learning_rate=0.1)

    # 测试
    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = nn.forward(test_data)
    print("Predictions:")
    print(predictions)


