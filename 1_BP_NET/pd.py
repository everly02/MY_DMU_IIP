import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#import nni
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
            
    def threshold_classify(self, output, threshold=0.5):
        # 二元分类
        return (output > threshold).astype(int)
    
def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # 对特征进行编码
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == np.object_:
            data[column] = label_encoder.fit_transform(data[column])

    # 最后一列是目标变量，前面的列是特征
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values.reshape(-1, 1)
    return features, target

# 划分数据集
def split_data(features, target, train_ratio=0.8):
    num_samples = len(features)
    num_train_samples = int(train_ratio * num_samples)

    train_features = features[:num_train_samples]
    train_target = target[:num_train_samples]
    test_features = features[num_train_samples:]
    test_target = target[num_train_samples:]

    return train_features, train_target, test_features, test_target

def plot_lr(losses):
    plt.figure(figsize=(6, 6))

    # 学习曲线
    #plt.subplot(1, 2, 1)
    plt.plot(losses, label='training loss')
    plt.title('training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
def plot_acc(accuracies):
    
    plt.figure(figsize=(6, 6))
    # 准确率曲线
    #plt.subplot(1, 2, 1)
    plt.plot(accuracies, label='test set accuracy', color='orange')
    plt.title('test set accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 加载数据
    file_path = "music_data.csv"  
    features, target = load_data(file_path)

    # 划分数据集
    train_features, train_target, test_features, test_target = split_data(features, target)

    # 创建神经网络
    input_size = train_features.shape[1]
    hidden_sizes = [7,6,4]
    output_size = 1
    nn = NeuralNetwork(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)

    # 训练BP神经网络
    epochs = 10000
    learning_rate = 0.001
    
    losses = []  # 记录训练损失
    accuracies = []  # 记录测试准确率

    for epoch in range(epochs):
        output = nn.forward(train_features)
        nn.backward(train_target, learning_rate)

        # 记录训练损失
        loss = np.mean(np.square(train_target - output))
        losses.append(loss)
    
        if epoch % 1000 == 0:
            # 在验证集上计算准确率
            predictions = nn.forward(test_features)
            threshold_predictions = nn.threshold_classify(predictions)
            accuracy = np.mean(threshold_predictions == test_target)
            accuracies.append(accuracy)

            print(f"Epoch {epoch}, Training Loss: {loss}, Test Accuracy: {accuracy}")
    
    
    # 计算测试集的准确率
    final_accuracy = np.mean(threshold_predictions == test_target)
    print("测试结果:")
    print(f"测试准确率: {final_accuracy}")
    
    # 绘制学习曲线
    #plot_lr(losses)
    
    # 测试BP神经网络
    predictions = nn.forward(test_features)
    threshold_predictions = nn.threshold_classify(predictions)
    for pd in range(1,len(predictions)):
        print(f"预测：{threshold_predictions[pd]}, 实际：{test_target[pd]}")
    #print(threshold_predictions)
    plot_acc(accuracies)
    
    
    

