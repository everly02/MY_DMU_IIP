import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner

# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # 使用LabelEncoder对非数值型特征进行编码
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == object:  # Replace np.object with object
            data[column] = label_encoder.fit_transform(data[column])

    # 假设数据的最后一列是目标变量，前面的列是特征
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values
    return features, target


# 定义搜索空间
search_space = {
    'hidden_layer_sizes': {
        '_type': 'choice',
        '_value': [(3,), (5,), (3, 2), (5, 2)]
    },
    'activation': {
        '_type': 'choice',
        '_value': ['logistic', 'tanh', 'relu']
    },
    'learning_rate_init': {
        '_type': 'choice',
        '_value': [0.001, 0.01, 0.1]
    }
}

# 加载数据
file_path = "music_data.csv"  # 替换成你的CSV文件路径
features, target = load_data(file_path)

# 划分数据集
train_features, test_features, train_target, test_target = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# 定义神经网络模型
def create_nn(hidden_layer_sizes, activation, learning_rate_init):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        learning_rate_init=learning_rate_init,
        max_iter=1000,
        random_state=42
    )

# 定义 NNITuner
tuner = HyperoptTuner('tpe', search_space)

# 定义神经网络评估函数
def evaluate_network(params):
    hidden_layer_sizes = params['hidden_layer_sizes']
    activation = params['activation']
    learning_rate_init = params['learning_rate_init']

    model = create_nn(hidden_layer_sizes, activation, learning_rate_init)
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_target, predictions)

    # NNI 要求最小化目标函数，因此这里使用 1 - accuracy
    return 1 - accuracy

# 使用 NNI 进行神经网络架构搜索
tuner.update_search_space(search_space)
tuner.receive_trial_parameters({'parameter_id': 0, 'parameters': search_space})
trial_result = tuner.receive_trial_result({'parameter_id': 0, 'value': evaluate_network(search_space)})

# 输出最佳超参数组合
best_params = tuner.get_best_trial()
print("Best Hyperparameters:", best_params)
