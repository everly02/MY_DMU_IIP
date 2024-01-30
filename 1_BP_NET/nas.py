import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
def load_data(file_path):
    data = pd.read_csv(file_path)
    # LabelEncoder对非数值型特征进行编码
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == np.object_:
            data[column] = label_encoder.fit_transform(data[column])
    # 最后一列是目标变量，前面的列是特征
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values
    return features, target
if __name__ == "__main__":
    # 加载
    file_path = "C:\\Users\\Eliezer\\Documents\\GitHub\\MY_DMU_IIP\\1_BP_NET\\music_data.csv"  
    features, target = load_data(file_path)
    # 划分数据集
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    #MLPClassifier模型
    mlp = MLPClassifier()
    # 定义超参数搜索空间
    param_grid = {
        'hidden_layer_sizes': [ (10), (10,50)],
        'learning_rate_init': [0.002,0.001,0.005],
        'max_iter': [9000,5000]
    }

    # 使用GridSearchCV进行超参数搜索
    grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(train_features, train_target)
    # 输出最佳参数
    print("Best Hyperparameters:", grid_search.best_params_)
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    # 在测试集上评估最佳模型
    predictions = best_model.predict(test_features)
    accuracy = accuracy_score(test_target, predictions)
    print("Test Accuracy with Best Model:", accuracy)
