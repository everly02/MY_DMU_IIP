import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autokeras import StructuredDataClassifier

# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # 使用LabelEncoder对非数值型特征进行编码
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == np.object_:
            data[column] = label_encoder.fit_transform(data[column])

   
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values
    return features, target

if __name__ == "__main__":
    # 加载数据
    file_path = "music_data.csv"  
    features, target = load_data(file_path)

    
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    
    clf = StructuredDataClassifier(max_trials=10, metrics='accuracy', seed=42)
    clf.fit(train_features, train_target, epochs=100)

    # 在测试集上评估最佳模型
    predictions = clf.predict(test_features)
    accuracy = accuracy_score(test_target, predictions)
    print("Test Accuracy with Best Model from AutoKeras:", accuracy)

     # 获取最佳模型的架构
    best_model = clf.export_model()

    # 打印最佳模型的架构
    print("Best Model Architecture:")
    print(best_model.summary())
