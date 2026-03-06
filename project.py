import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter

train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")
train.head()

target = 'label'
x_train = train.drop(target, axis=1)
y_train = train[target]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(test)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(x_train, y_train, x, k):
    distances = []
    for i in range(len(x_train)):
        dist = euclidean_distance(x, x_train[i])
        distances.append((dist, y_train.iloc[i]))
    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]
    labels = [label for _, label in k_neighbors]
    return Counter(labels).most_common(1)[0][0]

def knn_predict_all(x_train, y_train, x_test, k):
    y_pred = []
    for i in range(len(x_test)):
        pred = knn_predict(x_train, y_train, x_test[i], k)
        y_pred.append(pred)
    return np.array(y_pred)

y_test_predict = knn_predict_all(x_train, y_train, x_test[:20], k=10)

for i in range(20):
    image = test.iloc[i].values.reshape(28, 28)
    plt.figure(figsize=(5,5))
    plt.imshow(image, cmap='Blues')
    plt.axis('off')
    plt.show()
    print("Số dự đoán:", y_test_predict[i])
