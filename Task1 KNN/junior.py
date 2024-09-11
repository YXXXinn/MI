import numpy as np

# 加载数据集并分离特征与标签
data = np.loadtxt('semeion.data')
X = data[:, :256]
Y = np.argmax(data[:, 256:], axis=1)

# 欧氏距离函数
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# 实现kNN算法
def k_nearest_neighbors(X_train, Y_train, X_test, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], X_test)
        distances.append((dist, Y_train[i]))
    
    # 按距离排序，选择距离最近的k个点
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    
    # 统计k个最近邻中的标签频率
    class_votes = {}
    for neighbor in neighbors:
        label = neighbor[1]
        if label in class_votes:
            class_votes[label] += 1
        else:
            class_votes[label] = 1
    
    # 返回得票最多的标签
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]

# 留一法交叉验证
def leave_one_out_cross_validation(X, Y, k):
    correct_predictions = 0
    total_samples = len(X)
    
    for i in range(total_samples):
        X_train = np.delete(X, i, axis=0)
        Y_train = np.delete(Y, i, axis=0)
        X_test = X[i]
        Y_test = Y[i]
        
        predicted_label = k_nearest_neighbors(X_train, Y_train, X_test, k)
        
        if predicted_label == Y_test:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_samples
    return accuracy

# 选择不同的k值进行测试
k_values = [5, 9, 13]
accuracies = {}

for k in k_values:
    accuracy = leave_one_out_cross_validation(X, Y, k)
    accuracies[k] = accuracy
    print(f'k = {k}, 分类精度 = {accuracy * 100:.2f}%')

print("不同k值下的分类精度：", accuracies)

