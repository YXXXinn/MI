from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, confusion_matrix
import numpy as np

# 加载数据集 (1593, 256)
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
# 计算熵 H(Y)
def entropy(labels):
    label_counts = np.bincount(labels)
    probs = label_counts / len(labels)
    return -np.sum(probs * np.log2(probs + 1e-9))  # 避免 log(0)

# 计算互信息 I(Y_true, Y_pred)
def mutual_information(y_true, y_pred):
    total = len(y_true)
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    mi = 0.0
    for t in unique_true:
        for p in unique_pred:
            p_t = np.sum(y_true == t) / total
            p_p = np.sum(y_pred == p) / total
            p_tp = np.sum((y_true == t) & (y_pred == p)) / total
            
            if p_tp > 0:
                mi += p_tp * np.log2(p_tp / (p_t * p_p + 1e-9))  # 避免 log(0)
    
    return mi

# 计算归一化互信息 NMI
def normalized_mutual_information(y_true, y_pred):
    H_true = entropy(y_true)
    H_pred = entropy(y_pred)
    I = mutual_information(y_true, y_pred)
    
    return 2 * I / (H_true + H_pred + 1e-9)  # 避免分母为0

# 测试函数
y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 0, 2, 1, 2, 1])

nmi = normalized_mutual_information(y_true, y_pred)
print(f"NMI: {nmi:.4f}")

# 使用 scikit-learn 的 kNN 分类器
def sklearn_knn(X, Y, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, Y)
    
    # 预测标签
    y_pred = knn.predict(X)
    
    # 计算精度（ACC）
    acc = accuracy_score(Y, y_pred)
    
    # 计算归一化互信息（NMI）
    nmi = normalized_mutual_info_score(Y, y_pred)
    
    # 计算混淆矩阵并基于此计算混淆熵（CEN）
    cm = confusion_matrix(Y, y_pred)
    cen = confusion_entropy(cm)
    
    return acc, nmi, cen

# 混淆熵（CEN）计算函数，基于混淆矩阵
def confusion_entropy(conf_matrix):
    total = np.sum(conf_matrix)
    num_classes = len(conf_matrix)
    CEN = 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            if conf_matrix[i, j] > 0:
                pij = conf_matrix[i, j] / total
                pji = conf_matrix[j, i] / total
                CEN += pij * np.log2(1 + abs(pij - pji))
    return CEN

# 比较结果
k_values = [5, 9, 13]

for k in k_values:
    print(f"--- k = {k} ---")
    
    # 自己实现的kNN
    acc_ours = leave_one_out_cross_validation(X, Y, k)
    y_pred_ours = [k_nearest_neighbors(X, Y, X[i], k) for i in range(len(X))]
    nmi_ours = normalized_mutual_information(Y, y_pred_ours)
    cen_ours = confusion_entropy(confusion_matrix(Y, y_pred_ours))
    
    print(f"我们实现的kNN - 精度 (ACC): {acc_ours:.4f}, NMI: {nmi_ours:.4f}, CEN: {cen_ours:.4f}")
    
    # scikit-learn的kNN
    acc_sklearn, nmi_sklearn, cen_sklearn = sklearn_knn(X, Y, k)
    print(f"scikit-learn的kNN - 精度 (ACC): {acc_sklearn:.4f}, NMI: {nmi_sklearn:.4f}, CEN: {cen_sklearn:.4f}")

