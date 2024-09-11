import numpy as np
import scipy.ndimage

# 加载数据集 (1593, 256)
data = np.loadtxt('semeion.data')
X = data[:, :256].reshape(-1, 16, 16)  # 将每个样本重新调整为16x16的图像
Y = np.argmax(data[:, 256:], axis=1)   # 提取标签

# 图像旋转函数
def rotate_image(image, angle):
    return scipy.ndimage.rotate(image, angle, reshape=False)

# 对所有图像进行左上(45度)和左下(-45度)旋转
X_rotated_top_left = np.array([rotate_image(img, 45) for img in X])
#X_rotated_bottom_left = np.array([rotate_image(img, -45) for img in X])
#X_augmented=X_rotated_top_left
X_augmented=X
Y_augmented=Y
# 扩展数据集
#X_augmented = np.concatenate([X, X_rotated_top_left, X_rotated_bottom_left], axis=0)
#Y_augmented = np.concatenate([Y, Y, Y], axis=0)

# 将图像数据重新展平为向量 (以便与CNN兼容)
X_augmented = X_augmented.reshape(-1, 16, 16, 1)  # CNN需要输入 (batch_size, height, width, channels)


import tensorflow as tf
from tensorflow.keras import layers, models

# 创建CNN模型
def create_cnn_model():
    model = models.Sequential()
    
    # 第一个卷积层 + 池化层
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 第二个卷积层 + 池化层
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 全连接层
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # 10分类
    
    # 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 创建模型
cnn_model = create_cnn_model()

# 训练模型
cnn_model.fit(X_augmented, Y_augmented, epochs=10, batch_size=32, validation_split=0.2)

# 评估模型性能
test_loss, test_acc = cnn_model.evaluate(X_augmented, Y_augmented)
print(f"测试集精度: {test_acc * 100:.2f}%")
