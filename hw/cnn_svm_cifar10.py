import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# CIFAR-10 데이터셋 로드
cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# CIFAR-10 클래스 이름 정의
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# 데이터 정규화
X_train, X_test = X_train / 255.0, X_test / 255.0

# 레이블 One-Hot Encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

# CNN 모델 정의 (특징 추출용)
cnn_model = keras.Sequential([
    keras.layers.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# CNN 모델 학습
history = cnn_model.fit(
    X_train, y_train_onehot,
    epochs=10,
    batch_size=128,
    verbose=1
)


# CNN 특징 추출 (배치 크기 설정)
X_train_features = cnn_model.predict(X_train, batch_size=128, verbose=2)
X_test_features = cnn_model.predict(X_test, batch_size=128, verbose=2)


# SVM 모델 정의
svm_model = SVC(kernel='rbf', C=1, decision_function_shape='ovr')


# SVM 모델 학습
y_train_labels = np.argmax(y_train_onehot, axis=1)
svm_model.fit(X_train_features, y_train_labels)

# 테스트 데이터 예측
predicted_labels = svm_model.predict(X_test_features)

# 정확도 평가
true_labels = np.argmax(y_test_onehot, axis=1)
svm_accuracy = accuracy_score(true_labels, predicted_labels)
print("\nSVM 테스트 정확도:", svm_accuracy)

# 정확도 및 손실 그래프
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Training Accuracy and Loss for CNN+SVM_CIFAR10')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

# 혼동 행렬 계산
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xticks(rotation=45)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for CNN+SVM_CIFAR10')
plt.show()



