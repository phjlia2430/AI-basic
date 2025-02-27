import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 이미지 크기 설정
IMAGE_SIZE = (128, 128)

# 데이터 로드 함수
def load_train_data(folder_path):
    X = []
    y = []
    class_names = os.listdir(folder_path)
    print("클래스 이름:", class_names)

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(folder_path, class_name)
        for image_name in os.listdir(class_path):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(class_path, image_name)
                image = load_img(image_path, target_size=IMAGE_SIZE)
                image = img_to_array(image)
                X.append(image)
                y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names

# 데이터 로드
train_folder = "C:/pythonprojects/pythonProject1/flowers-dataset/train"
X, y, class_names = load_train_data(train_folder)

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화
X_train = X_train / 255.0
X_test = X_test / 255.0

# 원-핫 인코딩 (CNN 학습 시 필요)
num_classes = len(class_names)
y_train_onehot = to_categorical(y_train, num_classes)
y_test_onehot = to_categorical(y_test, num_classes)

# CNN 모델 정의 (특징 추출)
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# CNN 학습 준비
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

# 특징 추출
X_train_features = cnn_model.predict(X_train, verbose=1)
X_test_features = cnn_model.predict(X_test, verbose=1)

# MLP 모델 정의
mlp_model = Sequential([
    Dense(1256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(100, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# MLP 학습 준비
mlp_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# MLP 모델 학습
mlp_model.fit(X_train_features, y_train,
              epochs=15,  # 에폭 증가
              batch_size=128,
              verbose=1)

# MLP 모델 평가
y_pred_probs = mlp_model.predict(X_test_features, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

test_acc = accuracy_score(y_test, y_pred)

print('\nCNN+MLP 테스트 정확도:', test_acc)

# 정확도 및 손실 그래프
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Training Accuracy and Loss for CNN+MLP_Flowers')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

# 혼동 행렬 계산
conf_matrix = confusion_matrix(y_test, y_pred)

# 혼동 행렬 시각화
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xticks(rotation=45)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for CNN+MLP_Flowers')
plt.show()
