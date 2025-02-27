import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

# 원-핫 인코딩
num_classes = len(class_names)
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)

# MLP 모델 정의
model = Sequential([
    Flatten(input_shape=(128, 128, 3)),
    Dense(1256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(100, activation='relu'),
    Dense(5, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history=model.fit(X_train, y_train_onehot, epochs=15, batch_size=128, verbose=1)

# 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
print("\nmlp 테스트 정확도:",test_acc)

# 혼동행렬 생성 및 시각화
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title("Confusion Matrix for MLP_flower")
plt.show()

# 정확도 및 손실 그래프
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Training Accuracy and Loss for MLP_flower')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()


