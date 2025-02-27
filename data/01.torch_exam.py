# PyTorch 라이브러리 불러오기
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# MNIST 데이터셋 불러오기 및 전처리
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 간단한 신경망 모델 정의
model = nn.Sequential(
    nn.Flatten(),  # 28x28 이미지를 1차원으로 펼침
    nn.Linear(28 * 28, 128),  # 입력층에서 은닉층으로
    nn.ReLU(),  # 활성화 함수 ReLU
    nn.Linear(128, 10)  # 은닉층에서 출력층으로 (10개의 숫자 분류)
)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
epochs = 5
for epoch in range(epochs):
    for images, labels in train_loader:
        # 예측
        outputs = model(images)

        # 손실 계산
        loss = criterion(outputs, labels)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 테스트 데이터셋에서 모델 평가
correct = 0
total = 0
with torch.no_grad():  # 평가 시에는 기울기 계산 안함
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 첫 번째 테스트 이미지 예측 및 시각화
image, label = test_dataset[0]
with torch.no_grad():
    output = model(image.unsqueeze(0))
    _, predicted = torch.max(output.data, 1)

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Predicted: {predicted.item()}, Actual: {label}')
plt.show()
