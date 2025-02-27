from sklearn.datasets import load_iris

iris = load_iris()
iris.data[:3]     # 최초 3개 데이터의 값을 출력

print('iris 데이터의 형태:', iris.data.shape)
print('iris 데이터의 속성들:', iris.feature_names)
print('iris 데이터의 레이블:', iris.target)

import pandas as pd

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = pd.Series(iris.target)
iris_df.head()

# target을 기준으로 데이터의 개수 출력
print (iris_df['target'].value_counts())

# 붓꽃 데이터 값 출력
print (iris_df.values)

# 입력 데이터와 출력 데이터 구성
X = iris_df.iloc[:, :4]
y = iris_df.iloc[:, -1]

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def iris_dt(X, y):
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    # Create the confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Decision Tree Classification')
    plt.show()
    return metrics.accuracy_score(y_test, y_pred)


scores = iris_dt(X, y)
print('dt일 때 정확도: {0:.3f}'.format(scores))




