import numpy as np
import pandas
import pandas as pd

data = './life_expectancy.csv'
life = pd.read_csv(data)


life = life.select_dtypes(include=['number'])
print (life[['Schooling', 'Income composition of resources', 'Adult mortality','HIV/AIDS', 'Thinness 1-19 years' ]].isna().sum())

# 결측치 제거
life.dropna(inplace=True)
X = life[['Schooling', 'Income composition of resources', 'Adult mortality','HIV/AIDS', 'Thinness 1-19 years' ]]
y = life['Life expectancy']

# 모델 분석
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print ('선형 회귀 모델의 점수 = ', linear_model.score(X_test, y_test).round(3))


# 입력 데이터 변경
life2 = pd.read_csv(data)
life2.dropna(inplace=True)
y = life2['Life expectancy']
X = life2.drop(['Country', 'Year', 'Status', 'Life expectancy'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print ('선형 회귀 모델의 점수 = ', linear_model.score(X_test, y_test).round(3))

