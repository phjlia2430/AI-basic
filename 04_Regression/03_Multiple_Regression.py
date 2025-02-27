import numpy as np
import pandas as pd


data = './life_expectancy.csv'
life = pd.read_csv(data)
print (life.head(3))
print ('*'*20)

# 기대수명 데이터 개요
print (life.describe())
print ('*'*20)
# 데이터 컬럼
print (life.columns)
print ('*'*20)
# 데이터 도식화
import seaborn as sns
import matplotlib.pyplot as plt

life = life.select_dtypes(include=['number'])
sns.set(rc={'figure.figsize':(22,20)})
correlation_matrix = life.corr().round(2)
sns.heatmap(data=correlation_matrix,annot=True)
plt.show()

# 기대수명과 상관관계
print (life.corr().round(3)['Life expectancy'])
print ('*'*20)

# 상관도의 절대값이 높은 상위 7개 출력
c = life.corr().round(2)['Life expectancy']
c = np.abs(c)
print (c.sort_values(ascending=False)[1:8])
print ('*'*20)

# 기대 수명과 상관도가 높은 특성을 pairplot으로 그림
sns.pairplot(life[['Life expectancy','Schooling', 'Income composition of resources', 'Adult mortality','HIV/AIDS', 'Thinness 1-19 years' ]])
plt.show()

