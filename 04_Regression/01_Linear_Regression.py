import numpy as np
from sklearn.linear_model import LinearRegression

linear_model= LinearRegression()


#키의 변화에 따른 몸무게 변화
X = [[168,0],[166,0],[173,0],[165,0],[177,0],[163,0],[166,0],[174,0],[160,1],[164,1],[162,1],[153,1],[157,1],[158,1],[150,1],[167,1]] #밖에 있는 변수, 입력 데이터
y = [65,61,68,63,68,61,76,67,55,51,59,53,61,56,44,57] #x를 통해 알고 싶은 것, 타겟
linear_model.fit(X, y)

coef = linear_model.coef_
intercept = linear_model.intercept_
score=linear_model.score(X, y)

print ("y = {}*X + {:.2f}".format(coef.round(2), intercept))
print ("데이터와 선형 회귀 직선의 관계점수 :  {:.1%}".format(score))

'''
import matplotlib.pyplot as plt

plt.scatter(X, y, color='blue', marker='D')
y_pred = linear_model.predict(X)
plt.plot(X, y_pred, 'r:')
plt.show()'''

unseen = [[167]]
result = linear_model.predict(unseen)
print ("키 {}cm는 몸무게 {}kg으로 추정됨".format(unseen, result.round(1)))

