from sklearn.linear_model import LogisticRegression


# 제공된 데이터
X = [[168,0],[166,0],[173,0],[165,0],[177,0],[163,0],[166,0],[174,0],[160,1],[164,1],[162,1],[156,1],[169,1],[158,1],[149,1],[167,1]] #밖에 있는 변수, 입력 데이터
y = [65,61,68,63,68,61,76,67,55,51,52,53,59,56,44,57] #x를 통해 알고 싶은 것, 타겟

y_binary = [1 if weight >= 60 else 0 for weight in y]

# 로지스틱 회귀
logistic_model = LogisticRegression()
logistic_model.fit(X, y_binary)

# 계수와 절편, 점수
print('계수:', logistic_model.coef_)
print('절편:', logistic_model.intercept_)
print('점수:', logistic_model.score(X,y_binary))

testX = [[167,0],[167,1]]
# 예측 확률
y_pred = logistic_model.predict_proba(testX)
print('예측 확률:', y_pred)

# 예측 결과
y_pred_logistic = logistic_model.predict(testX)
print('예측 결과:',y_pred_logistic)
