"""
import numpy as np

# x 값과 y값
x = np.array( [1, 2, 3 ] )
y = np.array( [3, 5, 7 ] )

# x와 y의 평균값
mx = np.mean(x)
my = np.mean(y)
print("x의 평균값:", mx)
print("y의 평균값:", my)

# 기울기 공식의 분모
divisor = sum([(mx - i)**2 for i in x])

# 기울기 공식의 분자
# d = 0
# for i in range(len(x)):
#    d += (x[i] - mx) * (y[i] - my)

# dividend = d

dividend = sum([ (xi-mx)*(yi-my) for xi, yi in zip(x, y) ] )
     
print("분모:", divisor)
print("분자:", dividend)

# 기울기와 y 절편 구하기
W = dividend / divisor
b = my - (mx*W)
     
# 출력으로 확인
print("기울기 W =", W)
print("y 절편 b =", b)

"""

"""
# 오차 
import numpy as np
# x 값과 y값
x = np.array( [1, 2, 3 ] )
y = np.array( [4, 7, 9 ] )
# 기울기 W, 절편 b
W = 2.0
b = 1.0
# x값을 주고 예측값(y_hat)을 구한다
for xi, yi in zip(x, y):
  y_hat = W * xi + b
  # 실제값(y)과 예측값(y_hat)이 오차이다
  err = yi - y_hat
  print('%.f = %.f - %.f ' % (err, yi, y_hat))

"""
"""
import numpy as np
# MSE 함수

def mse(y, y_hat):
   return ((y - y_hat) ** 2).mean()

# x 값과 y값
x = np.array([1, 2, 3 ])
y = np.array([4, 7, 9 ])
W = 2.0
b = 1.0
y_hat = []

for xi, yi in zip(x, y):
  y_hat.append( W * xi + b )

y_hat = np.array(y_hat)
mse_val = mse(y, y_hat) 
print('MSE = %.f' % (mse_val))
    
""" 

"""
#비용함수와 경사하강법
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1., 2., 3., 4., 5., 6.])
y_data = np.array([9., 12., 15., 18., 21., 24.])
 
#그래프로 나타내 봅니다.
plt.figure(figsize=(8,5))
plt.scatter(x_data, y_data)
plt.show()

#리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸어 줍니다.(인덱스를 주어 하나씩 불러와 계산이 가능해 지도록 하기 위함입니다.)
# x_data = np.array(x)
# y_data = np.array(y)
 
# 기울기 W와 절편 b의 값을 초기화 합니다.
W = 0
b = 0
 
#학습률을 정합니다.
lr = 0.05 
 
#몇 번 반복될지를 설정합니다.(0부터 세므로 원하는 반복 횟수에 +1을 해 주어야 합니다.)
epochs = 1000 
 
#경사 하강법을 시작합니다.
for i in range(epochs): # epoch 수 만큼 반복
    y_hat = W * x_data + b  #y를 구하는 식을 세웁니다
    error = y_data - y_hat  #오차를 구하는 식입니다.
    W_diff = -(1/len(x_data)) * sum(x_data * (error)) # 오차함수를 a로 미분한 값입니다. 
    b_diff = -(1/len(x_data)) * sum(error)  # 오차함수를 b로 미분한 값입니다. 
    W = W - lr * W_diff  # 학습률을 곱해 기존의 a값을 업데이트합니다.
    b = b - lr * b_diff  # 학습률을 곱해 기존의 b값을 업데이트합니다.
    if i % 100 == 0:    # 100번 반복될 때마다 현재의 a값, b값을 출력합니다.
        print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i, W, b))
      
""" 

"""
#유클리드 거리
# K최근접이웃 알고리즘
def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5

print(euclidean_distance([5, 4, 3], [1, 7, 9]))
 
"""

"""
import math

def euclidean_distance(pt1, pt2):
  return math.sqrt(sum([(d1-d2)**2 for d1, d2 in zip(pt1,pt2)]))

print(euclidean_distance([5, 4, 3], [1, 7, 9]))            

#박쥐를 분류하는 K최근접이웃 알고리즘
# Test distance function
# [x, y, type]
zoo = [
  [ 2.78 , 2.55 , 0 ],
	[ 1.46 , 2.36 , 0 ],
	[ 3.39 , 4.40 , 0 ],
	[ 1.38 , 1.85 , 0 ],
	[ 3.06 , 3.00 , 0 ],
	[ 7.62 , 2.75 , 1 ],
	[ 5.33 , 2.08 , 1 ],
	[ 6.92 , 1.77 , 1 ],
	[ 8.67 ,-0.24 , 1 ],
	[ 7.67 , 3.50 , 1 ] 
]
     

bat = [3,4]

for animal in zoo:
	d = euclidean_distance(bat, animal)
	print(d)

distances = []

bat = [3,4]

for animal in zoo:
	d = euclidean_distance( bat, animal )
	distances.append( ( animal, d ) )

distances.sort( key=lambda tup: tup[1] )

k = 3
for i in range(k):  
  print(distances[i])
  
# Locate the most similar neighbors
def get_neighbors(train, new_one, k):
	distances = []
	for t in train:
		d = euclidean_distance(new_one, t)	
		distances.append((t, d))
	distances.sort(key=lambda tup: tup[1])
	neighbors = []
	for i in range(k):
		neighbors.append(distances[i][0])
	return neighbors

k = 3
neighbors = get_neighbors(zoo, bat, k)
for neighbor in neighbors:
	print(neighbor)

""" 
""" 
####K-최근접이웃 알고리즘을 이용한 회귀###
#1.라이브러리 임포트
import warnings
warnings.filterwarnings(action='ignore')
# warnings.filterwarnings(action='default')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as web
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
     
#2. 주가지수 데이터 가져오기     
symbol, source, start, end = 'SPY', 'yahoo', '2016-01-01', '2020-01-01'
df = web.DataReader(symbol, source, start, end)[['Open', 'High', 'Low', 'Close']]

# 또는
# df = web.DataReader('SPY', data_source='yahoo',start='2016-01-01', end='2020-01-01')[['Open', 'High', 'Low', 'Close']]

# 또는
# df = web.DataReader('SPY', data_source='yahoo',start='2016-01-01', end='2020-01-01')
# df = df[['Open', 'High', 'Low', 'Close']]
print(df.head(), df.tail())

#3. 예측변수 설정
df = df.dropna()
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low
X = df[['Open-Close', 'High-Low']]
print(X.head())

#4.목적변수 설정
Y = np.where( df['Close'].shift(-1) > df['Close'],1,-1)
print(Y)

#5. 데이터셋 분할
train_pct = 0.7

split = int( train_pct*len(df) )

X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]

# 위의 코드는 간략하게
#X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

#6. KNN모델 설정
knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(X_train, Y_train)
accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

print('Train data Accuracy: %.2f' % accuracy_train)
print('Test  data Accuracy: %.2f' % accuracy_test)

#7. 모델을 바탕으로 전략 실행
df['Signal'] = knn.predict(X)
df['SPY_Returns'] = np.log(df['Close']/df['Close'].shift(1))
Cum_SPY_Returns = df[split:]['SPY_Returns'].cumsum()*100

df['STR_Returns'] = df['SPY_Returns']*df['Signal'].shift(1)
Cum_STR_Returns = df[split:]['STR_Returns'].cumsum()*100

plt.figure(figsize=(10,5))
plt.plot(Cum_SPY_Returns, color='r', label='SPY Returns')
plt.plot(Cum_STR_Returns, color='b', label='Strategy Returns')
plt.legend()
plt.show()

#8. 샤프비율 계산
Std = Cum_STR_Returns.std()
Sharpe = (Cum_STR_Returns - Cum_SPY_Returns)/Std
Sharpe = Sharpe.mean()
print('Sharpe ratio: %.2f' % Sharpe)

""" 
###4.로지스틱 회귀 ###
import matplotlib.pyplot as plt
import numpy as np
import math
     

def sigmoid(x):
    v = []
    for item in x:
        v.append(1/(1+math.exp(-item)))
    return v
     

x = np.arange(-10., 10., 0.2)
sig = sigmoid(x)
     

plt.plot(x,sig)
plt.show()
     
#1. 라이브러리 임포트
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score

#2. 데이터 가져오기
symbol, source, start, end = 'SPY', 'yahoo', '2016-01-01', '2020-01-01'
df = web.DataReader(symbol, source, start, end)[['Open', 'High', 'Low', 'Close']]
df = df.dropna()

#3. 예측변수/독립변수 설정
df['MA10'] = df['Close'].rolling(window=10).mean()
df['Corr'] = df['Close'].rolling(window=10).corr(df['MA10'])

df['OpenClose'] = df['Open'] - df['Close'].shift(1)
df['OpenOpen'] = df['Open'] - df['Open'].shift(1)
df = df.dropna()
X = df[['Open', 'High', 'Low', 'Close', 'MA10', 'Corr', 'OpenClose', 'OpenOpen']]

#4. 목표변수/종속변수 설정
y = np.where(df['Close'].shift(-1) > df['Close'],1,-1)

#5. 데이터넷 분할
train_pct = 0.7
split = int(train_pct * len(df))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

#6. 로지스틱 회귀모델 설정 및 훈련
model = LogisticRegression()
model = model.fit(X_train, y_train)
pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))

#7. 클래스 확률 예측
len(X_test)

probability = model.predict_proba(X_test)
print(probability)
predicted = model.predict(X_test)
print(predicted)

#8. 모델 평가
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))

print(model.score(X_test,y_test))
     
cross_val = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print(cross_val)
print(cross_val.mean())

#9. 매매전략
df['Signal'] = model.predict(X)
df['SPY_returns'] = np.log(df['Close']/df['Close'].shift(1))
Cumulative_SPY_returns = np.cumsum(df[split:]['SPY_returns'])

df['STR_returns'] = df['SPY_returns']* df['Signal'].shift(1)
Cumulative_STR_returns = np.cumsum(df[split:]['STR_returns'])

plt.figure(figsize=(10,5))
plt.plot(Cumulative_SPY_returns, color='r',label = 'SPY Returns')
plt.plot(Cumulative_STR_returns, color='b', label = 'Strategy Returns')
plt.legend()
plt.show()




     