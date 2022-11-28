import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Personal Data 레이블 인코딩 (범주형 데이터를 연속형 변수로 바꿈)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# XGBoost 에 필요한 라이브러리 임포트
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

# 훈련셋
airline_train = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/train.csv", index_col=0)
airline_train.drop(['id'], axis=1, inplace=True)

# 시험셋
airline_test = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/test.csv", index_col=0)
airline_test.drop(['id'], axis=1, inplace=True)

# 결측치 제거
airline_train.dropna(inplace=True)
airline_test.dropna(inplace=True)

# 레이블 인코딩
# 범주형 columns 리스트 : temp_list
temp_list = ["satisfaction", "Gender", "Customer Type", "Type of Travel", "Class"]

for i in temp_list:
  # 훈련셋
  encoder.fit(airline_train[i])
  airline_train['{}_score'.format(i)] = encoder.transform(airline_train[i])
  # 시험셋
  encoder.fit(airline_test[i])
  airline_test['{}_score'.format(i)] = encoder.transform(airline_test[i])

# 고객 평가 지표 데이터 프레임
airline_score = airline_train[['Inflight wifi service', 'Departure/Arrival time convenient',
       'Ease of Online booking', 'Gate location', 'Food and drink',
       'Online boarding', 'Seat comfort', 'Inflight entertainment',
       'On-board service', 'Leg room service', 'Baggage handling',
       'Checkin service', 'Inflight service', 'Cleanliness',
       'Departure Delay in Minutes', 'Arrival Delay in Minutes',
       'satisfaction_score']]


# y, X 정의
X_train = airline_train[['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 
'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service',
'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 
'Arrival Delay in Minutes']]   # 여기에 특성 추가
X_test = airline_test[['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 
'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service',
'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 
'Arrival Delay in Minutes']]
y_train = airline_train['satisfaction_score']
y_test = airline_test['satisfaction_score']

X_train.astype(int)
X_test.astype(int)
y_train.astype(int)
y_test.astype(int)

# 평가지표 함수 정의
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

def evaluation(airline_test, pred):
    acc = accuracy_score(airline_test, pred)
    pre = precision_score(airline_test, pred)
    rec = recall_score(airline_test, pred)
    f1 = f1_score(airline_test, pred)
    roc = roc_auc_score(airline_test, pred)
    cf_matrix = confusion_matrix(airline_test, pred)
    print("정확도: {0:.4f}".format(acc))
    print("정밀도: {0:.4f}".format(pre))
    print("재현율: {0:.4f}".format(rec))
    print("f1 score: {0:.4f}".format(f1))
    print("roc_auc_score: {0:.4f}".format(roc))
    group_names = ['TN','FP','FN','TP']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='coolwarm')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()
       
# 그리드서치 라이브러리 임포트
from sklearn.model_selection import GridSearchCV

def xgb_tuning(train_set, test_set, parameters):
    model = XGBClassifier()
    grid = GridSearchCV(model, parameters, scoring="roc_auc", cv=5, n_jobs=-1, refit = True) # cv=K-fold
    grid.fit(train_set, test_set)
    pred= grid.predict(X_test)
    return grid.best_params_, grid.best_score_, pred


feature_score = pd.DataFrame({'features': X_train.columns, 'values': xgb_model.feature_importances_})
feature_score.head()

plt.figure(figsize=(20,10))
palette = sns.color_palette('coolwarm',10)
sns.barplot(x='values', y='features', data=feature_score.sort_values(by='values', ascending=False).head(10), palette=palette)

xgb_model = XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.3, subsample=1)
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)

joblib.dump(xgb_model, './xgb_model.pkl')