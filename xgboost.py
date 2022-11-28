import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""# **2. 데이터 읽어오기 & 보기**
### 커플 성사여부 예측하기
**데이터** : Speed Dating Experiment

-----

## 변수 목록
* has_null
    - 변수 중 Null값이 있는지 여부. 단, 이 데이터는 기존 데이터에서 일부 변수들이 생략된 축소판이기 때문에, 여기서 보이는 Null값 여부와 다소 차이가 있을 수 있음.
    - 전반적으로 무응답 항목이 있는지에 대한 정보이므로 그대로 사용
* age / age_o : age는 본인 나이이며 age_o는 상대방 나이.
* race / race_o : 마찬가지로 본인과 상대의 인종 정보.
* importance_same_race / importance_same_religion
    * 인종과 종교를 중요시 여기는지에 대한 응답
* attractive(매력적인), sincere(성실한), intelligence(지적), funny(재미난), ambitious(야심찬), shared_ interests(공통관심사) : 이 항목들은 4가지 관점에서 평가되어 총 변수가 24(6 × 4)개
    * pref_o_xxx( 예 : pref_o_attractive) : 상대방이 xxx 항목을 얼마나 중요하게 생각하는가에 대한 응답
    * xxx_o(예: attractive_o) : 상대방이 본인에 대한 xxx 항목을 평가한 항목
    * xxx_important(예 : attractive_important) : xxx 항목에 대해 본인이 얼마나 중요하게 생각하는가에 대한 응답
    * xxx_partner(예 : attractive_partner) : 본인이 상대방에 대한 xxx 항목을 평가한 항목
* interests_correlate : 관심사(취미 등) 연관도
* expected_happy_with_sd_people : 스피드 데이팅을 통해 만난 사람과 함께할 때 얼마나 좋을
지에 대한 기대치
* expected_num_interested_in_me : 얼마나 많은 사람이 나에게 관심을 보일지에 대한 기대치
* like : 파트너가 마음에 들었는지 여부
* guess_prob_liked : 파트너가 나를 마음에 들어했을지에 대한 예상
* met: 파트너를 스피드 데이팅 이벤트 이전에 만난 적이 있는지 여부
"""

# https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment
file_url = 'https://raw.githubusercontent.com/bigdata-young/bigdata_16th/main/data/dating.csv'
df = pd.read_csv(file_url)

pd.options.display.max_columns = 40 # 총 40개 컬럼까지 출력되도록 설정

df.head()

df.info() # 결측치와 변수 타입

pd.options.display.float_format = '{:.2f}'.format

df.describe()

"""# **3. 데이터 전처리**

## **1) 결측치 해결**
"""

df.isna().mean().sort_values(ascending=False)

#@title # 일부 변수에서 결측치 제거

df = df.dropna(
    subset=['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
            'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests',
            'attractive_important', 'sincere_important', 'intellicence_important',
            'funny_important', 'ambtition_important', 'shared_interests_important'])

# 나머지 결측치 -99 (결측치라는 걸 나타냄)
df.fillna(-99, inplace=True)

df.isna().mean().sort_values(ascending=False)

"""## **2) 피처 엔지니어링**
- 피처 : 독립변수들 / 엔지니어링 -> 가공해서 더 유의미하게 쓰겠다는 것

- 나이? 중요도? -> 계산 -> 합쳐주거나, 새로운 변수화

### **나이 age**
* age / age_o : age는 본인 나이이며 age_o는 상대방 나이.

* 데이터에 상대방 나이와 본인 나이가 있기 때문에, 이를 토대로 계산할 수 있다.
"""

# 나이 차이 + 성별간의 차이

# apply(axis=1)
# df.age = 본인의 나이, df.age_o = 상대방 나이
def age_gap(x) : # 행 전체
    if x['age'] == -99: # 내 나이가 결측치면
        return -99 # 나이 차이도 결측치
    if x['age_o'] == -99: # 상대방 결측치면
        return -99

# 남녀 중 한 명이라도 나이가 -99이면 -99를 반환한다.


    if x['gender'] == 'female':
        return x['age_o'] - x['age'] # 상대방의 나이가 얼마나 더 많은지 (여성)
    if x['gender'] == 'male':
        return x['age'] - x['age_o'] # 내가 상대방보다 나이가 얼마나 많은지 (남성)

# 그렇지 않으면 남자가 연상이면 플러스값이, 여자가 연상이면 마이너스값이 반환된다.

"""결측치를 -99로 채워넣었으므로 단순히 나이차를 계산해서는 안되고, '알 수 없음'의 의미로 -99를 사용했다. 

또 한가지는 성별과 관련된 요인이다. 단순한 나이 차이보다는 남자가 여자보다 많은지, 반대 경우인지도 고려하는게 좋다.

여러 조건을 반영하여 계산하는 함수를 만들었다.
"""

# 정의한 age_gap() 함수 - 데이터프레임에 적용

df['age_gap'] = df.apply(age_gap, axis=1)
df.age_gap.head()

# 나이 차이만 (절대값 적용)
# 남녀 중 어느 쪽이 더 나이가 많은지와 상관없이, 나이 차이 자체가 중요한 변수가 될 수 있다.
df['age_gap_abs'] = abs(df.age_gap)
df.age_gap_abs.unique()

"""### **인종정보 race**
* race / race_o : 마찬가지로 본인과 상대의 인종 정보.

* importance_same_race
인종을 중요시 여기는지에 대한 응답
"""

# df.race, df.race_o
def same_race(x):
  if x['race'] == -99: return -99
  if x['race_o'] == -99: return -99
  if x['race'] == x['race_o']: return 1  # 같으면 1
  return -1  # 다르면 -1

df['same_race'] = df.apply(same_race, axis=1)
df.same_race.unique()

def same_race_point(x): # apply(axis=1)
    if x['same_race'] == -99: # 결측치면
        return -99 # 결측치로 두고
    # 1, -1
    return x['same_race'] * x['importance_same_race']

df['same_race_point'] = df.apply(same_race_point, axis=1)
df.same_race_point.value_counts()

df[['race', 'race_o', 'same_race', 'same_race_point', 'same_race_point']]

"""## **점수x중요도 arrtactive, sincere**

평가/중요도를 계산하여 새로운 변수를 만든다.

같은 계산을 여러 변수에 반복하므로 함수로 만든다.

결측치는 -99이고, 매개 변수는 3개이다.

첫번째 매개변수는 df = 데이터 프레임, 나머지는 중요도와 평가 변수를 받는다.
"""

# 중요도  * 점수 => 파생변수 = rating(함수)
# importance 중요도 , score 점수 
# data : 열(row)
def rating(data, importance, score): # 점수를 부여하는 함수
    if data[importance] == -99: return -99 # 결측치
    if data[score] == -99: return -99 # 결측치
    return data[importance] * data[score]  # 중요도와 그것에 대한 점수 곱을 리턴한다.

#df.columns  중요도 평가에 대한 변수 확인하기
print(f"상대방의 선호도 : {df.columns[8:14]}")
print(f"본인에 대한 상대방의 평가 : {df.columns[14:20]}")
print(f"본인의 선호도 : {df.columns[20:26]}")
print(f"상대방에 대한 본인의 평가 : {df.columns[26:32]}")

#@title #### ***중요도와 평가에 대한 변수 이름 리스트 생성**
partner_imp = df.columns[8:14]
partner_rate_me = df.columns[14:20]
my_imp = df.columns[20:26]
my_rate_partner = df.columns[26:32]

#@markdown ####**본인관련 새 변수, 상대방 관련 새 변수**

new_label_partner = ['attrative_p','sincere_partner_p','intelligence_p',
                     'funny_p','ambition_p','shared_interests_p']
new_label_me = ['attrative_m','sincere_partner_m','intelligence_m',
                     'funny_m','ambition_m','shared_interests_m']

#@markdown **새변수 이름 = 중요도 변수 이름X 평가 변수이름** <br> # 파트너가 나에게 느끼는 점수 / 상대방의 선호도 / 나에 대한 파트너의 평가

# 평가 점수 x 중요도 => 새로운 라벨

for i, j, k in zip(new_label_partner, partner_imp, partner_rate_me):
    print(f"{i} & {j} & {k}")

#@title df 전체에 대해 apply(), lambda 함수로 rating() 함수 사용
#@markdown 매개변수 : 각각 x = 데이터프레인 , j : 중요도 변수 이름, k = 평가 변수 
#@markdown 

# 파트너가 나에게 느끼는 점수 / 상대방의 선호도 / 나에 대한 파트너의 평가
for i, j, k in zip(new_label_partner, partner_imp, partner_rate_me):    # 순회
    # i => new_label_partner (새로운 컬럼의 이름) -> df[i]
    # j = 상대방의 특정 영역에 대한 선호도 (importance)
    # k = 나에 대한 파트너의 평가 (score) 
    df[i] = df.apply(lambda x: rating(x, j, k), axis=1)

# 파트너가 나에게 느끼는 점수 / 상대방의 선호도 / 나에 대한 파트너의 평가
for i, j, k in zip(new_label_me, my_imp, my_rate_partner):
    # i => new_label_me (새로운 컬럼의 이름) -> df[i]
    # j = 나의 상대방 특정 영역에 대한 선호도 (importance)
    # k = 상대방에 대한 나의 평가 (score)
    df[i] = df.apply(lambda x: rating(x, j, k), axis=1)

df.columns

"""### **3) 범주형 변수를 더미 변수로 변환**"""

df.info()

df.describe(include=['O'])

#@title #### **더미 변수로 변환**
df = pd.get_dummies(df, columns=['gender', 'race', 'race_o'], drop_first=True)
df.info()

"""# **4. 모델링**"""

#@title 훈련셋/시험셋 나누기
from sklearn.model_selection import train_test_split
X = df.drop('match', axis=1)
y = df.match
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=100)

#@title xgboost 모델링
import xgboost as xgb

# 모델 객채 생성
#@markdown n_estimators = 500, max_depth = 5, random_state = 100)
model = xgb.XGBClassifier(n_estimators = 500, max_depth = 5, random_state = 100)

# 모델 학습(훈련)
model.fit(X_train, y_train)

"""# **5. 모델 예측 및 평가**"""

# 예측
pred = model.predict(X_test)

# 정확도 accuracy_score, 혼동행렬 confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 매칭 여부
# df.match.mean()
1 - df.match.mean()

#정확도 : 예측 결과 전체 중 정확하게 예측한 비율
accuracy_score(y_test, pred)

# 혼동행렬 
# TN 1297  : 실제 부정 데이터를 부정으로 제대로 예측
# FP 68 : 실제 부정(0) 데이터를 긍정(1)으로 잘못 예측 1종 오류
# FN 147 : 실제 긍정 데이터를 부정으로 잘못 예측
# TP 114 : 실제 긍정(1) 데이터를 긍정(0)으로 제대로 예측 2종 오류
confusion_matrix(y_test, pred)

print(classification_report(y_test, pred))

# 평가 수치표
# precision 정밀도 / recall 재현율 / f1-score / support 인덱스에 해당하는 개수
print(classification_report(y_test, pred))

"""# **6. 하이퍼 파라미터 튜닝**

### 그리드 서치
그리드서치는 한번 시도로 수백 가지의 하이퍼파라미터값을 시도해볼 수 있다.

그리드 서치 원리 : 입력할 하이퍼파라미터 후보들을 입력하면, 각 조합에 대해 모두 모델링 해보고 최적의 결과가 나오는 하이퍼파라미터 조합을 알려준다..


---

**경사하강법 (여러개를 시도해보는 것 : 근사값 찾기)**
* 머신러닝이 학습시킬 때 최소의 오차를 찾는 방법
* 오차 함수에 대한 경사도(미분계수)를 기준으로 매개변수를 반복적으로 이동해가며 최소 오차를 찾는다.
"""

max_depth = [3,5,10]  # 트리의 깊이 (오버피팅)
learning_rate = [0.01, 0.05, 0.1]  # 경사하강법 : '매개변수' -> 최소오차 -> 보폭 크기

# k-fold(5)
# 45개

parameter = {
    'learning_rate': [0.01, 0.1, 0.3], # 경사하강법 : '매개변수' -> 최소오차 -> 보폭 크기
    'max_depth': [5, 7, 10], # 트리의 깊이 (오버피팅)
    'subsample': [0.5, 0.7, 1], # 추출할 데이터 비율
    'n_estimators': [300, 500, 1000] # 트리 개수
}

# 그리드 서치 모듈 - 사이킷런 라이브러리
from sklearn.model_selection import GridSearchCV

#model = xgb.XGBClassifier()
#gs_model = GridSearchCV(model, parameter, n_jobs=-1, scoring='f1', cv = 5)

#모델 학습
#gs_model.fit(X_train, y_train)

# parameter = {
#     'learning_rate': [0.01],
#     'max_depth': [5],
#     'subsample': [0.5],
#     'n_estimators': [300]
# }

# gs_model = GridSearchCV(model, parameter, n_jobs=-1, scoring='f1', cv = 5)
# gs_model.fit(X_train, y_train)

# 최적의 조합 확인
# gs_model.best_params_

# pred = gs_model.predict(X_test)

# accuracy_score(y_test, pred)

"""# **7. 변수의 영향력**"""

#@title 변수의 영향력 -> 중요 변수 확인
model = xgb.XGBClassifier(learning_rate=0.3, max_depth=5, 
                          n_estimators= 1000, subsample=0.5, random_state=100)
model.fit(X_train, y_train)

# 중요한 변수
model.feature_importances_

feature_imp = pd.DataFrame({'features': X_train.columns, 'values': model.feature_importances_})

#values :  변수 중요도
pd.options.display.float_format = '{:.6f}'.format
feature_imp.head()

feature_imp.sort_values(by='values', ascending=False)

feature_imp = pd.DataFrame({'features': X_train.columns, 'values': model.feature_importances_})

pd.options.display.float_format = '{:.6f}'.format
feature_imp.head()

feature_imp.sort_values(by='values', ascending=False)

plt.figure(figsize=(20, 10))  # 그래프 크기 설정
sns.barplot(x='values', y='features',
            data=feature_imp.sort_values(by='values', ascending=False).head(10))

