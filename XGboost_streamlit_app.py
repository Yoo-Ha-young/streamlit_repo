import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd


st.title("항공사 고객만족도 XGboost")
# 첫 번째 행
r1_col1, r1_col2, r1_col3 = st.columns(3)
Inflight_wifi_service = r1_col1.slider("Inflight wifi service", 0, 5)
Departure_Arrival_time_convenient = r1_col2.slider("Departure/Arrival time convenient", 0, 5)
Ease_of_Online_booking = r1_col3.slider("Ease of Online booking", 0,5)

# 두번째 행
r2_col1, r2_col2, r2_col3 = st.columns(3)
Gate_location = r2_col1.slider("Gate location",0,5 )
Food_and_drink = r2_col2.slider("Food and drink", 0,5)
Online_boarding = r2_col3.slider("Online boarding", 0,5)

# 세번째 행
r3_col1, r3_col2, r3_col3 = st.columns(3)
Leg_room_service=r3_col1.slider("Leg room service ",0,5)
Baggage_handling =r3_col2.slider("Baggage handling",0,5)
On_board_service =r3_col3.slider("On-board service",0,5)

# 네 번째 행
r4_col1, r4_col2, r4_col3 = st.columns(3)
Seat_comfort =r4_col1.slider("Seat comfort ",0,5)
Inflight_entertainment =r4_col2.slider("Inflight entertainment",0,5)
Check_in_service =r4_col3.slider("Check-in service",0,5)

# 다섯번째 행
r5_col1, r5_col2 = st.columns(2)
Inflight_service = r5_col1.slider("Inflight service ",0,5)
Cleanliness = r5_col2.slider("Cleanliness",0,5)
Departure_Delay_in_Minutes = r5_col2.slider("Departure Delay in Minutes",0,5)
Arrival_Delay_in_Minutes = r5_col2.slider("Arrival Delay in Minutes",0,5)


# 예측 버튼
predict_button = st.button("예측")
st.write("---")

# 예측 결과
if predict_button:
    model_from_joblib = joblib.load('xgb_model.pkl')
    pred = model_from_joblib.predict(np.array([['Inflight wifi service', 'Departure/Arrival time convenient',
       'Ease of Online booking', 'Gate location', 'Food and drink',
       'Online boarding', 'Seat comfort', 'Inflight entertainment',
       'On-board service', 'Leg room service', 'Baggage handling',
       'Checkin service', 'Inflight service', 'Cleanliness',
       'Departure Delay in Minutes', 'Arrival Delay in Minutes']]))
    st.metric("예상 만족 여부", pred[0])
