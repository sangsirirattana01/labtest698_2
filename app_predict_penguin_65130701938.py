
import streamlit as st
import pickle
import numpy as np

# โหลดโมเดลและ encoders
with open('model_penguin_65130701938.pkl', 'rb') as file:
    model = pickle.load(file)

with open('species_encoder.pkl', 'rb') as file:
    species_encoder = pickle.load(file)

with open('island_encoder.pkl', 'rb') as file:
    island_encoder = pickle.load(file)

with open('sex_encoder.pkl', 'rb') as file:
    sex_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ชื่อแอป
st.title("Penguin Species Prediction App")

# ฟอร์มกรอกข้อมูล
st.header("Input Features")

# รับค่าจากผู้ใช้
island = st.selectbox("Island", island_encoder.classes_)
culmen_length_mm = st.number_input("Culmen Length (mm)", min_value=20.0, max_value=70.0, step=0.1)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", min_value=10.0, max_value=30.0, step=0.1)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=150.0, max_value=250.0, step=0.1)
body_mass_g = st.number_input("Body Mass (g)", min_value=2500.0, max_value=6500.0, step=0.1)
sex = st.selectbox("Sex", sex_encoder.classes_)

# แปลงข้อมูลจากผู้ใช้
if st.button("Predict Species"):
    # แปลงข้อมูล categorical ด้วย encoders
    island_encoded = island_encoder.transform([island])[0]
    sex_encoded = sex_encoder.transform([sex])[0]

    # รวมข้อมูลทั้งหมด
    input_data = np.array([[island_encoded, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, sex_encoded]])

    # Scale ข้อมูล
    input_data_scaled = scaler.transform(input_data)

    # ทำนายผล
    prediction = model.predict(input_data_scaled)
    species = species_encoder.inverse_transform(prediction)

    # แสดงผลลัพธ์
    st.subheader("Predicted Species")
    st.write(f"The predicted species is: **{species[0]}**")

