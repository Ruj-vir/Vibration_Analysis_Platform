import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from scipy.stats import kurtosis, skew

# --- ฟังก์ชันสกัด Feature จากเนื้อหาในไฟล์ RTF ---
def extract_features_from_content(content):
    try:
        data_start = content.find('---------')
        if data_start == -1: return None
        raw_text = content[data_start:]
        numbers = re.findall(r'-?\d*\.\d+|-?\d+', str(raw_text))
        if len(numbers) < 2: return None
        # ดึงเฉพาะค่า Amplitude
        amplitudes = np.array([float(numbers[i]) for i in range(1, len(numbers), 2)])
        
        # คำนวณค่าสถิติ
        rms = np.sqrt(np.mean(amplitudes**2))
        peak = np.max(np.abs(amplitudes))
        kurt = kurtosis(amplitudes, fisher=False)
        skw = skew(amplitudes)
        crest = peak / rms if rms != 0 else 0
        
        return pd.DataFrame([{
            'RMS': rms, 'Peak': peak, 'Kurtosis': kurt, 
            'Skewness': skw, 'Crest_Factor': crest
        }])
    except:
        return None

# --- ส่วนหน้าเว็บ Streamlit ---
st.set_page_config(page_title="Bearing AI Predictor", layout="wide")
st.title("⚙️ Bearing Health Automated Analysis")

# 1. โหลด Model
@st.cache_resource
def load_assets():
    return joblib.load('bearing_model.pkl')

data = load_assets()
model = data['model']
features_list = data['features']

# 2. ส่วน Upload ไฟล์
st.header("1. Upload Vibration Data (.rtf)")
uploaded_file = st.file_uploader("ลากไฟล์ RTF มาวางที่นี่", type=['rtf'])

if uploaded_file is not None:
    # อ่านเนื้อหาไฟล์
    content = uploaded_file.read().decode('latin-1')
    df_input = extract_features_from_content(content)
    
    if df_input is not None:
        st.success(f"สกัดข้อมูลสำเร็จจากไฟล์: {uploaded_file.name}")
        
        # แสดงค่าที่สกัดได้
        st.subheader("📊 Extracted Features")
        st.dataframe(df_input.style.highlight_max(axis=0))

        # 3. ทำนายผล
        st.header("2. AI Prediction")
        prediction = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)
        
        # แสดงสถานะด้วยสี
        cols = st.columns(2)
        with cols[0]:
            if prediction == 1:
                st.balloons()
                st.success("✅ Severity Level 1: ปกติ (Normal)")
            elif prediction == 2:
                st.warning("⚠️ Severity Level 2: เริ่มผิดปกติ (Warning)")
            elif prediction == 3:
                st.error("🚨 Severity Level 3: อันตราย (Alarm)")
            else:
                st.error("💀 Severity Level 4: วิกฤต (Critical)")

        with cols[1]:
            st.write("AI Confidence (%)")
            prob_df = pd.DataFrame(prob, columns=model.classes_).T
            st.bar_chart(prob_df)

        # 4. แสดง Feature Importance (ความสำคัญของตัวแปร)
        st.header("3. Model Insights")
        st.write("กราฟนี้บอกว่า AI ให้น้ำหนักกับค่าใดมากที่สุดในการตัดสินใจครั้งนี้")
        importance = pd.Series(model.feature_importances_, index=features_list).sort_values()
        st.bar_chart(importance)
        
    else:
        st.error("ไม่สามารถอ่านข้อมูลจากไฟล์นี้ได้ กรุณาตรวจสอบรูปแบบไฟล์ RTF")