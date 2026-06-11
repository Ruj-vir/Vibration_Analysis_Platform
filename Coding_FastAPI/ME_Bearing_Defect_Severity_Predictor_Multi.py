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

        # กรอง inf / NaN ออกจาก amplitudes ก่อนคำนวณ
        amplitudes = amplitudes[np.isfinite(amplitudes)]
        if len(amplitudes) < 2: return None

        # คำนวณค่าสถิติ
        rms = np.sqrt(np.mean(amplitudes**2))
        peak = np.max(np.abs(amplitudes))
        kurt = kurtosis(amplitudes, fisher=False)
        skw = skew(amplitudes)
        crest = peak / rms if rms > 1e-10 else 0

        features = {
            'RMS': rms, 'Peak': peak, 'Kurtosis': kurt,
            'Skewness': skw, 'Crest_Factor': crest
        }

        # ถ้า feature ใดเป็น inf หรือ NaN ให้ return None (ไฟล์นี้ใช้ไม่ได้)
        if not all(np.isfinite(v) for v in features.values()):
            return None

        return pd.DataFrame([features])
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
uploaded_files = st.file_uploader("ลากไฟล์ RTF มาวางที่นี่ (เลือกได้หลายไฟล์)", type=['rtf'], accept_multiple_files=True)

if uploaded_files:
    all_features = []
    failed_files = []

    for uploaded_file in uploaded_files:
        content = uploaded_file.read().decode('latin-1')
        df_input = extract_features_from_content(content)
        if df_input is not None:
            df_input.insert(0, 'File', uploaded_file.name)
            all_features.append(df_input)
        else:
            failed_files.append(uploaded_file.name)

    if failed_files:
        st.error(f"ไม่สามารถอ่านข้อมูลจากไฟล์เหล่านี้ได้ (รูปแบบผิด หรือมีค่า infinity/NaN): {', '.join(failed_files)}")

    if all_features:
        df_all = pd.concat(all_features, ignore_index=True)
        st.success(f"สกัดข้อมูลสำเร็จ {len(all_features)} ไฟล์")

        # แสดงค่าที่สกัดได้
        st.subheader("📊 Extracted Features")
        st.dataframe(df_all)

        # 3. ทำนายผล
        st.header("2. AI Prediction")
        feature_cols = [c for c in df_all.columns if c != 'File']
        predictions = model.predict(df_all[feature_cols])
        probs = model.predict_proba(df_all[feature_cols])

        severity_map = {
            1: "✅ Level 1: ปกติ (Normal)",
            2: "⚠️ Level 2: เริ่มผิดปกติ (Warning)",
            3: "🚨 Level 3: อันตราย (Alarm)",
            4: "💀 Level 4: วิกฤต (Critical)",
        }

        for i, row in df_all.iterrows():
            pred = predictions[i]
            prob = probs[i]
            with st.expander(f"📄 {row['File']}  —  {severity_map.get(pred, str(pred))}"):
                if pred == 1:
                    st.success(severity_map[pred])
                    st.balloons()
                elif pred == 2:
                    st.warning(severity_map[pred])
                elif pred == 3:
                    st.error(severity_map[pred])
                else:
                    st.error(severity_map.get(pred, str(pred)))

                st.write("AI Confidence (%)")
                prob_df = pd.DataFrame([prob], columns=model.classes_).T
                st.bar_chart(prob_df)

        # 4. แสดง Feature Importance (ความสำคัญของตัวแปร)
        st.header("3. Model Insights")
        st.write("กราฟนี้บอกว่า AI ให้น้ำหนักกับค่าใดมากที่สุดในการตัดสินใจ")
        importance = pd.Series(model.feature_importances_, index=features_list).sort_values()
        st.bar_chart(importance)