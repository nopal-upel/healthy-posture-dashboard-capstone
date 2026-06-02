import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# Konfigurasi awal

st.set_page_config(
    page_title="Healthy Posture Reminder Dashboard",
    layout="wide"
)

FEATURE_ORDER = [
    "total_sitting_minutes",
    "number_of_breaks",
    "avg_break_duration_minutes",
    "longest_sitting_streak_minutes",
    "fatigue_level",
    "age",
    "daily_work_hours",
    "bmi",
    "sleep_hours",
    "gender",
    "work_type",
    "fitness_level",
    "day_of_week",
    "time_of_day_dominant",
    "device_preference",
]

CAT_COLS = [
    "gender",
    "work_type",
    "fitness_level",
    "day_of_week",
    "time_of_day_dominant",
    "device_preference",
]

NUM_COLS = [col for col in FEATURE_ORDER if col not in CAT_COLS]

TARGET_COLS = ["risk_level", "risk_level_encoded"]


# Fungsi load data dan file model

@st.cache_data
def load_dataset_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_artifacts():
    """
    File yang dibutuhkan:
    - healthy_posture_model.keras
    - scaler.pkl
    - label_encoders.pkl
    - label_encoder_target.pkl
    """
    artifacts = {
        "model": None,
        "scaler": None,
        "label_encoders": None,
        "target_encoder": None,
        "error": None,
    }

    try:
        artifacts["model"] = tf.keras.models.load_model(
            "healthy_posture_model.keras",
            compile=False
        )

        with open("scaler.pkl", "rb") as f:
            artifacts["scaler"] = pickle.load(f)

        with open("label_encoders.pkl", "rb") as f:
            artifacts["label_encoders"] = pickle.load(f)

        with open("label_encoder_target.pkl", "rb") as f:
            artifacts["target_encoder"] = pickle.load(f)

    except Exception as e:
        artifacts["error"] = str(e)

    return artifacts


def find_dataset_file():
    candidates = [
        "healthy_posture_modeling.csv",
        "healthy_posture_clean.csv",
        "healthy_posture_dataset.csv",
    ]

    for file in candidates:
        if os.path.exists(file):
            return file

    return None


def get_target_column(df: pd.DataFrame):
    for col in TARGET_COLS:
        if col in df.columns:
            return col
    return None


def decode_risk_labels(series: pd.Series, target_encoder=None):
    """
    Mengubah risk_level_encoded menjadi label Low/Medium/Tinggi jika memungkinkan.
    """
    if series.name == "risk_level":
        return series.astype(str)

    if target_encoder is not None:
        try:
            return pd.Series(
                target_encoder.inverse_transform(series.astype(int)),
                index=series.index,
                name="risk_level"
            )
        except Exception:
            pass

    default_map = {0: "Low", 1: "Medium", 2: "Tinggi"}
    return series.map(default_map).fillna(series.astype(str))


def encode_input(input_data: dict, label_encoders: dict) -> pd.DataFrame:
    """
    Encode fitur kategorikal sesuai LabelEncoder dari notebook.
    """
    encoded = input_data.copy()

    for col in CAT_COLS:
        value = encoded[col]
        encoder = label_encoders[col]

        if value not in encoder.classes_:
            raise ValueError(
                f"Nilai {value} tidak valid untuk kolom {col}. "
                f"Pilihan valid: {list(encoder.classes_)}"
            )

        encoded[col] = int(encoder.transform([value])[0])

    return pd.DataFrame([encoded])[FEATURE_ORDER]


def predict_risk(input_data: dict, artifacts: dict):
    input_df = encode_input(input_data, artifacts["label_encoders"])
    input_scaled = artifacts["scaler"].transform(input_df).astype("float32")

    proba = artifacts["model"](input_scaled, training=False).numpy()[0]
    pred_idx = int(np.argmax(proba))

    target_encoder = artifacts["target_encoder"]
    pred_label = str(target_encoder.inverse_transform([pred_idx])[0])
    confidence = float(np.max(proba))

    proba_df = pd.DataFrame({
        "Risk Level": target_encoder.classes_,
        "Probabilitas": proba
    }).sort_values("Probabilitas", ascending=False)

    return pred_label, confidence, proba_df


# Header

st.title("Healthy Posture Reminder Dashboard")
st.write(
    "Dashboard interaktif untuk menampilkan insight, visualisasi data, "
    "dan prediksi tingkat risiko postur pengguna berdasarkan pola duduk harian."
)


# Load dataset

artifacts = load_artifacts()
dataset_path = find_dataset_file()

with st.sidebar:
    st.header("Pengaturan Data")

    uploaded_file = st.file_uploader(
        "Upload dataset CSV jika file belum tersedia",
        type=["csv"]
    )

    st.caption("File utama yang direkomendasikan: healthy_posture_modeling.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    data_source = "File upload"
elif dataset_path is not None:
    df = load_dataset_from_path(dataset_path)
    data_source = dataset_path
else:
    st.error(
        "Dataset belum ditemukan. Upload salah satu file CSV berikut: "
        "healthy_posture_modeling.csv, healthy_posture_clean.csv, atau healthy_posture_dataset.csv."
    )
    st.stop()

target_col = get_target_column(df)

if target_col is None:
    st.warning(
        "Kolom target risk_level/risk_level_encoded belum ditemukan. "
        "Dashboard tetap bisa menampilkan preview data, tetapi visualisasi risk level tidak tersedia."
    )


# Ringkasan utama

st.subheader("Ringkasan Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Jumlah Data", f"{df.shape[0]:,}")

with col2:
    st.metric("Jumlah Kolom", df.shape[1])

with col3:
    st.metric("Jumlah Fitur Model", len([c for c in FEATURE_ORDER if c in df.columns]))

with col4:
    if target_col:
        st.metric("Target", target_col)
    else:
        st.metric("Target", "Belum tersedia")

st.caption(f"Sumber data: {data_source}")


# Tabs

tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset",
    "Visualisasi",
    "Prediksi Risk Level",
    "Insight & Kesimpulan"
])

# Tab 1 - Dataset

with tab1:
    st.subheader("Preview Dataset")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Informasi Kolom")
    info_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": [str(df[col].dtype) for col in df.columns],
        "Missing Values": [int(df[col].isna().sum()) for col in df.columns],
        "Jumlah Nilai Unik": [int(df[col].nunique()) for col in df.columns],
    })
    st.dataframe(info_df, use_container_width=True)

    if target_col:
        st.subheader("Distribusi Risk Level")
        risk_label = decode_risk_labels(
            df[target_col],
            artifacts["target_encoder"] if artifacts["target_encoder"] is not None else None
        )
        risk_counts = risk_label.value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Jumlah Data"]
        st.dataframe(risk_counts, use_container_width=True)


# Tab 2 - Visualisasi

with tab2:
    st.subheader("Visualisasi Data")

    if target_col:
        risk_label = decode_risk_labels(
            df[target_col],
            artifacts["target_encoder"] if artifacts["target_encoder"] is not None else None
        )

        risk_counts = risk_label.value_counts()

        fig, ax = plt.subplots()
        ax.bar(risk_counts.index.astype(str), risk_counts.values)
        ax.set_title("Distribusi Risk Level")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Jumlah Data")
        st.pyplot(fig)

    available_numeric = [col for col in NUM_COLS if col in df.columns]

    if available_numeric:
        selected_numeric = st.selectbox(
            "Pilih fitur numerik untuk divisualisasikan",
            available_numeric,
            index=available_numeric.index("total_sitting_minutes") if "total_sitting_minutes" in available_numeric else 0
        )

        fig, ax = plt.subplots()
        ax.hist(df[selected_numeric].dropna(), bins=30)
        ax.set_title(f"Distribusi {selected_numeric}")
        ax.set_xlabel(selected_numeric)
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

    if {"total_sitting_minutes", "number_of_breaks"}.issubset(df.columns):
        st.subheader("Hubungan Durasi Duduk dan Jumlah Istirahat")
        chart_df = df[["total_sitting_minutes", "number_of_breaks"]].dropna()
        st.scatter_chart(
            chart_df,
            x="total_sitting_minutes",
            y="number_of_breaks"
        )

    if len(available_numeric) >= 2:
        st.subheader("Korelasi Fitur Numerik")
        corr = df[available_numeric].corr(numeric_only=True)
        st.dataframe(corr, use_container_width=True)


# Tab 3 - Prediksi

with tab3:
    st.subheader("Prediksi Tingkat Risiko Postur")

    if artifacts["error"] is not None:
        st.warning(
            "File model/encoder/scaler belum lengkap atau belum bisa dibaca. "
            "Letakkan file berikut satu folder dengan app.py: "
            "healthy_posture_model.keras, scaler.pkl, label_encoders.pkl, label_encoder_target.pkl."
        )
        st.code(artifacts["error"])
    else:
        st.write("Masukkan data pengguna untuk memprediksi risk level.")

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                total_sitting_minutes = st.number_input("Total sitting minutes", min_value=0.0, max_value=600.0, value=300.0)
                number_of_breaks = st.number_input("Number of breaks", min_value=0, max_value=30, value=5)
                avg_break_duration_minutes = st.number_input("Average break duration", min_value=0.0, max_value=60.0, value=5.0)
                longest_sitting_streak_minutes = st.number_input("Longest sitting streak", min_value=0.0, max_value=600.0, value=120.0)
                fatigue_level = st.slider("Fatigue level", min_value=1.0, max_value=10.0, value=5.0)

            with c2:
                age = st.number_input("Age", min_value=10, max_value=100, value=22)
                daily_work_hours = st.number_input("Daily work hours", min_value=1, max_value=16, value=8)
                bmi = st.number_input("BMI", min_value=12.0, max_value=50.0, value=22.0)
                sleep_hours = st.number_input("Sleep hours", min_value=0.0, max_value=12.0, value=7.0)

            with c3:
                le_dict = artifacts["label_encoders"]

                gender = st.selectbox("Gender", list(le_dict["gender"].classes_))
                work_type = st.selectbox("Work type", list(le_dict["work_type"].classes_))
                fitness_level = st.selectbox("Fitness level", list(le_dict["fitness_level"].classes_))
                day_of_week = st.selectbox("Day of week", list(le_dict["day_of_week"].classes_))
                time_of_day_dominant = st.selectbox("Time of day dominant", list(le_dict["time_of_day_dominant"].classes_))
                device_preference = st.selectbox("Device preference", list(le_dict["device_preference"].classes_))

            submitted = st.form_submit_button("Prediksi Risk Level")

        if submitted:
            input_data = {
            "total_sitting_minutes": total_sitting_minutes,
            "number_of_breaks": number_of_breaks,
            "avg_break_duration_minutes": avg_break_duration_minutes,
            "longest_sitting_streak_minutes": longest_sitting_streak_minutes,
            "fatigue_level": fatigue_level,
            "age": age,
            "daily_work_hours": daily_work_hours,
            "bmi": bmi,
            "sleep_hours": sleep_hours,
            "gender": gender,
            "work_type": work_type,
            "fitness_level": fitness_level,
            "day_of_week": day_of_week,
            "time_of_day_dominant": time_of_day_dominant,
            "device_preference": device_preference,
    }

            try:
                with st.spinner("Sedang memproses prediksi..."):
                    pred_label, confidence, proba_df = predict_risk(input_data, artifacts)

                st.success(f"Hasil prediksi: {pred_label}")
                st.metric("Confidence", f"{confidence:.2%}")

                st.subheader("Probabilitas Tiap Kelas")
                st.dataframe(proba_df, use_container_width=True)
                st.bar_chart(proba_df.set_index("Risk Level"))

            except Exception as e:
                st.error("Prediksi gagal dijalankan.")
                st.code(str(e))


# Tab 4 - Insight & Kesimpulan

with tab4:
    st.subheader("Insight Utama")

    st.markdown(
        """
        Berdasarkan hasil analisis Data Science pada notebook:

        1. **Durasi duduk yang semakin lama cenderung meningkatkan risiko postur buruk.**
           Oleh karena itu, pengguna perlu diberi pengingat ketika sudah duduk terlalu lama.

        2. **Jumlah istirahat berperan penting dalam menjaga skor kesehatan dan postur.**
           Pengguna yang lebih sering mengambil jeda memiliki pola aktivitas yang lebih sehat.

        3. **Longest sitting streak menjadi salah satu indikator penting.**
           Duduk terlalu lama tanpa jeda dapat menjadi sinyal risiko yang kuat.

        4. **Faktor profil seperti sleep hours, BMI, fitness level, dan fatigue level**
           dapat membantu model memahami kondisi pengguna secara lebih personal.

        5. **Risk level sudah dibuat menjadi tiga kelas seimbang: Low, Medium, dan Tinggi.**
           Hal ini membantu proses training model agar tidak berat ke salah satu kelas.
        """
    )

    st.subheader("Rekomendasi Aplikasi")

    st.markdown(
        """
        - Aplikasi sebaiknya memberi notifikasi ketika pengguna duduk lebih dari 30–60 menit.
        - Pengguna dengan risk level **Tinggi** perlu diberi reminder yang lebih sering.
        - Dashboard dapat digunakan untuk memantau pola duduk, jumlah istirahat, dan risiko postur.
        - Fitur gamifikasi seperti skor postur sehat dapat meningkatkan motivasi pengguna untuk beristirahat.
        """
    )

    st.subheader("Kesimpulan")

    st.write(
        "Model dan dashboard ini dapat membantu aplikasi Healthy Posture Reminder "
        "dalam memberikan rekomendasi berbasis data. Dengan memanfaatkan pola duduk, "
        "frekuensi istirahat, durasi tidur, kondisi fisik, dan preferensi perangkat, "
        "sistem dapat memprediksi tingkat risiko postur pengguna dan memberikan "
        "intervensi yang lebih tepat."
    )
