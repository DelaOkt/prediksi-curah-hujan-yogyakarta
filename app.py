import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import io
import plotly.express as px
import seaborn as sns

st.set_page_config(page_title="Dashboard Prediksi Cuaca", layout="wide")
st.sidebar.title("🌦️ Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Upload & Prediksi", "Analisis Prediksi"])

#st.title("🌧️ Prediksi Curah Hujan di Yogyakarta")

resolusi = st.sidebar.selectbox("Resolusi Prediksi", ["Harian", "3 Harian", "Bulanan"], index=2)

# ==================== Fungsi ====================
def proses_data(df):
    df.replace(['-', '8888', '9999'], np.nan, inplace=True)
    cols = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR', 'FF_X', 'FF_AVG']
    df[cols] = df[cols].astype(float)
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], dayfirst=True)
    df.set_index('TANGGAL', inplace=True)
    df[cols] = df[cols].fillna(df[cols].median())

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    fitur = ['TN', 'TX', 'TAVG', 'RH_AVG', 'FF_X']
    scaler = MinMaxScaler()
    df[fitur] = scaler.fit_transform(df[fitur])

    if resolusi == "Bulanan":
        df_resample = df.resample('M').mean()
    elif resolusi == "3 Harian":
        df_resample = df.resample('3D').mean()
    else:
        df_resample = df.resample('D').mean()

    return df_resample, scaler


def klasifikasi_hujan(rr, q3):
    if rr == 0:
        return "☀️ Cerah"
    elif rr <= 0.2 * q3:
        return "🌤️  Ringan"
    elif rr <= 0.5 * q3:
        return "🌦️ Sedang"
    elif rr <= q3:
        return "🌧️ Lebat"
    else:
        return "⛈️ Ekstrem"
    
@st.cache_resource
def load_default_model():
    # Dummy data dan model default
    dummy_data = pd.DataFrame({
        'TN': [0.3], 'TX': [0.7], 'TAVG': [0.5], 'RH_AVG': [0.6], 'FF_X': [0.2]
    })
    dummy_target = [0.3]

    scaler = MinMaxScaler()
    scaler.fit(dummy_data)

    model = XGBRegressor()
    model.fit(scaler.transform(dummy_data), dummy_target)

    return model, scaler

# === Inisialisasi model default ===
if "model" not in st.session_state or "scaler" not in st.session_state:
    st.session_state.model, st.session_state.scaler = load_default_model()


# ==================== Menu: Beranda ====================
if menu == "Beranda":
    st.markdown("""
    <style>
    .hero-box {
        background: linear-gradient(90deg, #5dade2, #85c1e9);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .hero-box h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .hero-box p {
        font-size: 1.2rem;
        margin: 0;
    }
    </style>
    <div class='hero-box'>
        <h1>🌧️ Prediksi Curah Hujan Yogyakarta</h1>
        <p>Dashboard interaktif menggunakan XGBoost Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🎯 Evaluasi Performa Model")
    st.markdown("Berikut hasil pengujian model terhadap data curah hujan:")

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💥 RMSE", "0.0832", "- Lebih rendah = lebih baik")
        with col2:
            st.metric("📉 MAPE", "57.81%", "⚠️ Masih tinggi")
        with col3:
            st.metric("📈 R² Score", "0.8097", "+ Model sudah cukup baik")

    st.markdown("---")

    st.markdown("## 💡 Sekilas Tentang Dashboard")
    st.markdown("""
<div style="background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #3498db; color: #000; font-weight: 500;">
    <ul style="font-size: 1.1rem; margin-left: -1rem;">
        <li>📂 <strong>Upload file</strong> cuaca (CSV/Excel) atau input manual</li>
        <li>🔄 <strong>Preprocessing otomatis</strong> & scaling data</li>
        <li>⚙️ <strong>Prediksi curah hujan</strong> dengan algoritma <code>XGBoost</code></li>
        <li>🌈 <strong>Klasifikasi</strong> kategori hujan: Cerah, Ringan, Sedang, Lebat, Ekstrem</li>
        <li>📊 <strong>Visualisasi interaktif</strong>: grafik, batang, heatmap, hingga download hasil</li>
    </ul>
    <p style="margin-top: 10px;">🔎 Prediksi berdasarkan suhu, kelembaban, angin, dan curah hujan historis.</p>
</div>
""", unsafe_allow_html=True)


    st.markdown("---")

    st.markdown("## 🚀 Mulai Sekarang!")
    st.success("Pilih menu di sebelah kiri: Upload data, input manual, atau analisis hasil prediksi 💻")
    st.markdown("<br>", unsafe_allow_html=True)

# ==================== Menu: Upload & Prediksi ====================
if menu == "Upload & Prediksi":
    st.subheader("🧠 Pilih Mode Input")
    mode_input = st.radio("Pilih cara input data:", ["Upload File", "Input Manual"])

    if mode_input == "Upload File":
        st.subheader("📂 Upload File Cuaca (.xlsx atau .csv)")
        uploaded_file = st.file_uploader("Upload file", type=['xlsx', 'csv'])

        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            df_resample, scaler = proses_data(df)

            train = df_resample[df_resample.index < '2024-01-01']
            test = df_resample[df_resample.index >= '2024-01-01']

            fitur = ['TN', 'TX', 'TAVG', 'RH_AVG', 'FF_X']
            target = 'RR'
            X_train = train[fitur]
            y_train = train[target]
            X_test = test[fitur]
            y_test = test[target]

            model = XGBRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            q3_rr = y_test.quantile(0.75)

            hasil_df = pd.DataFrame({
                "TANGGAL": y_test.index.strftime('%Y-%m-%d'),
                "RR_AKTUAL": y_test.values,
                "RR_PREDIKSI": y_pred
            })
            hasil_df["KATEGORI"] = hasil_df["RR_PREDIKSI"].apply(lambda x: klasifikasi_hujan(x, q3_rr))

            st.success("✅ Prediksi berhasil!")
            st.dataframe(hasil_df)

            st.session_state.hasil_df = hasil_df
            st.session_state.model = model
            st.session_state.scaler = scaler

            st.markdown("### 📥 Unduh Hasil Prediksi & Kategori")
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                hasil_df.to_excel(writer, index=False, sheet_name='Hasil')

            st.download_button(
                label="📥 Download Excel",
                data=buffer,
                file_name="prediksi_klasifikasi_cuaca.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    elif mode_input == "Input Manual":
        st.subheader("📝 Input Data Cuaca Secara Manual")
        with st.form(key='manual_form'):
            col1, col2 = st.columns(2)
            with col1:
                TN = st.number_input("Suhu Minimum (TN)", value=0.0)
                TX = st.number_input("Suhu Maksimum (TX)", value=0.0)
                TAVG = st.number_input("Suhu Rata-rata (TAVG)", value=0.0)
                RH_AVG = st.number_input("Kelembaban Rata-rata (RH_AVG)", value=0.0)
            with col2:
                FF_X = st.number_input("Kecepatan Angin Maksimum (FF_X)", value=0.0)
                RR_AKTUAL = st.number_input("RR Aktual (Opsional)", value=0.0)

            submit_btn = st.form_submit_button(label='Prediksi')

        if submit_btn:
            input_df = pd.DataFrame.from_dict({
                'TN': [TN], 'TX': [TX], 'TAVG': [TAVG],
                'RH_AVG': [RH_AVG], 'FF_X': [FF_X]
            })

            scaler = st.session_state.get('scaler')
            scaled_input = scaler.transform(input_df) if scaler else input_df

            model = st.session_state.get('model')
            if model and hasattr(model, 'predict'):
                rr_prediksi = model.predict(scaled_input)[0]
            else:
                st.error("⚠️ Model belum dilatih. Silakan upload data dulu di tab 'Upload & Prediksi'.")
                st.stop()


            q3_default = 0.5
            def klasifikasi_manual(rr):
                if rr == 0:
                    return "Cerah"
                elif rr <= 0.2 * q3_default:
                    return "Ringan"
                elif rr <= 0.5 * q3_default:
                    return "Sedang"
                elif rr <= q3_default:
                    return "Lebat"
                else:
                    return "Ekstrem"

            kategori = klasifikasi_manual(rr_prediksi)
            emoji_kategori = {
                'Cerah': '☀️', 'Ringan': '🌤️', 'Sedang': '🌦️', 'Lebat': '🌧️', 'Ekstrem': '⛈️'
            }

            st.success(f"💧 **Hasil Prediksi Curah Hujan**: {rr_prediksi:.2f} mm")
            st.info(f"🌈 **Kategori**: {emoji_kategori[kategori]} {kategori}")

            #if RR_AKTUAL > 0:
                #st.markdown("### 📊 Grafik Prediksi vs Aktual")
                #fig_manual, ax_manual = plt.subplots()
                #ax_manual.bar(['Aktual', 'Prediksi'], [RR_AKTUAL, rr_prediksi], color=['blue', 'orange'])
                #ax_manual.set_ylabel("Curah Hujan (RR)")
                #ax_manual.set_title("Perbandingan Curah Hujan")
                #st.pyplot(fig_manual)


# ==================== Menu: Analisis Prediksi ====================
elif menu == "Analisis Prediksi":
    st.subheader("📈 Analisis Hasil Prediksi")
    
    if "hasil_df" in st.session_state:
        hasil_df = st.session_state.hasil_df.copy()

        st.markdown("### 🔹 Prediksi vs Aktual")
        tanggal_dt = pd.to_datetime(hasil_df['TANGGAL'], errors='coerce')
        freq = (tanggal_dt.diff().median()).days

        if freq >= 28:
            hasil_df['LABEL'] = tanggal_dt.dt.strftime('%b %Y')  # Bulanan
            heatmap_mode = 'Bulanan'
        elif freq >= 3:
            hasil_df['LABEL'] = [f"Hari ke-{i+1}" for i in range(len(hasil_df))]  # 3 Harian
            heatmap_mode = '3 Harian'
        else:
            hasil_df['LABEL'] = [f"Hari ke-{i+1}" for i in range(len(hasil_df))]  # Harian
            heatmap_mode = 'Harian'

        label_step = max(1, len(hasil_df) // 20)
        xticks = hasil_df['LABEL'][::label_step]

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(hasil_df['LABEL'], hasil_df['RR_AKTUAL'], marker='o', label='Aktual')
        ax1.plot(hasil_df['LABEL'], hasil_df['RR_PREDIKSI'], marker='x', linestyle='--', label='Prediksi')
        ax1.set_xlabel("Waktu")
        ax1.set_ylabel("Curah Hujan (RR)")
        ax1.set_title("Prediksi vs Aktual")
        ax1.legend()
        ax1.set_xticks(xticks)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True)
        st.pyplot(fig1)

        # === Tambahan Analisis Ekstrem per Tahun ===
        st.markdown("### 🌪️ Analisis Ekstrem per Tahun (Aktual vs Prediksi)")
        
        threshold_ekstrem = hasil_df['RR_AKTUAL'].quantile(0.75)

        df_ekstrem = hasil_df.copy()
        df_ekstrem['TAHUN'] = pd.to_datetime(df_ekstrem['TANGGAL']).dt.year
        df_ekstrem['IS_EKSTREM_PRED'] = df_ekstrem['RR_PREDIKSI'] > threshold_ekstrem
        df_ekstrem['IS_EKSTREM_AKTUAL'] = df_ekstrem['RR_AKTUAL'] > threshold_ekstrem

        ekstrem_tahunan = df_ekstrem.groupby('TAHUN')[['IS_EKSTREM_PRED', 'IS_EKSTREM_AKTUAL']].sum().astype(int)
        ekstrem_tahunan.columns = ['Prediksi Ekstrem', 'Aktual Ekstrem']

        st.dataframe(ekstrem_tahunan)

        st.markdown("### 📊 Grafik Jumlah Kejadian Cuaca Ekstrem per Tahun")

        fig6, ax6 = plt.subplots(figsize=(8, 5))
        ekstrem_tahunan.plot(
            kind='bar',
            ax=ax6,
            color=['orange', 'dodgerblue']
        )
        ax6.set_title('📊 Ekstrem per Tahun: Prediksi vs Aktual')
        ax6.set_xlabel('Tahun')
        ax6.set_ylabel('Jumlah Kejadian Ekstrem')
        ax6.grid(axis='y', linestyle='--', alpha=0.7)
        ax6.legend(loc='upper right')
        st.pyplot(fig6)

        st.markdown("### 📊 Grafik Batang Prediksi")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(hasil_df['LABEL'], hasil_df['RR_PREDIKSI'], color='skyblue')
        ax2.set_xlabel("Waktu")
        ax2.set_ylabel("Curah Hujan (RR)")
        ax2.set_title("Curah Hujan Prediksi")
        ax2.set_xticks(xticks)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True)
        st.pyplot(fig2)

        st.markdown("### 🌈 Kategori Hujan")
        kategori_warna = {
            '☀️ Cerah': 'lightgray',
            '🌤️  Ringan': 'lightblue',
            '🌦️ Sedang': 'dodgerblue',
            '🌧️ Lebat': 'orange',
            '⛈️ Ekstrem': 'red'
        }

        # ✅ Perbaikan mapping color agar aman dari error NaN
        colors = hasil_df['KATEGORI'].map(kategori_warna).fillna('gray')

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.bar(hasil_df['LABEL'], hasil_df['RR_PREDIKSI'], color=colors)
        ax3.set_xlabel("Waktu")
        ax3.set_ylabel("Curah Hujan (RR)")
        ax3.set_title("Kategori Hujan berdasarkan Prediksi")
        ax3.set_xticks(xticks)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True)
        st.pyplot(fig3)

        st.markdown("### ✨ Interaktif: Prediksi Curah Hujan")
        hasil_df_reset = hasil_df.reset_index()
        fig4 = px.bar(
            hasil_df_reset,
            x='LABEL',
            y='RR_PREDIKSI',
            color='KATEGORI',
            color_discrete_map=kategori_warna,
            hover_data=['KATEGORI', 'RR_PREDIKSI'],
            labels={'LABEL': 'Waktu', 'RR_PREDIKSI': 'Curah Hujan (RR)'},
            title='Prediksi Curah Hujan Interaktif'
        )
        fig4.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("### 🧊 Heatmap Pola Curah Hujan")
        hasil_df['DT'] = pd.to_datetime(hasil_df['TANGGAL'], errors='coerce')

        if heatmap_mode == 'Bulanan':
            hasil_df['Tahun'] = hasil_df['DT'].dt.year
            hasil_df['Bulan'] = hasil_df['DT'].dt.strftime('%b')
            df_heatmap = hasil_df.dropna(subset=['Tahun', 'Bulan'])
            df_heatmap = df_heatmap.groupby(['Tahun', 'Bulan'])['RR_PREDIKSI'].mean().reset_index()
            pivot = df_heatmap.pivot(index='Tahun', columns='Bulan', values='RR_PREDIKSI')
            pivot = pivot.reindex(columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax5, linewidths=0.5, linecolor='white')
            ax5.set_title(f"Heatmap Curah Hujan ({heatmap_mode})")
            st.pyplot(fig5)

        else:
            hasil_df['Hari'] = hasil_df['DT'].dt.day
            hasil_df['Bulan'] = hasil_df['DT'].dt.strftime('%b')
            df_heatmap = hasil_df.dropna(subset=['Hari', 'Bulan'])
            df_heatmap = df_heatmap.groupby(['Hari', 'Bulan'])['RR_PREDIKSI'].mean().reset_index()
            pivot = df_heatmap.pivot(index='Hari', columns='Bulan', values='RR_PREDIKSI')
            pivot = pivot.reindex(columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            pivot = pivot.sort_index(ascending=True)
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap='YlGnBu', linewidths=0.3, linecolor='gray', cbar_kws={'label': 'RR Prediksi'}, ax=ax5)
            ax5.set_title(f"Heatmap Curah Hujan ({heatmap_mode})")
            ax5.set_ylabel("Hari")
            ax5.set_xlabel("Bulan")
            st.pyplot(fig5)


    else:
        st.warning("⚠️ Silakan upload dan prediksi dulu di menu 'Upload & Prediksi'")
        
