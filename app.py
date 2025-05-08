import streamlit as st
import pandas as pd
import numpy as np
import time
from preprocessing.preprocessing import preproces
from tfidf import calculate_tfidf
from labelling import apply_score_based_labeling
from collections import defaultdict, Counter
import math

st.set_page_config(layout="wide")
st.title('ANALISIS SENTIMEN 100 HARI KERJA :orange[PRESIDEN PRABOWO SUBIANTO]')

# Sidebar navigasi
st.sidebar.markdown("## 📌 Navigasi")
upload_btn = st.sidebar.button("📤 Upload Data")
preprocess_btn = st.sidebar.button("🩹 Preprocessing")
label_btn = st.sidebar.button("🏷️ Labeling")
tfidf_btn = st.sidebar.button("🧮 TF-IDF")
nb_btn = st.sidebar.button("🤖 Naive Bayes")

# Set tab berdasarkan tombol yang diklik
if upload_btn:
    st.session_state['tab'] = "Upload Data"
elif preprocess_btn:
    st.session_state['tab'] = "Preprocessing"
elif label_btn:
    st.session_state['tab'] = "Labeling"
elif tfidf_btn:
    st.session_state['tab'] = "TF-IDF"
elif nb_btn:
    st.session_state['tab'] = "Naive Bayes"

# Default tab
if 'tab' not in st.session_state:
    st.session_state['tab'] = "Upload Data"

# ========== Upload Data ==========
if st.session_state['tab'] == "Upload Data":
    uploaded_file = st.file_uploader("Pilih File CSV", type=["csv"])

    if uploaded_file is not None:
        df_awal = pd.read_csv(uploaded_file)
        st.session_state['df_awal'] = df_awal  # simpan data awal

        # Tampilkan data awal
        st.info(f"📥 Jumlah data awal: {len(df_awal)} baris")
        st.subheader('Data Awal')
        st.write(df_awal)

        # Bersihkan data
        df_bersih = df_awal.drop_duplicates(subset=['full_text'])
        df_bersih = df_bersih.dropna()
        st.session_state['df'] = df_bersih  # simpan data bersih

        st.success(f"✅ Jumlah data setelah dibersihkan (tanpa duplikat & NaN): {len(df_bersih)} baris")
        st.subheader('Data Setelah Dibersihkan')
        st.write(df_bersih)

    else:
        # Jika file sudah pernah diupload dan disimpan
        if 'df_awal' in st.session_state:
            st.info(f"📥 Jumlah data awal: {len(st.session_state['df_awal'])} baris")
            st.subheader('Data Awal')
            st.write(st.session_state['df_awal'])

        if 'df' in st.session_state:
            st.success(f"✅ Jumlah data setelah dibersihkan: {len(st.session_state['df'])} baris")
            st.subheader('Data Setelah Dibersihkan')
            st.write(st.session_state['df'])


# ========== Preprocessing ==========
elif st.session_state['tab'] == "Preprocessing":
    st.markdown("---")
    st.subheader("Upload Hasil Preprocessing (Opsional)")

    if st.button("📥 Ambil dari Database"):
        with st.spinner("Mengambil data dari database..."):
            from db import ambil_preprocessing
            df_db, message = ambil_preprocessing()
            if df_db is not None and not df_db.empty:
                st.session_state['preprocessed_df'] = df_db
                st.session_state['run_preprocessing'] = False
                st.success(message)
            else:
                st.error(message)

    if 'df' not in st.session_state:
        st.warning("Silakan upload file data mentah di Tab Upload Data.")
        
    else:
        # Kalau belum upload hasil preprocessing, berikan opsi preprocessing dari awal
        if 'preprocessed_df' in st.session_state:
            if st.button('🔄 Ulangi Preprocessing'):
                st.session_state.pop('preprocessed_df', None)
                st.session_state['run_preprocessing'] = True

        if 'preprocessed_df' not in st.session_state:
            if st.button('Mulai Preprocessing'):
                st.session_state['run_preprocessing'] = True

        if st.session_state.get('run_preprocessing', False):
            st.subheader("Proses Preprocessing")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(percent, message):
                progress_bar.progress(percent)
                status_text.text(message)

            with st.spinner('Sedang melakukan preprocessing...'):
                start_time = time.time()
                preprocessed_df = preproces(st.session_state['df'], progress_callback=update_progress)
                end_time = time.time()

                processing_time = end_time - start_time
                minutes, seconds = divmod(processing_time, 60)
                st.session_state['preprocessed_df'] = preprocessed_df
                st.session_state['run_preprocessing'] = False
                st.success(f"Preprocessing selesai dalam {int(minutes)} menit {int(seconds)} detik!")

    # Tampilkan hasil preprocessing, baik yang di-upload atau dari proses
    if 'preprocessed_df' in st.session_state:
        st.subheader('Data Setelah Preprocessing')
        st.write(st.session_state['preprocessed_df'][['full_text', 'casefolding', 'cleanedtext', 'slangremoved', 'stopwordremoved', 'tokenized', 'stemming', 'stemming_str']])

        word_counts = st.session_state['preprocessed_df']['stemming_str'].str.split(expand=True).stack().value_counts()
        st.subheader('Visualisasi 10 Kata Teratas')
        st.bar_chart(word_counts.head(10))
        st.subheader('10 Kata Teratas')
        st.write(word_counts.head(10))
        st.write(word_counts)

        # Tombol Download CSV
        csv = st.session_state['preprocessed_df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Hasil Preprocessing (CSV)",
            data=csv,
            file_name='hasil_preprocessing.csv',
            mime='text/csv'
        )

        # Tambahkan tombol update ke MySQL
        if st.button("🔄 Update ke MySQL"):
            with st.spinner("Memperbarui database..."):
                from db import update_preprocessing
                success, message = update_preprocessing(st.session_state['preprocessed_df'])
                if success:
                    st.success(message)
                else:
                    st.error(message)   

# ========== Labeling ==========
elif st.session_state['tab'] == "Labeling":
    st.subheader("Labeling")

    # Tombol untuk mengambil data dari database
    if st.button("📥 Ambil Data dari Database"):
        from db import ambil_labeling
        df, pesan = ambil_labeling()
        if df is not None:
            st.session_state['labeled_df'] = df
            st.success(pesan)
        else:
            st.error(pesan)

    if 'preprocessed_df' not in st.session_state:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")
        
    else:
        df = st.session_state['preprocessed_df']

        # Labeling otomatis
        if st.button("💡 Jalankan Labeling Otomatis"):
            try:
                from labelling import apply_score_based_labeling
                labeled_df = apply_score_based_labeling(df)
                st.session_state['preprocessed_df'] = labeled_df
                st.success("Labeling otomatis selesai!")
            except Exception as e:
                st.error(str(e))

        if 'sentiment' in df.columns:
            st.subheader("✏️ Edit Sentimen")

            # Editor interaktif: hanya kolom 'sentiment' yang bisa diedit
            edited_df = st.data_editor(
                df[['full_text', 'stemming', 'sentiment']],
                column_config={
                    "sentiment": st.column_config.SelectboxColumn(
                        "sentiment",
                        options=["positif", "netral", "negatif"]
                    )
                },
                disabled=["full_text", "stemming"],
                use_container_width=True,
                num_rows="dynamic"
            )

            # Simpan ke database
            if st.button("🗃️ Simpan ke Database"):
                st.session_state['preprocessed_df']['sentiment'] = edited_df['sentiment']
                from db import simpan_labeling
                berhasil, pesan = simpan_labeling(st.session_state['preprocessed_df'])
                if berhasil:
                    st.success(pesan)
                else:
                    st.error(pesan)

            # Tampilkan distribusi sentimen
            st.subheader("📊 Distribusi Sentimen")
            sentiment_counts = edited_df['sentiment'].value_counts()
            st.write(sentiment_counts)

            # Download data ke CSV
            csv = st.session_state['preprocessed_df'].to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Hasil Labelling (CSV)",
                data=csv,
                file_name="hasil_label_final.csv",
                mime="text/csv"
            )

# ========== TF-IDF ==========
elif st.session_state['tab'] == "TF-IDF":
    st.subheader("TF-IDF")

    if 'preprocessed_df' not in st.session_state:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")
    else:
        # Pastikan kolom tokenized dalam bentuk list
        df = st.session_state['preprocessed_df'].copy()

        if 'tfidf_ranking' in st.session_state:
            if st.button('🔄 Ulangi TF-IDF'):
                st.session_state.pop('tfidf_ranking', None)
                st.session_state['run_tfidf'] = True

        if 'tfidf_ranking' not in st.session_state:
            if st.button("Hitung TF-IDF"):
                st.session_state['run_tfidf'] = True

        if st.session_state.get('run_tfidf', False):
            with st.spinner("Sedang menghitung TF-IDF..."):
                ranking = calculate_tfidf(df)
                st.session_state['tfidf_ranking'] = ranking
                st.session_state['run_tfidf'] = False
                st.success("TF-IDF berhasil dihitung!")

        if 'tfidf_ranking' in st.session_state:
            st.subheader("Hasil TF-IDF (Top Terms)")
            st.dataframe(st.session_state['tfidf_ranking'].reset_index(drop=True))

            st.subheader("Visualisasi TF-IDF 10 Teratas")
            top_terms = st.session_state['tfidf_ranking'].head(10).set_index('term')
            st.bar_chart(top_terms)

# ========== Naive Bayes ==========
elif st.session_state['tab'] == "Naive Bayes":
    st.subheader("Klasifikasi Sentimen dengan Naive Bayes (Manual Tanpa Library)")

    if 'labeled_df' not in st.session_state:
        st.warning("Silakan lakukan labeling terlebih dahulu.")
    else:
        df = st.session_state['labeled_df']

        if 'sentiment' not in df.columns:
            st.error("Data belum memiliki kolom 'sentiment'. Lakukan labeling terlebih dahulu.")
        