import streamlit as st
import pandas as pd
import numpy as np
import time
from preprocessing.preprocessing import preproces
from tfidf import calculate_tfidf
from labelling import apply_score_based_labeling
from naivebayes import train_naive_bayes, predict, evaluate_model, stratified_split, plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

    from preprocessing.stopwords import tambah_stopwords  # atau nama file kamu
    st.header("Tambah Stopwords")

    input_kata = st.text_area("Masukkan kata-kata stopwords, pisahkan dengan koma (,) atau baris baru:")

    if st.button("Tambahkan ke Database"):
        # Normalisasi input
        kata_list = [k.strip() for k in input_kata.replace(",", "\n").splitlines() if k.strip()]
        if kata_list:
            hasil = tambah_stopwords(kata_list)
            st.success(hasil)
        else:
            st.warning("Masukkan minimal satu kata.")
    
    from preprocessing.kamus_slang import tambah_slangword
    st.header("Tambah Entri Kamus Slang")

    col1, col2 = st.columns(2)
    with col1:
        slang = st.text_input("Kata slang (contoh: 'gk')")

    with col2:
        formal = st.text_input("Versi formal (contoh: 'tidak')")

    if st.button("Tambahkan ke Kamus"):
        if slang.strip() and formal.strip():
            hasil = tambah_slangword(slang.strip(), formal.strip())
            st.success(hasil)
        else:
            st.warning("Kedua kolom harus diisi.")

    st.markdown("---")
    st.subheader("Hasil Preprocessing")

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
                st.session_state['labeled_df'] = labeled_df
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
                st.session_state['labeled_df'] = st.session_state['preprocessed_df']
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

    if 'labeled_df' not in st.session_state:
        st.warning("Silakan lakukan labeling terlebih dahulu.")
    else:
        df = st.session_state['labeled_df'].copy()

        if 'tokenized' not in df.columns:
            st.error("Kolom 'tokenized' tidak ditemukan. Pastikan preprocessing sudah dilakukan.")
        elif 'sentiment' not in df.columns:
            st.error("Kolom 'sentiment' tidak ditemukan. Pastikan data sudah dilabel.")


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
                st.session_state['labeled_df'] = df  # update hasil TF-IDF ke dalam session
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
    st.subheader("🤖 Naive Bayes Classification")

    if 'labeled_df' not in st.session_state:
        st.warning("Silakan lakukan labeling terlebih dahulu.")
    else:
        df = st.session_state['labeled_df']

        # Pastikan TF-IDF sudah tersedia
        if 'TF-IDF_dict' not in df.columns or df['TF-IDF_dict'].isnull().all():
            st.error("Kolom 'TF-IDF_dict' tidak ditemukan atau kosong. Pastikan TF-IDF sudah dihitung.")
        else:
            # Split data
            train_df, test_df = stratified_split(df, label_col='sentiment', test_ratio=0.2)
            st.info(f"Jumlah Data Latih: {len(train_df)} | Jumlah Data Uji: {len(test_df)}")

            # Training & prediction
            model = train_naive_bayes(train_df['TF_dict'], train_df['sentiment'], alpha=0.01)

            try:
                y_true = test_df['sentiment'].tolist()
                y_pred = [predict(tfidf, model) for tfidf in test_df['TF-IDF_dict']]
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
                st.stop()

            report, conf_matrix, accuracy = evaluate_model(y_true, y_pred)

            st.markdown("## 🎯 Performa Model")
            st.metric("🎯 Akurasi Total", f"{accuracy:.4f}")

            labels = list(report.keys())
            if 'accuracy' in labels:
                labels.remove('accuracy')
            if 'macro avg' in labels:
                labels.remove('macro avg')

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Recall per Kelas:")
                for label in labels:
                    st.write(f"{label.capitalize()}: {report[label]['recall']:.4f}")

            with col2:
                st.markdown("### Precision per Kelas:")
                for label in labels:
                    st.write(f"{label.capitalize()}: {report[label]['precision']:.4f}")

            with col3:
                st.markdown("### F1-Score per Kelas:")
                for label in labels:
                    st.write(f"{label.capitalize()}: {report[label]['f1-score']:.4f}")

            if 'macro avg' in report:
                st.markdown("### 📌 Macro Average")
                st.write(f"Precision: {report['macro avg']['precision']:.4f}")
                st.write(f"Recall: {report['macro avg']['recall']:.4f}")
                st.write(f"F1-Score: {report['macro avg']['f1-score']:.4f}")

            # Confusion Matrix
            st.markdown("### 📊 Confusion Matrix")
            conf_df = pd.DataFrame(
                conf_matrix,
                index=[f"True {l.capitalize()}" for l in labels],
                columns=[f"Pred {l.capitalize()}" for l in labels]
            )
            st.dataframe(conf_df, use_container_width=True)

            # Heatmap
            st.markdown("### 🔥 Visualisasi Confusion Matrix")
            fig = plot_confusion_matrix(conf_matrix, labels)
            st.pyplot(fig)

            # Hasil prediksi
            st.markdown("### 🔍 Contoh Prediksi")
            test_df = test_df.copy()
            test_df['Prediksi'] = y_pred
            test_df['Benar/Salah'] = test_df['sentiment'] == test_df['Prediksi']
            st.dataframe(test_df[['full_text', 'sentiment', 'Prediksi']].head(10))