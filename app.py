import streamlit as st
import pandas as pd
import numpy as np
import time
from preprocessing.preprocessing import preproces
from tfidf import calculate_tfidf
from labelling import apply_score_based_labeling
from naivebayes import evaluate_model, k_fold_cross_validation_nb, plot_confusion_matrix_streamlit, prepare_naive_bayes_model, stratified_split
from mlr import LogisticRegression, plot_confusion_matrix, k_fold_cross_validation
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title('ANALISIS SENTIMEN 100 HARI KERJA :orange[PRESIDEN PRABOWO SUBIANTO]')

# Sidebar navigasi
st.sidebar.markdown("## 📌 Navigasi")
upload_btn = st.sidebar.button("📤 Upload Data")
preprocess_btn = st.sidebar.button("🩹 Preprocessing")
label_btn = st.sidebar.button("🏷️ Labeling")
tfidf_btn = st.sidebar.button("🧮 TF-IDF")
nb_btn = st.sidebar.button("🤖 Pengujian")
compare_btn = st.sidebar.button("📊 Perbandingan Model")

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
    st.session_state['tab'] = "Pengujian"
elif compare_btn:
    st.session_state['tab'] = "Perbandingan Model"

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

    st.subheader("🩹 Preprocessing Data")

    tab1, tab2 = st.tabs(["Preprocessing", "Setting Kamus"])

    with tab1:
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
            st.subheader('Kata Teratas')
            st.write(word_counts)

            left, right = st.columns(2)
            csv = st.session_state['preprocessed_df'].to_csv(index=False).encode('utf-8')
            if left.download_button( use_container_width=True,
                label="⬇️ Download Hasil Preprocessing (CSV)",
                data=csv,
                file_name='hasil_preprocessing.csv',
                mime='text/csv'
            ):
                left.markdown("You clicked the plain button.")

            if right.button("🔄 Update ke Database", use_container_width=True):
                with st.spinner("Memperbarui database..."):
                    from db import update_preprocessing
                    success, message = update_preprocessing(st.session_state['preprocessed_df'])
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

    with tab2:
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


# ========== Labeling ==========
elif st.session_state['tab'] == "Labeling":
    st.subheader("🏷️ Labeling")

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
    st.subheader("🧮 TF-IDF")

    if 'labeled_df' not in st.session_state:
        st.warning("Silakan lakukan labeling terlebih dahulu.")
    else:
        df = st.session_state['labeled_df'].copy()

        if 'tokenized' not in df.columns:
            st.error("Kolom 'tokenized' tidak ditemukan. Pastikan preprocessing sudah dilakukan.")
        elif 'sentiment' not in df.columns:
            st.error("Kolom 'sentiment' tidak ditemukan. Pastikan data sudah dilabel.")
        else:
            # Pastikan kolom 'stemming' ada
            if 'stemming' not in df.columns:
                st.error("Kolom 'stemming' tidak ditemukan. Pastikan stemming sudah dilakukan.")
            else:
                # Option untuk pengaturan TF-IDF
                with st.expander("Pengaturan TF-IDF", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        top_k = st.number_input("Jumlah Fitur (Top-K)", 
                                            min_value=500, max_value=5000, value=1000, step=100)
                        apply_smote = st.checkbox("Terapkan SMOTE", value=True, help="Aktifkan SMOTE untuk menyeimbangkan kelas")
                    
                    with col2:
                        min_n = st.number_input("N-Gram Minimum", min_value=1, max_value=3, value=1)
                        max_n = st.number_input("N-Gram Maximum", min_value=1, max_value=3, value=2)
                        ngram_range = (min_n, max_n)

                if 'tfidf_ranking' in st.session_state:
                    if st.button('🔄 Ulangi TF-IDF'):
                        st.session_state.pop('tfidf_ranking', None)
                        st.session_state.pop('tfidf_df', None)
                        st.session_state.pop('tfidf_resampled', None)  # Hapus data SMOTE
                        st.session_state.pop('tfidf_terms', None)
                        st.session_state.pop('tfidf_idf', None)
                        st.session_state['run_tfidf'] = True

                if 'tfidf_ranking' not in st.session_state:
                    if st.button("Hitung TF-IDF"):
                        st.session_state['run_tfidf'] = True

                if st.session_state.get('run_tfidf', False):
                    with st.spinner("Sedang menghitung TF-IDF..."):
                        try:
                            # Memanggil fungsi dengan parameter yang sesuai
                            ranking, result_df, df_resampled, selected_terms, idf = calculate_tfidf(
                                df, 
                                top_k=top_k, 
                                ngram_range=ngram_range,
                                label_column='sentiment',
                                apply_smote=apply_smote,
                                smote_random_state=42
                            )
                            
                            # Simpan semua hasil ke session state
                            st.session_state['tfidf_ranking'] = ranking
                            st.session_state['tfidf_df'] = result_df
                            st.session_state['tfidf_resampled'] = df_resampled  # Simpan data SMOTE
                            st.session_state['selected_terms'] = selected_terms
                            st.session_state['tfidf_idf'] = idf
                            st.session_state['ngram_range'] = ngram_range
                            st.session_state['run_tfidf'] = False
                            st.success("TF-IDF berhasil dihitung!" + (" Data telah di-resample dengan SMOTE." if apply_smote else ""))
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat menghitung TF-IDF: {str(e)}")
                            st.session_state['run_tfidf'] = False

                # Menampilkan hasil TF-IDF jika sudah dihitung
                if 'tfidf_ranking' in st.session_state:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Hasil TF-IDF(Top Terms)")
                        tfidf_cols = st.session_state['tfidf_ranking'].columns.tolist()
                        st.dataframe(st.session_state['tfidf_ranking'].reset_index(drop=True))
                        # Ekspor hasil ranking
                        csv = st.session_state['tfidf_ranking'].to_csv(index=False)
                        st.download_button(
                            label="Unduh Ranking Term (CSV)",
                            data=csv,
                            file_name="tfidf_ranking.csv",
                            mime="text/csv",
                        )
                    with col2:
                        # Wordcloud untuk hasil TF-IDF
                        st.subheader("Wordcloud Top TF-IDF Terms")
                        from wordcloud import WordCloud
                        # Deteksi nama kolom skor tfidf
                        tfidf_df = st.session_state['tfidf_ranking']
                        score_col = None
                        for col in ['score', 'tfidf', 'value', 'weight']:
                            if col in tfidf_df.columns:
                                score_col = col
                                break
                        if score_col is None:
                            score_col = tfidf_df.columns[1]  # fallback ke kolom kedua
                        tfidf_dict = dict(zip(tfidf_df['term'], tfidf_df[score_col]))
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_dict)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)

# ========== Pengujian ==========
elif st.session_state['tab'] == "Pengujian":
    st.subheader("🤖 Pengujian")
    # Create tabs
    tab1, tab2 = st.tabs(["Naive Bayes", "Logistic Regression"])

    # Tab Naive Bayes
    with tab1:
        st.subheader("Naive Bayes")

        # Ensure TF-IDF is calculated
        if 'tfidf_df' not in st.session_state:
            st.warning("Silakan lakukan perhitungan TF-IDF terlebih dahulu.")
        else:
            # Select data based on whether SMOTE is applied
            if 'tfidf_resampled' in st.session_state and st.session_state['tfidf_resampled'] is not None:
                df = st.session_state['tfidf_resampled'].copy()
                st.info("Menggunakan data yang telah di-resample dengan SMOTE untuk klasifikasi.")
            else:
                df = st.session_state['tfidf_df'].copy()
                st.info("Menggunakan data asli (tanpa SMOTE) untuk klasifikasi.")

            # Validate important columns
            if 'TF_IDF_Vec' not in df.columns or df['TF_IDF_Vec'].isnull().all():
                st.error("Kolom 'TF_IDF_Vec' tidak ditemukan atau kosong. Pastikan TF-IDF sudah dihitung.")
            elif 'sentiment' not in df.columns:
                st.error("Kolom 'sentiment' tidak ditemukan. Pastikan data sudah dilabel.")
            else:
                selected_terms = st.session_state.get('selected_terms', None)
                if selected_terms is None:
                    st.error("Vocabulary TF_IDF_Vec (selected_terms) tidak ditemukan di session_state.")
                    st.stop()

                # Convert TF-IDF vectors to appropriate format
                features = [list(vec) for vec in df['TF_IDF_Vec']]
                labels = df['sentiment'].tolist()

                # Choose splitting method
                with st.expander("Pengaturan Naive Bayes", expanded=False):
                    # Pilihan metode pelatihan
                    split_method = st.radio(
                        "Pilih Metode Pelatihan",
                        ["Stratified Split (80:20)", "K-Fold Cross-Validation"],
                        key="split_method_nb"
                    )
                    alpha = st.number_input(
                        "Alpha (Laplace Smoothing)",
                        min_value=0.0,
                        max_value=2.0,
                        value=0.01,
                        step=0.01,
                        format="%.2f",
                        key="alpha_nb"
                    )                    
                    n_folds = st.number_input("Jumlah Folds (K-Fold)", min_value=2, max_value=15, value=5, step=1, key="n_folds_nb")
                                
                if split_method == "Stratified Split (80:20)":
                    # Stratified Split
                    train_df, test_df = stratified_split(df, label_col='sentiment', test_ratio=0.2)                    
                    train_features = [list(vec) for vec in train_df['TF_IDF_Vec']]
                    test_features = [list(vec) for vec in test_df['TF_IDF_Vec']]
                    train_labels = train_df['sentiment'].tolist()
                    test_labels = test_df['sentiment'].tolist()

                    if st.button("Latih Model Naive Bayes"):
                        st.info(f"Jumlah Data Latih: {len(train_df)} | Jumlah Data Uji: {len(test_df)}")
                        with st.spinner("Melatih model Naive Bayes dengan metode {split_method}..."):
                            try:
                                model = prepare_naive_bayes_model(
                                    X_train=train_features,
                                    y_train=train_labels,
                                    selected_terms=selected_terms,
                                    alpha=alpha
                                )
                                st.session_state['naive_bayes_model'] = model
                                results = evaluate_model(model, test_features, test_labels)
                                st.session_state['naive_bayes_results'] = results
                                st.success("Model Naive Bayes berhasil dilatih dengan Stratified Split (80:20)!")
                            except Exception as e:
                                st.error(f"Terjadi kesalahan saat melatih model: {str(e)}")
                                st.stop()

                else:  # K-Fold Cross-Validation
                    if st.button("Latih Model Naive Bayes"):
                        with st.spinner(f"Melatih model Naive Bayes dengan metode {split_method}..."):
                            try:
                                # Convert features to numpy array as required by k_fold_cross_validation_nb
                                features_np = np.array(features)
                                labels_np = np.array(labels)

                                # Run k-fold cross-validation using provided function
                                avg_accuracy, avg_precision, avg_recall, avg_cm, avg_class_f1_scores = k_fold_cross_validation_nb(
                                    X=features_np,
                                    y=labels_np,
                                    selected_terms=selected_terms,
                                    alpha=alpha,
                                    k=n_folds
                                )

                                # Train a final model on all data for predictions
                                model = prepare_naive_bayes_model(
                                    X_train=features,
                                    y_train=labels,
                                    selected_terms=selected_terms,
                                    alpha=alpha
                                )

                                # Prepare results in the same format as evaluate_model
                                classes = np.unique(labels)
                                results = {
                                    'accuracy': avg_accuracy,
                                    'confusion_matrix': avg_cm.astype(int),  # Convert to int for display
                                    'precision': {c: p for c, p in zip(classes, avg_precision)},
                                    'recall': {c: r for c, r in zip(classes, avg_recall)},
                                    'f1_score': {c: f for c, f in zip(classes, avg_class_f1_scores)},
                                    'macro_precision': np.mean(avg_precision),
                                    'macro_recall': np.mean(avg_recall),
                                    'macro_f1': np.mean(avg_class_f1_scores),
                                    'classes': classes,
                                    split_method : 'kfold'
                                }

                                # Store results and model
                                st.session_state['naive_bayes_model'] = model
                                st.session_state['naive_bayes_results'] = results
                                st.success(f"Model Naive Bayes dengan {n_folds}-fold cross-validation berhasil dilatih!")
                            except Exception as e:
                                st.error(f"Terjadi kesalahan saat melatih model: {str(e)}")
                                st.stop()

                # Display results if available
                if 'naive_bayes_results' in st.session_state:
                    results = st.session_state['naive_bayes_results']
                    labels = results['classes']
                    st.markdown(f"## 🎯 Performa Model")
                    st.metric("🎯 Akurasi Total", f"{results['accuracy']:.4f}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("### Recall per Kelas:")
                        for label in labels:
                            st.write(f"{label.capitalize()}: {results['recall'][label]:.4f}")
                    with col2:
                        st.markdown("### Precision per Kelas:")
                        for label in labels:
                            st.write(f"{label.capitalize()}: {results['precision'][label]:.4f}")
                    with col3:
                        st.markdown("### F1-Score per Kelas:")
                        for label in labels:
                            st.write(f"{label.capitalize()}: {results['f1_score'][label]:.4f}")
                    st.markdown("### 📈 Macro Average Metrics")
                    st.write(f"Macro Precision: {results['macro_precision']:.4f}")
                    st.write(f"Macro Recall: {results['macro_recall']:.4f}")
                    st.write(f"Macro F1-Score: {results['macro_f1']:.4f}")
                    st.markdown("### 📊 Confusion Matrix")
                    conf_df = pd.DataFrame(
                        results['confusion_matrix'],
                        index=[f"True {l.capitalize()}" for l in labels],
                        columns=[f"Pred {l.capitalize()}" for l in labels]
                    )
                    st.dataframe(conf_df, use_container_width=True)
                    st.markdown("### 🔥 Visualisasi Confusion Matrix")
                    fig = plot_confusion_matrix_streamlit(results['confusion_matrix'], labels)
                    st.pyplot(fig)
                    st.session_state['nb_cm'] = results['confusion_matrix']
                    st.session_state['nb_labels'] = labels
                    st.markdown("### 🔍 Contoh Prediksi")
                    if split_method == "Stratified Split":
                        test_df = test_df.copy()
                        test_df['Prediksi'] = st.session_state['naive_bayes_model'].predict(test_features)
                        test_df['Benar/Salah'] = test_df['sentiment'] == test_df['Prediksi']
                        st.dataframe(test_df[['sentiment', 'Prediksi', 'Benar/Salah']].head(10))
                    else:
                        # For k-fold, show predictions from the final model on a subset
                        subset_df = df.head(10).copy()
                        subset_features = [list(vec) for vec in subset_df['TF_IDF_Vec']]
                        subset_labels = subset_df['sentiment'].tolist()
                        subset_df['Prediksi'] = st.session_state['naive_bayes_model'].predict(subset_features)
                        subset_df['Benar/Salah'] = subset_df['sentiment'] == subset_df['Prediksi']
                        st.dataframe(subset_df[['sentiment', 'Prediksi', 'Benar/Salah']])

    # Tab Logistic Regression
    with tab2:
        st.subheader("Logistic Regression")
        
        if 'tfidf_df' not in st.session_state:
            st.warning("Silakan lakukan perhitungan TF-IDF terlebih dahulu.")
        else:
            if 'tfidf_resampled' in st.session_state and st.session_state['tfidf_resampled'] is not None:
                df = st.session_state['tfidf_resampled'].copy()
                st.info("Menggunakan data yang telah di-resample dengan SMOTE untuk klasifikasi.")
            else:
                df = st.session_state['tfidf_df'].copy()
                st.info("Menggunakan data asli (tanpa SMOTE) untuk klasifikasi.")

            if 'TF_IDF_Vec' not in df.columns or df['TF_IDF_Vec'].isnull().all():
                st.error("Kolom 'TF_IDF_Vec' tidak ditemukan atau kosong. Pastikan TF-IDF sudah dihitung.")
            elif 'sentiment' not in df.columns:
                st.error("Kolom 'sentiment' tidak ditemukan. Pastikan data sudah dilabel.")
            else:
                label_map = {label: idx for idx, label in enumerate(np.unique(df['sentiment']))}
                inverse_label_map = {idx: label for label, idx in label_map.items()}
                df['sentiment_idx'] = df['sentiment'].map(label_map)
                X = np.array([list(vec) for vec in df['TF_IDF_Vec']])
                y = df['sentiment_idx'].values
                with st.expander("Pengaturan Logistic Regression", expanded=False):
                    # Pilihan metode pelatihan
                    training_method = st.radio(
                        "Pilih Metode Pelatihan",
                        ("K-Fold Cross Validation", "Stratified Split (80:20)")
                    )
                    num_iter = st.number_input("Jumlah Iterasi", min_value=100, max_value=5000, value=1000, step=100)
                    learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.01, step=0.001, format="%.3f")
                    k_folds = st.number_input("Jumlah Folds (K-Fold)", min_value=2, max_value=15, value=5, step=1)
                if st.button("Latih Model Logistic Regression"):
                    with st.spinner(f"Melatih model Logistic Regression dengan metode {training_method}..."):
                        try:
                            model = LogisticRegression(num_iter=num_iter, learning_rate=learning_rate)
                            if training_method == "K-Fold Cross Validation":
                                avg_accuracy, avg_precision, avg_recall, avg_cm, avg_class_f1_scores, _, _ = k_fold_cross_validation(X, y, model, k=k_folds)
                                st.session_state['lr_model'] = model
                                st.session_state['lr_results'] = {
                                    'avg_accuracy': avg_accuracy,
                                    'avg_precision': avg_precision,
                                    'avg_recall': avg_recall,
                                    'avg_cm': avg_cm,
                                    'avg_f1_scores': avg_class_f1_scores,
                                    'labels': [inverse_label_map[i] for i in range(len(label_map))],
                                    'training_method': 'kfold'
                                }
                            else:  # Stratified Split
                                from mlr import stratified_split, calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score, calculate_confusion_matrix
                                train_df, test_df = stratified_split(df, label_col='sentiment', test_ratio=0.2)
                                st.info(f"Jumlah Data Latih: {len(train_df)} | Jumlah Data Uji: {len(test_df)}")
                                X_train = np.array([list(vec) for vec in train_df['TF_IDF_Vec']])
                                y_train = train_df['sentiment_idx'].values
                                X_test = np.array([list(vec) for vec in test_df['TF_IDF_Vec']])
                                y_test = test_df['sentiment_idx'].values
                                model.train(X_train, y_train)
                                y_pred = model.predict(X_test)
                                accuracy = calculate_accuracy(y_test, y_pred)
                                precision = calculate_precision(y_test, y_pred)
                                recall = calculate_recall(y_test, y_pred)
                                avg_class_f1_scores = calculate_f1_score(y_test, y_pred)
                                cm = calculate_confusion_matrix(y_test, y_pred, len(label_map))
                                st.session_state['lr_model'] = model
                                st.session_state['lr_results'] = {
                                    'avg_accuracy': accuracy,
                                    'avg_precision': precision,
                                    'avg_recall': recall,
                                    'avg_cm': cm,
                                    'avg_f1_scores': avg_class_f1_scores,
                                    'labels': [inverse_label_map[i] for i in range(len(label_map))],
                                    'training_method': 'split',
                                    'test_df': test_df  # simpan untuk contoh prediksi
                                }
                            st.success("Model Logistic Regression berhasil dilatih!")
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat melatih model: {str(e)}")
                            st.stop()

                # Tampilkan hasil jika sudah dilatih
                if 'lr_results' in st.session_state:
                    results = st.session_state['lr_results']
                    labels = results['labels']
                    training_method = results.get('training_method', 'kfold')

                    # Tampilkan performa model
                    st.markdown(f"## 🎯 Performa Model ({'Rata-rata K-Fold' if training_method == 'kfold' else 'Stratified Split'})")
                    st.metric("🎯 Akurasi", f"{results['avg_accuracy']:.4f}")

                    # Tampilkan metrik per kelas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("### Recall per Kelas:")
                        for i, label in enumerate(labels):
                            st.write(f"{label.capitalize()}: {results['avg_recall'][i]:.4f}")

                    with col2:
                        st.markdown("### Precision per Kelas:")
                        for i, label in enumerate(labels):
                            st.write(f"{label.capitalize()}: {results['avg_precision'][i]:.4f}")

                    with col3:
                        st.markdown("### F1-Score per Kelas:")
                        for i, label in enumerate(labels):
                            st.write(f"{label.capitalize()}: {results['avg_f1_scores'][i]:.4f}")

                    # Confusion Matrix
                    st.markdown(f"### 📊 Confusion Matrix ({'Rata-rata' if training_method == 'kfold' else 'Split'})")
                    conf_df = pd.DataFrame(
                        results['avg_cm'],
                        index=[f"True {l.capitalize()}" for l in labels],
                        columns=[f"Pred {l.capitalize()}" for l in labels]
                    )
                    st.dataframe(conf_df, use_container_width=True)

                    # Heatmap confusion matrix
                    st.markdown("### 🔥 Visualisasi Confusion Matrix")
                    fig = plot_confusion_matrix(results['avg_cm'], labels)
                    st.pyplot(fig)
                    st.session_state['plot_confusion_matrix'] = fig

                    # Contoh hasil prediksi
                    st.markdown(f"### 🔍 Contoh Prediksi ({'K-Fold' if training_method == 'kfold' else 'Split'})")
                    if training_method == 'kfold':
                        sample_df = df.sample(n=min(10, len(df)), random_state=42).copy()
                        X_sample = np.array([list(vec) for vec in sample_df['TF_IDF_Vec']])
                        y_pred_sample = st.session_state['lr_model'].predict(X_sample)
                        y_pred_labels = [inverse_label_map[pred] for pred in y_pred_sample]
                        sample_df['Prediksi'] = y_pred_labels
                        sample_df['Benar/Salah'] = sample_df['sentiment'] == sample_df['Prediksi']
                        st.dataframe(sample_df[['sentiment', 'Prediksi', 'Benar/Salah']])
                    else:
                        test_df = results.get('test_df')
                        if test_df is not None:
                            sample_df = test_df.sample(n=min(10, len(test_df)), random_state=42).copy()
                            X_sample = np.array([list(vec) for vec in sample_df['TF_IDF_Vec']])
                            y_pred_sample = st.session_state['lr_model'].predict(X_sample)
                            y_pred_labels = [inverse_label_map[pred] for pred in y_pred_sample]
                            sample_df['Prediksi'] = y_pred_labels
                            sample_df['Benar/Salah'] = sample_df['sentiment'] == sample_df['Prediksi']
                            st.dataframe(sample_df[['sentiment', 'Prediksi', 'Benar/Salah']])

# ========== Perbandingan Model ==========
elif st.session_state['tab'] == "Perbandingan Model":
    st.subheader("📊 Perbandingan Model Naive Bayes vs Logistic Regression")
    # Pastikan hasil kedua model tersedia
    nb_ready = 'naive_bayes_model' in st.session_state and 'tfidf_df' in st.session_state
    lr_ready = 'lr_results' in st.session_state and 'tfidf_df' in st.session_state
    if not nb_ready or not lr_ready:
        st.warning("Silakan latih kedua model terlebih dahulu.")
    else:
        # Ambil hasil Naive Bayes dari session_state hasil split test
        nb_cm = st.session_state.get('nb_cm', None)
        nb_labels = st.session_state.get('nb_labels', None)
        nb_results = st.session_state.get('naive_bayes_results', None)
        # Ambil hasil Logistic Regression
        lr_results = st.session_state['lr_results']
        labels = lr_results['labels']
        # Bar chart perbandingan akurasi, precision, recall
        import plotly.graph_objects as go
        st.markdown("### 📈 Perbandingan Akurasi, Precision, Recall (Rata-rata)")
        metrics = ['Akurasi', 'Precision', 'Recall']
        if nb_results is not None:
            nb_values = [nb_results['accuracy'], np.mean(list(nb_results['precision'].values())), np.mean(list(nb_results['recall'].values()))]
        else:
            nb_values = [0, 0, 0]
        lr_values = [lr_results['avg_accuracy'], np.mean(lr_results['avg_precision']), np.mean(lr_results['avg_recall'])]
        fig = go.Figure(data=[
            go.Bar(name='Naive Bayes', x=metrics, y=nb_values),
            go.Bar(name='Logistic Regression', x=metrics, y=lr_values)
        ])
        fig.update_layout(barmode='group', yaxis=dict(title='Skor'))
        st.plotly_chart(fig, use_container_width=True)
        # Pie chart distribusi prediksi
        st.markdown("### 🥧 Distribusi Prediksi Model")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Naive Bayes")
            pred_nb = st.session_state['naive_bayes_model'].predict([list(vec) for vec in st.session_state['tfidf_df']['TF_IDF_Vec']])
            pred_nb_labels = pd.Series(pred_nb).value_counts().sort_index()
            st.plotly_chart({
                "data": [
                    {"labels": list(pred_nb_labels.index), "values": pred_nb_labels.values, "type": "pie"}
                ],
            }, use_container_width=True)
        with col2:
            st.markdown("#### Logistic Regression")
            model_lr = st.session_state['lr_model']
            pred_lr = model_lr.predict(np.array([list(vec) for vec in st.session_state['tfidf_df']['TF_IDF_Vec']]))
            pred_lr_labels = pd.Series([lr_results['labels'][i] for i in pred_lr]).value_counts().sort_index()
            st.plotly_chart({
                "data": [
                    {"labels": list(pred_lr_labels.index), "values": pred_lr_labels.values, "type": "pie"}
                ],
            }, use_container_width=True)
        # Confusion matrix visualisasi
        st.markdown("### 🔥 Confusion Matrix Kedua Model")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Naive Bayes")
            from naivebayes import plot_confusion_matrix_streamlit
            if nb_cm is not None and nb_labels is not None:
                fig_nb = plot_confusion_matrix_streamlit(nb_cm, [l.capitalize() for l in nb_labels])
                st.pyplot(fig_nb)
            else:
                st.info("Confusion matrix Naive Bayes belum tersedia.")
        with col2:
            st.markdown("#### Logistic Regression")
            from mlr import plot_confusion_matrix
            cm_lr = lr_results['avg_cm']
            lr_labels = [l.capitalize() for l in lr_results['labels']]
            fig_lr = plot_confusion_matrix(cm_lr, lr_labels)
            st.pyplot(fig_lr)