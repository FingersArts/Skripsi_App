import mysql.connector
from mysql.connector import Error
import pandas as pd
import ast

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="skripsi"
    )

def ambil_preprocessing(table_name="preprocessing_result"):
    try:
        conn = connect_to_db()
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)

        if 'tokenized' in df.columns:
            df['tokenized'] = df['tokenized'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        if 'stemming' in df.columns:
            df['stemming'] = df['stemming'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        conn.close()
        return df, f"{len(df)} data berhasil diambil dari tabel `{table_name}`!"
    except Error as e:
        return None, f"Gagal mengambil data dari MySQL: {e}"

def update_preprocessing(df, table_name="preprocessing_result"):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        # Tambahkan kolom stemming_str ke CREATE TABLE
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                full_text TEXT,
                casefolding TEXT,
                cleanedtext TEXT,
                slangremoved TEXT,
                stopwordremoved TEXT,
                tokenized TEXT,
                stemming TEXT,
                stemming_str TEXT
            )
        ''')

        for _, row in df.iterrows():
            tokenized_str = str(row['tokenized']) if isinstance(row['tokenized'], list) else row['tokenized']
            stemming_list_str = str(row['stemming']) if isinstance(row['stemming'], list) else row['stemming']

            cursor.execute(f'''
                INSERT INTO {table_name} 
                (full_text, casefolding, cleanedtext, slangremoved, stopwordremoved, tokenized, stemming, stemming_str)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    casefolding=VALUES(casefolding),
                    cleanedtext=VALUES(cleanedtext),
                    slangremoved=VALUES(slangremoved),
                    stopwordremoved=VALUES(stopwordremoved),
                    tokenized=VALUES(tokenized),
                    stemming=VALUES(stemming),
                    stemming_str=VALUES(stemming_str)
            ''', (
                row['full_text'],
                row['casefolding'],
                row['cleanedtext'],
                row['slangremoved'],
                row['stopwordremoved'],
                tokenized_str,
                stemming_list_str,
                row['stemming_str']
            ))

        conn.commit()
        cursor.close()
        conn.close()
        return True, f"{len(df)} data berhasil diperbarui atau ditambahkan di tabel `{table_name}`!"
    except Error as e:
        return False, f"Gagal menyimpan ke MySQL: {e}"


def simpan_labeling(df, table_name="preprocessing_result"):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        # Periksa apakah kolom 'sentiment' sudah ada (tetap diperlukan)
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = '{conn.database}' 
            AND TABLE_NAME = '{table_name}' 
            AND COLUMN_NAME = 'sentiment'
        """)
        if cursor.fetchone()[0] == 0:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN sentiment VARCHAR(10)")

        # Siapkan data (list of tuples): (sentiment, id)
        # INI BAGIAN YANG DIUBAH: Gunakan 'id' bukan 'full_text'
        data_to_update = [(row['sentiment'], row['full_text']) for _, row in df.iterrows()]

        # Gunakan executemany untuk update batch yang jauh lebih cepat
        query = f'''
            UPDATE {table_name}
            SET sentiment = %s
            WHERE full_text = %s 
        '''
        cursor.executemany(query, data_to_update)

        conn.commit()
        
        updated_rows = cursor.rowcount
        
        cursor.close()
        conn.close()
        return True, f"{updated_rows} data berhasil diperbarui kolom `sentiment` di tabel `{table_name}`!"
    except Error as e:
        return False, f"Gagal menyimpan data ke MySQL: {e}"
    except KeyError:
        return False, "Gagal menyimpan data: Pastikan DataFrame Anda memiliki kolom 'id' dan 'sentiment'."


def ambil_labeling(table_name="preprocessing_result"):
    try:
        conn = connect_to_db()
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)

        # Parsing stemming dari string ke list (jika perlu)
        if 'stemming' in df.columns:
            df['stemming'] = df['stemming'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        conn.close()
        return df, f"{len(df)} data berhasil diambil dari tabel `{table_name}`!"
    except Error as e:
        return None, f"Gagal mengambil data dari MySQL: {e}"