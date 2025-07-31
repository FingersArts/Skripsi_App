import mysql.connector
from mysql.connector import Error
import os
import csv
import pandas as pd

# Koneksi ke database MySQL
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="skripsi"
    )

def ambil_slangwords(table_name="kamus_slang", csv_path="database\kamus_slang.csv"):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute(f"SELECT slang, formal FROM {table_name}")
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return dict(result)
    except Exception as e:
        print(f"[INFO] Gagal konek ke database: {e}")
        print("[INFO] Mengambil data dari CSV sebagai alternatif...")
        if os.path.exists(csv_path):
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                return {rows[0]: rows[1] for rows in reader if len(rows) >= 2}
        else:
            print(f"[ERROR] File CSV tidak ditemukan: {csv_path}")
            return {}

def tambah_slangword(slang, formal, table_name="kamus_slang"):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        # Cek apakah slang sudah ada
        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE slang = %s", (slang,))
        if cursor.fetchone()[0] > 0:
            return f"Kata slang '{slang}' sudah ada di database."
        
        # Tambahkan entri baru
        query = f"INSERT INTO {table_name} (slang, formal) VALUES (%s, %s)"
        cursor.execute(query, (slang, formal))
        conn.commit()
        return f"Berhasil menambahkan: '{slang}' → '{formal}'."
    except Error as e:
        return f"Gagal menambahkan slangword: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
