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

def ambil_stopwords(table_name="stopwords", csv_path="database\stopwords.csv"):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute(f"SELECT word FROM {table_name}")
        result = cursor.fetchall()
        stopwords = [row[0] for row in result]
        cursor.close()
        conn.close()
        return stopwords
    except Exception as e:
        print(f"[INFO] Gagal konek ke database: {e}")
        print("[INFO] Mengambil stopwords dari CSV sebagai alternatif...")
        if os.path.exists(csv_path):
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                return [row[0] for row in reader if row]  # Ambil kata jika baris tidak kosong
        else:
            print(f"[ERROR] File CSV tidak ditemukan: {csv_path}")
            return []

def tambah_stopwords(kata_list, table_name="stopwords"):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        # Ambil semua kata yang sudah ada
        cursor.execute(f"SELECT word FROM {table_name}")
        existing = set(row[0] for row in cursor.fetchall())

        # Filter kata yang belum ada
        new_words = [kata for kata in kata_list if kata not in existing]

        if new_words:
            query = f"INSERT INTO {table_name} (word) VALUES (%s)"
            cursor.executemany(query, [(word,) for word in new_words])
            conn.commit()
            return f"{len(new_words)} kata berhasil ditambahkan."
        else:
            return "Semua kata sudah ada di database."
    except Error as e:
        return f"Gagal menambahkan kata: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
