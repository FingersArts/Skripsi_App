import mysql.connector
from mysql.connector import Error
import pandas as pd

# Koneksi ke database MySQL
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="skripsi"
    )

def ambil_slangwords(table_name="kamus_slang"):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute(f"SELECT slang, formal FROM {table_name}")
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    # ubah jadi dictionary
    return dict(result)

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
