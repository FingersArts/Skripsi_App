import pandas as pd
import mysql.connector

# Baca file kamus-slang.csv
df = pd.read_csv("D:\Collage\Akademis\Semester 8\Skripsi_App\preprocessing\kamus-slang.csv", header=None, names=["slang", "formal"])

conn = mysql.connector.connect(
    host="localhost",
        user="root",
        password="",
        database="skripsi"
)
cursor = conn.cursor()

# Masukkan data ke tabel kamus_slang
for _, row in df.iterrows():
    cursor.execute(
        "INSERT INTO kamus_slang (slang, formal) VALUES (%s, %s)",
        (row['slang'], row['formal'])
    )

conn.commit()
cursor.close()
conn.close()

print(f"{len(df)} data slang berhasil dimasukkan ke database.")