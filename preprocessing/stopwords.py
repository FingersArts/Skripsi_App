import mysql.connector

# Baca file stopwords.txt
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f if line.strip()]

# Koneksi ke database MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="skripsi"
)
cursor = conn.cursor()

# Masukkan setiap kata ke tabel
for word in stopwords:
    cursor.execute("INSERT INTO stopwords (word) VALUES (%s)", (word,))

# Simpan perubahan dan tutup koneksi
conn.commit()
cursor.close()
conn.close()

print(f"{len(stopwords)} stopwords berhasil dimasukkan ke database.")
