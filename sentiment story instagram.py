import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from sklearn.utils.class_weight import compute_sample_weight
from google.colab import files

# Upload file
uploaded = files.upload()

# Jika CSV
df = pd.read_csv(next(iter(uploaded)), sep='\t')

# Jika Excel (.xlsx)
# df = pd.read_excel(next(iter(uploaded)))

# Tampilkan 5 baris pertama
df.head()

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Menghapus URL
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Menghapus karakter selain huruf dan spasi
    return text

# Bersihkan kolom 'Instagram'
df['clean_text'] = df['Instagram'].astype(str).apply(clean_text)

# Pembobotan Sentimen (misalnya, memberi bobot berdasarkan sentimen)
# Sentimen: -1 = Negatif, 0 = Netral, 1 = Positif
sentiment_weights = {1: 1.0, 0: 0.5, -1: -1.0}

# Menambahkan kolom bobot sentimen berdasarkan 'sentimen'
df['weight'] = df['sentimen'].map(sentiment_weights)

# Menampilkan tabel pembobotan untuk beberapa baris pertama
print("\nTabel Pembobotan Data:")
print(df[['Instagram', 'sentimen', 'weight']].head())

# --- Evaluasi Berdasarkan Distribusi Sentimen ---
# Visualisasi distribusi sentimen
plt.figure(figsize=(7,5))
sns.countplot(x='sentimen', data=df, palette='coolwarm', hue='sentimen', legend=False)
plt.title("Frekuensi Sentimen Komentar Instagram")
plt.xlabel("Sentimen (-1 = Negatif, 0 = Netral, 1 = Positif)")
plt.ylabel("Jumlah Komentar")
plt.show()

# --- Evaluasi Berdasarkan Pembobotan Sentimen ---
# Melihat distribusi bobot berdasarkan sentimen
plt.figure(figsize=(7,5))
sns.countplot(x='weight', data=df, palette='coolwarm')
plt.title("Distribusi Bobot Sentimen")
plt.xlabel("Bobot Sentimen")
plt.ylabel("Jumlah Komentar")
plt.show()

# --- Evaluasi Berdasarkan Word Cloud ---
# Word Cloud dari komentar
text = " ".join(df['clean_text'])
wordcloud = WordCloud(width=1200, height=800,
                      background_color='white',
                      colormap='viridis',
                      max_words=200).generate(text)

plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud Komentar Instagram", fontsize=18)
plt.show()

# --- Evaluasi Berdasarkan Sentimen Positif, Negatif, dan Netral ---
# Menampilkan analisis komentar dengan sentimen positif, negatif, dan netral
df_positive = df[df['sentimen'] == 1]
df_negative = df[df['sentimen'] == -1]
df_neutral = df[df['sentimen'] == 0]

print("\nAnalisis Komentar Positif (Sentimen = 1):")
print(df_positive[['Instagram', 'clean_text']].head())

print("\nAnalisis Komentar Negatif (Sentimen = -1):")
print(df_negative[['Instagram', 'clean_text']].head())

print("\nAnalisis Komentar Netral (Sentimen = 0):")
print(df_neutral[['Instagram', 'clean_text']].head())

# --- Evaluasi Berdasarkan Total Bobot Sentimen ---
# Menghitung total bobot sentimen untuk evaluasi lebih lanjut
total_weight = df['weight'].sum()
print("\nTotal Bobot Sentimen dalam Data:")
print(f"Total Bobot Sentimen: {total_weight}")