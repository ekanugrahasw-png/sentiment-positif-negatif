import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

from google.colab import files
uploaded = files.upload()

# Jika CSV
df = pd.read_csv(next(iter(uploaded)), sep='\t')

# Jika Excel (.xlsx)
# df = pd.read_excel(next(iter(uploaded)))

df.head()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df['clean_text'] = df['Tweet'].astype(str).apply(clean_text)

plt.figure(figsize=(7,5))
sns.countplot(x='sentimen', data=df, palette='coolwarm', hue='sentimen', legend=False)

plt.title("Frekuensi Sentimen Komentar Twitter")
plt.xlabel("Sentimen (-1 = Negatif, 0 = Netral, 1 = Positif)")
plt.ylabel("Jumlah Komentar")
plt.show()

sentiment_count = df['sentimen'].value_counts()

plt.figure(figsize=(7,7))
plt.pie(sentiment_count, labels=['Negatif (-1)', 'Netral (0)', 'Positif (1)'],
        autopct='%1.1f%%', colors=['red', 'grey', 'green'])
plt.title("Perbandingan Sentimen Komentar Twitter")
plt.show()

text = " ".join(df['clean_text'])

wordcloud = WordCloud(width=1200, height=800,
                      background_color='white',
                      colormap='viridis',
                      max_words=200).generate(text)

plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud Komentar Twitter", fontsize=18)
plt.show()