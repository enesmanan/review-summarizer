import os
import re
import warnings
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from textblob import TextBlob
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
plt.style.use("seaborn")

nltk.download("stopwords")
nltk.download("punkt")


class ReviewAnalyzer:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.turkish_stopwords = self.get_turkish_stopwords()

        # Lojistik ve satıcı ile ilgili kelimeleri genişletilmiş liste ile tanımla
        self.logistics_seller_words = {
            # Kargo ve teslimat ile ilgili
            "kargo",
            "kargocu",
            "paket",
            "paketleme",
            "teslimat",
            "teslim",
            "gönderi",
            "gönderim",
            "ulaştı",
            "ulaşım",
            "geldi",
            "kurye",
            "dağıtım",
            "hasarlı",
            "hasar",
            "kutu",
            "ambalaj",
            "zamanında",
            "geç",
            "hızlı",
            "yavaş",
            "günde",
            "saatte",
            # Satıcı ve mağaza ile ilgili
            "satıcı",
            "mağaza",
            "sipariş",
            "trendyol",
            "tedarik",
            "stok",
            "garanti",
            "fatura",
            "iade",
            "geri",
            "müşteri",
            "hizmet",
            "destek",
            "iletişim",
            "şikayet",
            "sorun",
            "çözüm",
            "hediye",
            # Fiyat ve ödeme ile ilgili
            "fiyat",
            "ücret",
            "para",
            "bedava",
            "ücretsiz",
            "indirim",
            "kampanya",
            "taksit",
            "ödeme",
            "bütçe",
            "hesap",
            "kur",
            # Zaman ile ilgili teslimat kelimeleri
            "bugün",
            "yarın",
            "dün",
            "hafta",
            "gün",
            "saat",
            "süre",
            "bekleme",
            "gecikme",
            "erken",
            "geç",
        }

        # Sentiment analizi için kelimeler
        self.positive_words = {
            "güzel",
            "harika",
            "mükemmel",
            "süper",
            "iyi",
            "muhteşem",
            "teşekkür",
            "memnun",
            "başarılı",
            "kaliteli",
            "kusursuz",
            "özgün",
            "şahane",
            "enfes",
            "ideal",
        }

        self.negative_words = {
            "kötü",
            "berbat",
            "rezalet",
            "yetersiz",
            "başarısız",
            "vasat",
            "korkunç",
            "düşük",
            "zayıf",
            "çöp",
            "pişman",
            "kırık",
            "bozuk",
        }

        # Türkçe-İngilizce ay çevirisi
        self.month_map = {
            "Ocak": "January",
            "Şubat": "February",
            "Mart": "March",
            "Nisan": "April",
            "Mayıs": "May",
            "Haziran": "June",
            "Temmuz": "July",
            "Ağustos": "August",
            "Eylül": "September",
            "Ekim": "October",
            "Kasım": "November",
            "Aralık": "December",
        }

    def get_turkish_stopwords(self):
        """Türkçe stop words listesini oluştur"""
        turkish_stops = set(stopwords.words("turkish"))

        github_url = "https://raw.githubusercontent.com/sgsinclair/trombone/master/src/main/resources/org/voyanttools/trombone/keywords/stop.tr.turkish-lucene.txt"
        try:
            response = requests.get(github_url)
            if response.status_code == 200:
                github_stops = set(
                    word.strip() for word in response.text.split("\n") if word.strip()
                )
                turkish_stops.update(github_stops)
        except Exception as e:
            print(f"GitHub'dan stop words çekilirken hata oluştu: {e}")

        custom_stops = {
            "bir",
            "ve",
            "çok",
            "bu",
            "de",
            "da",
            "için",
            "ile",
            "ben",
            "sen",
            "o",
            "biz",
            "siz",
            "onlar",
            "bu",
            "şu",
            "ama",
            "fakat",
            "ancak",
            "lakin",
            "ki",
            "dahi",
            "mi",
            "mı",
            "mu",
            "mü",
        }
        turkish_stops.update(custom_stops)

        return turkish_stops

    def filter_product_reviews(self):
        """Salt ürün yorumlarını filtrele"""

        def is_pure_product_review(text):
            if not isinstance(text, str):
                return False

            text_lower = text.lower()
            return not any(word in text_lower for word in self.logistics_seller_words)

        # Filtrelenmiş DataFrame
        original_count = len(self.df)
        self.df = self.df[self.df["Yorum"].apply(is_pure_product_review)]
        filtered_count = len(self.df)

        print(f"\nFiltreleme İstatistikleri:")
        print(f"Orijinal yorum sayısı: {original_count}")
        print(f"Salt ürün yorumu sayısı: {filtered_count}")
        print(f"Çıkarılan yorum sayısı: {original_count - filtered_count}")
        print(
            f"Filtreleme oranı: {((original_count - filtered_count) / original_count * 100):.2f}%"
        )

        print("\nÖrnek Salt Ürün Yorumları:")
        sample_reviews = self.df["Yorum"].sample(min(3, len(self.df)))
        for idx, review in enumerate(sample_reviews, 1):
            print(f"{idx}. {review[:100]}...")

    def convert_turkish_date(self, date_str):
        """Türkçe tarihleri İngilizce'ye çevir"""
        try:
            day, month, year = date_str.split()
            english_month = self.month_map[month]
            return f"{day} {english_month} {year}"
        except:
            return None

    def preprocess_text(self, text):
        """Metin ön işleme"""
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\d+", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        return ""

    def analyze_timestamps(self):
        """Zaman bazlı analizler"""
        # Tarihleri dönüştür
        self.df["Tarih"] = self.df["Tarih"].apply(self.convert_turkish_date)
        self.df["Tarih"] = pd.to_datetime(self.df["Tarih"], format="%d %B %Y")

        # Günlük dağılım
        plt.figure(figsize=(12, 6))
        plt.hist(self.df["Tarih"], bins=20, edgecolor="black")
        plt.title("Yorumların Zaman İçindeki Dağılımı")
        plt.xlabel("Tarih")
        plt.ylabel("Yorum Sayısı")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("images/yorum_zaman_dagilimi.png")
        plt.close()

        # Aylık dağılım
        monthly_reviews = self.df.groupby(self.df["Tarih"].dt.to_period("M")).size()
        plt.figure(figsize=(12, 6))
        monthly_reviews.plot(kind="bar")
        plt.title("Aylık Yorum Dağılımı")
        plt.xlabel("Ay")
        plt.ylabel("Yorum Sayısı")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("images/aylik_yorum_dagilimi.png")
        plt.close()

        # Mevsimsel analiz
        self.df["Mevsim"] = self.df["Tarih"].dt.month.map(
            {
                12: "Kış",
                1: "Kış",
                2: "Kış",
                3: "İlkbahar",
                4: "İlkbahar",
                5: "İlkbahar",
                6: "Yaz",
                7: "Yaz",
                8: "Yaz",
                9: "Sonbahar",
                10: "Sonbahar",
                11: "Sonbahar",
            }
        )
        seasonal_reviews = self.df.groupby("Mevsim").size()
        plt.figure(figsize=(10, 6))
        seasonal_reviews.plot(kind="bar")
        plt.title("Mevsimsel Yorum Dağılımı")
        plt.xlabel("Mevsim")
        plt.ylabel("Yorum Sayısı")
        plt.tight_layout()
        plt.savefig("images/mevsimsel_dagilim.png")
        plt.close()

    def analyze_ratings(self):
        """Yıldız bazlı analizler"""
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x="Yıldız Sayısı")
        plt.title("Yıldız Dağılımı")
        plt.xlabel("Yıldız Sayısı")
        plt.ylabel("Yorum Sayısı")
        plt.savefig("images/yildiz_dagilimi.png")
        plt.close()

        return {
            "Ortalama Yıldız": self.df["Yıldız Sayısı"].mean(),
            "Medyan Yıldız": self.df["Yıldız Sayısı"].median(),
            "Mod Yıldız": self.df["Yıldız Sayısı"].mode()[0],
            "Standart Sapma": self.df["Yıldız Sayısı"].std(),
        }

    def create_wordcloud(self):
        """Kelime bulutu oluştur"""
        all_comments = " ".join(
            [self.preprocess_text(str(comment)) for comment in self.df["Yorum"]]
        )

        words = word_tokenize(all_comments)
        filtered_words = [word for word in words if word not in self.turkish_stopwords]
        clean_text = " ".join(filtered_words)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=100,
            font_path="C:/Windows/Fonts/arial.ttf",  # Windows varsayılan font
        ).generate(clean_text)

        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig("images/wordcloud.png")
        plt.close()

    def analyze_ngrams(self, max_n=3, top_n=10):
        """N-gram analizi"""
        all_texts = []
        for comment in self.df["Yorum"]:
            if isinstance(comment, str):
                words = self.preprocess_text(comment).split()
                filtered_words = [
                    word for word in words if word not in self.turkish_stopwords
                ]
                all_texts.extend(filtered_words)

        for n in range(1, max_n + 1):
            print(f"\n{n}-gram Analizi:")

            if n == 1:
                ngrams_list = all_texts
            else:
                ngrams_list = list(ngrams(all_texts, n))

            ngram_freq = Counter(ngrams_list).most_common(top_n)

            if n == 1:
                labels = [item[0] for item in ngram_freq]
            else:
                labels = [" ".join(item[0]) for item in ngram_freq]

            values = [item[1] for item in ngram_freq]

            plt.figure(figsize=(12, 6))
            bars = plt.barh(range(len(values)), values)
            plt.yticks(range(len(labels)), labels)
            plt.title(f"En Sık Kullanılan {n}-gramlar")
            plt.xlabel("Frekans")

            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(
                    width,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(width)}",
                    ha="left",
                    va="center",
                    fontweight="bold",
                )

            plt.tight_layout()
            plt.savefig(f"images/{n}gram_analizi.png")
            plt.close()

            print(f"\nEn sık kullanılan {n}-gramlar:")
            for ngram, freq in ngram_freq:
                if n == 1:
                    print(f"{ngram}: {freq}")
                else:
                    print(f"{' '.join(ngram)}: {freq}")

    def analyze_sentiment(self):
        """Duygu analizi"""

        def count_sentiment_words(text):
            if not isinstance(text, str):
                return 0, 0

            text_lower = text.lower()
            words = text_lower.split()
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)
            return positive_count, negative_count

        sentiment_counts = self.df["Yorum"].apply(count_sentiment_words)
        self.df["Pozitif_Kelime_Sayisi"] = [count[0] for count in sentiment_counts]
        self.df["Negatif_Kelime_Sayisi"] = [count[1] for count in sentiment_counts]
        self.df["Sentiment_Skor"] = (
            self.df["Pozitif_Kelime_Sayisi"] - self.df["Negatif_Kelime_Sayisi"]
        )

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x="Yıldız Sayısı", y="Sentiment_Skor")
        plt.title("Yıldız Sayısı ve Sentiment Skoru İlişkisi")
        plt.savefig("images/sentiment_yildiz_iliskisi.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(self.df["Sentiment_Skor"], bins=20)
        plt.title("Sentiment Skor Dağılımı")
        plt.xlabel("Sentiment Skoru")
        plt.ylabel("Yorum Sayısı")
        plt.savefig("images/sentiment_dagilimi.png")
        plt.close()

    def analyze_comment_lengths(self):
        """Yorum uzunluğu analizi"""
        self.df["Yorum_Uzunlugu"] = self.df["Yorum"].str.len()

        plt.figure(figsize=(10, 6))
        plt.hist(self.df["Yorum_Uzunlugu"].dropna(), bins=30)
        plt.title("Yorum Uzunluğu Dağılımı")
        plt.xlabel("Karakter Sayısı")
        plt.ylabel("Yorum Sayısı")
        plt.savefig("images/yorum_uzunluk_dagilimi.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x="Yıldız Sayısı", y="Yorum_Uzunlugu")
        plt.title("Yıldız Sayısı ve Yorum Uzunluğu İlişkisi")
        plt.xlabel("Yıldız")
        plt.ylabel("Yorum Uzunluğu (Karakter)")
        plt.savefig("images/yildiz_uzunluk_iliskisi.png")
        plt.close()

    def run_analysis(self):
        """Ana analiz fonksiyonu"""
        print("Analiz başlatılıyor...")

        if not os.path.exists("images"):
            os.makedirs("images")

        print("\nÜrün odaklı yorum filtresi uygulanıyor...")
        self.filter_product_reviews()

        print("\n1. Yorum Uzunluğu Analizi")
        self.analyze_comment_lengths()

        print("\n2. Zaman Analizi")
        self.analyze_timestamps()

        print("\n3. Yıldız Analizi")
        rating_stats = self.analyze_ratings()
        print("\nYıldız İstatistikleri:")
        for key, value in rating_stats.items():
            print(f"{key}: {value:.2f}")

        print("\n4. Kelime Bulutu Oluşturuluyor")
        self.create_wordcloud()

        print("\n5. N-gram Analizleri")
        self.analyze_ngrams(max_n=3, top_n=10)

        print("\n6. Duygu Analizi")
        self.analyze_sentiment()

        print("\nAnaliz tamamlandı! Tüm görseller 'images' klasörüne kaydedildi.")


if __name__ == "__main__":
    analyzer = ReviewAnalyzer("data/macbook_product_comments_with_ratings.csv")
    analyzer.run_analysis()
