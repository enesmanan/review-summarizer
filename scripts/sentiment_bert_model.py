import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")


class TurkishSentimentAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # sentiment model
        model_name = "savasy/bert-base-turkish-sentiment-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )

        # Lojistik ve satıcı kelimeleri
        self.logistics_seller_words = {
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
            "satıcı",
            "mağaza",
            "sipariş",
            "trendyol",
            "tedarik",
            "stok",
            "fiyat",
            "ücret",
            "para",
            "bedava",
            "indirim",
            "kampanya",
            "havale",
            "ödeme",
            "garanti",
            "fatura",
        }

    def predict_sentiment(self, text):
        """Tek bir metin için sentiment tahmini yap"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {"label": "Nötr", "score": 0.5}

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                prediction = probs.cpu().numpy()[0]

            # İki sınıflı model için (positive/negative)
            score = float(prediction[1])  # Pozitif sınıfın olasılığı

            # Daha hassas skor eşikleri
            if score > 0.75:  # Yüksek güvenle pozitif
                label = "Pozitif"
            elif score < 0.25:  # Yüksek güvenle negatif
                label = "Negatif"
            elif score > 0.55:  # Hafif pozitif eğilim
                label = "Pozitif"
            elif score < 0.45:  # Hafif negatif eğilim
                label = "Negatif"
            else:
                label = "Nötr"

            return {"label": label, "score": score}

        except Exception as e:
            print(f"Error in sentiment prediction: {e}")
            return {"label": "Nötr", "score": 0.5}

    def filter_product_reviews(self, df):
        """Ürün ile ilgili olmayan yorumları filtrele"""

        def is_product_review(text):
            if not isinstance(text, str):
                return False
            return not any(word in text.lower() for word in self.logistics_seller_words)

        filtered_df = df[df["Yorum"].apply(is_product_review)].copy()

        print(f"\nFiltreleme İstatistikleri:")
        print(f"Toplam yorum sayısı: {len(df)}")
        print(f"Ürün yorumu sayısı: {len(filtered_df)}")
        print(f"Filtrelenen yorum sayısı: {len(df) - len(filtered_df)}")
        print(
            f"Filtreleme oranı: {((len(df) - len(filtered_df)) / len(df) * 100):.2f}%"
        )

        return filtered_df

    def analyze_reviews(self, df):
        """Tüm yorumları analiz et"""
        print("\nSentiment analizi başlatılıyor...")

        filtered_df = self.filter_product_reviews(df)

        # Sentiment analizi
        results = []
        for text in filtered_df["Yorum"]:
            sentiment = self.predict_sentiment(text)
            results.append(sentiment)

        filtered_df["sentiment_score"] = [r["score"] for r in results]
        filtered_df["sentiment_label"] = [r["label"] for r in results]

        return filtered_df

    def create_visualizations(self, df):
        """Analiz sonuçlarını görselleştir"""
        if not os.path.exists("images"):
            os.makedirs("images")

        # 1. Sentiment Dağılımı
        plt.figure(figsize=(12, 6))
        sns.countplot(
            data=df, x="sentiment_label", order=["Pozitif", "Nötr", "Negatif"]
        )
        plt.title("Sentiment Dağılımı")
        plt.tight_layout()
        plt.savefig("images/sentiment_distribution.png", bbox_inches="tight", dpi=300)
        plt.close()

        # 2. Yıldız-Sentiment İlişkisi
        plt.figure(figsize=(12, 6))
        df_mean = df.groupby("Yıldız Sayısı")["sentiment_score"].mean().reset_index()
        sns.barplot(data=df_mean, x="Yıldız Sayısı", y="sentiment_score")
        plt.title("Yıldız Sayısına Göre Ortalama Sentiment Skoru")
        plt.tight_layout()
        plt.savefig("images/star_sentiment_relation.png", bbox_inches="tight", dpi=300)
        plt.close()

        # 3. Sentiment Score Dağılımı
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x="sentiment_score", bins=30)
        plt.title("Sentiment Score Dağılımı")
        plt.tight_layout()
        plt.savefig(
            "images/sentiment_score_distribution.png", bbox_inches="tight", dpi=300
        )
        plt.close()

    def print_statistics(self, df):
        """Analiz istatistiklerini yazdır"""
        print("\nSentiment Analizi Sonuçları:")
        print("-" * 50)

        sentiment_counts = df["sentiment_label"].value_counts()
        total_reviews = len(df)

        for label, count in sentiment_counts.items():
            percentage = (count / total_reviews) * 100
            print(f"{label}: {count} yorum ({percentage:.2f}%)")

        print("\nYıldız Bazlı Sentiment Skorları:")
        print("-" * 50)
        star_means = df.groupby("Yıldız Sayısı")["sentiment_score"].mean()
        for star, score in star_means.items():
            print(f"{star} Yıldız ortalama sentiment skoru: {score:.3f}")


def main():
    df = pd.read_csv("data/macbook_product_comments_with_ratings.csv")

    analyzer = TurkishSentimentAnalyzer()

    print("Analiz başlatılıyor...")
    analyzed_df = analyzer.analyze_reviews(df)

    print("\nGörselleştirmeler oluşturuluyor...")
    analyzer.create_visualizations(analyzed_df)

    analyzer.print_statistics(analyzed_df)

    output_file = "sentiment_analyzed_reviews.csv"
    analyzed_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSonuçlar '{output_file}' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
