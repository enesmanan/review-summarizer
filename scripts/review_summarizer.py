import os
import re
import warnings
from collections import Counter

import google.generativeai as genai
import nltk
import numpy as np
import pandas as pd
import requests
import torch
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


class ReviewAnalyzer:
    def __init__(self, gemini_api_key):
        self.turkish_stopwords = self.get_turkish_stopwords()
        self.setup_sentiment_model()
        self.setup_gemini_model(gemini_api_key)

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

    def get_turkish_stopwords(self):
        """Türkçe stop words listesi oluştur"""
        github_url = "https://raw.githubusercontent.com/sgsinclair/trombone/master/src/main/resources/org/voyanttools/trombone/keywords/stop.tr.turkish-lucene.txt"
        stop_words = set()

        try:
            response = requests.get(github_url)
            if response.status_code == 200:
                github_stops = set(
                    word.strip() for word in response.text.split("\n") if word.strip()
                )
                stop_words.update(github_stops)
        except Exception as e:
            print(f"GitHub'dan stop words çekilirken hata oluştu: {e}")

        stop_words.update(set(nltk.corpus.stopwords.words("turkish")))

        additional_stops = {
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
            "var",
            "yok",
            "olan",
            "içinde",
            "üzerinde",
            "bana",
            "sana",
            "ona",
            "bize",
            "size",
            "onlara",
            "evet",
            "hayır",
            "tamam",
            "oldu",
            "olmuş",
            "olacak",
            "etmek",
            "yapmak",
            "kez",
            "kere",
            "defa",
            "adet",
        }
        stop_words.update(additional_stops)

        print(f"Toplam {len(stop_words)} adet stop words yüklendi.")
        return stop_words

    def preprocess_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\d+", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            words = text.split()
            words = [word for word in words if word not in self.turkish_stopwords]
            return " ".join(words)
        return ""

    def setup_sentiment_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device for sentiment: {self.device}")

        model_name = "savasy/bert-base-turkish-sentiment-cased"
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sentiment_model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(self.device)
            .to(torch.float32)
        )

    def setup_gemini_model(self, api_key):
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel("gemini-pro")

    def filter_reviews(self, df):
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

    def analyze_sentiment(self, df):
        def predict_sentiment(text):
            if not isinstance(text, str) or len(text.strip()) == 0:
                return {"label": "Nötr", "score": 0.5}

            try:
                cleaned_text = self.preprocess_text(text)
                inputs = self.sentiment_tokenizer(
                    cleaned_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    prediction = probs.cpu().numpy()[0]

                score = float(prediction[1])

                if score > 0.75:
                    label = "Pozitif"
                elif score < 0.25:
                    label = "Negatif"
                elif score > 0.55:
                    label = "Pozitif"
                elif score < 0.45:
                    label = "Negatif"
                else:
                    label = "Nötr"

                return {"label": label, "score": score}

            except Exception as e:
                print(f"Error in sentiment prediction: {e}")
                return {"label": "Nötr", "score": 0.5}

        print("\nSentiment analizi yapılıyor...")
        results = [predict_sentiment(text) for text in df["Yorum"]]

        df["sentiment_score"] = [r["score"] for r in results]
        df["sentiment_label"] = [r["label"] for r in results]
        df["cleaned_text"] = df["Yorum"].apply(self.preprocess_text)

        return df

    def get_key_phrases(self, text_series):
        text = " ".join(text_series.astype(str))
        words = self.preprocess_text(text).split()
        word_freq = Counter(words)
        return {
            word: count
            for word, count in word_freq.items()
            if count >= 3 and len(word) > 2
        }

    def generate_summary(self, df):
        # en onemli yorumları sec
        high_rated = df[df["Yıldız Sayısı"] >= 4]
        low_rated = df[df["Yıldız Sayısı"] <= 2]

        # onemli kelimleri ve yorumlari al
        positive_features = self.get_key_phrases(high_rated["cleaned_text"])
        negative_features = self.get_key_phrases(low_rated["cleaned_text"])

        top_positive = (
            high_rated.sort_values("sentiment_score", ascending=False)["Yorum"]
            .head(3)
            .tolist()
        )
        top_negative = (
            low_rated.sort_values("sentiment_score")["Yorum"].head(2).tolist()
        )

        summary_prompt = f"""Bu ürünün genel değerlendirmesini doğal bir dille özetleyeceksin.

Veriler:
- Toplam {len(df)} değerlendirme var
- Ortalama puan: {df['Yıldız Sayısı'].mean():.1f}/5
- Pozitif yorum oranı: {(len(df[df['sentiment_label'] == 'Pozitif']) / len(df) * 100):.1f}%

En çok tekrar eden olumlu ifadeler: {', '.join(list(positive_features.keys())[:5])}
En çok tekrar eden olumsuz ifadeler: {', '.join(list(negative_features.keys())[:5])}

Örnek olumlu yorumlar:
{' '.join(top_positive)}

Örnek olumsuz yorumlar:
{' '.join(top_negative)}

Lütfen bu bilgileri kullanarak, ürünle ilgili kullanıcı deneyimlerini tek bir paragrafta, sohbet eder gibi doğal bir dille özetle.
İstatistikleri direkt verme, onları cümlelerin içine yerleştir. Olumlu ve olumsuz yönleri dengeli bir şekilde aktar."""

        response = self.gemini_model.generate_content(summary_prompt)
        return response.text


def analyze_reviews(file_path, api_key):
    print("Analiz başlatılıyor...")
    df = pd.read_csv(file_path)

    analyzer = ReviewAnalyzer(api_key)

    filtered_df = analyzer.filter_reviews(df)

    analyzed_df = analyzer.analyze_sentiment(filtered_df)

    summary = analyzer.generate_summary(analyzed_df)

    return summary, analyzed_df
