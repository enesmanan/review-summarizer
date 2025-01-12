import os
import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

nltk.download("stopwords")
nltk.download("punkt")


class ReviewAnalyzer:
    def __init__(self):
        self.turkish_stopwords = self.get_turkish_stopwords()
        self.setup_sentiment_model()
        self.setup_summary_model()

        # Lojistik ve satıcı ile ilgili kelimeleri tanımla
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

    def get_turkish_stopwords(self):
        """Genişletilmiş stop words listesini hazırla"""
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
        """Metin ön işleme"""
        if isinstance(text, str):
            # Küçük harfe çevir
            text = text.lower()
            # Özel karakterleri temizle
            text = re.sub(r"[^\w\s]", "", text)
            # Sayıları temizle
            text = re.sub(r"\d+", "", text)
            # Fazla boşlukları temizle
            text = re.sub(r"\s+", " ", text).strip()
            # Stop words'leri çıkar
            words = text.split()
            words = [word for word in words if word not in self.turkish_stopwords]
            return " ".join(words)
        return ""

    def setup_sentiment_model(self):
        """Sentiment analiz modelini hazırla"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device for sentiment: {self.device}")

        model_name = "savasy/bert-base-turkish-sentiment-cased"
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sentiment_model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(self.device)
            .to(torch.float32)
        )

    def setup_summary_model(self):
        """Özet modelini hazırla"""
        print("Loading Trendyol-LLM model...")
        model_id = "Trendyol/Trendyol-LLM-8b-chat-v2.0"

        self.summary_pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
        )

        self.terminators = [
            self.summary_pipe.tokenizer.eos_token_id,
            self.summary_pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        self.sampling_params = {
            "do_sample": True,
            "temperature": 0.3,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        }

    def filter_reviews(self, df):
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

    def analyze_sentiment(self, df):
        """Sentiment analizi yap"""

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
        """En önemli anahtar kelimeleri bul"""
        text = " ".join(text_series.astype(str))
        words = self.preprocess_text(text).split()
        word_freq = Counter(words)
        # En az 3 kez geçen kelimeleri al
        return {
            word: count
            for word, count in word_freq.items()
            if count >= 3 and len(word) > 2
        }

    def generate_summary(self, df):
        """Yorumların genel özetini oluştur"""
        # En önemli yorumları seç
        high_rated = df[df["Yıldız Sayısı"] >= 4]
        low_rated = df[df["Yıldız Sayısı"] <= 2]

        # Önemli kelimeleri bul
        positive_phrases = self.get_key_phrases(high_rated["cleaned_text"])
        negative_phrases = self.get_key_phrases(low_rated["cleaned_text"])

        # En anlamlı yorumları seç
        top_positive = (
            high_rated.sort_values("sentiment_score", ascending=False)["Yorum"]
            .head(3)
            .tolist()
        )
        top_negative = (
            low_rated.sort_values("sentiment_score")["Yorum"].head(2).tolist()
        )

        # En sık kullanılan kelimeler
        pos_features = ", ".join(
            [f"{word} ({count})" for word, count in list(positive_phrases.items())[:5]]
        )
        neg_features = ", ".join(
            [f"{word} ({count})" for word, count in list(negative_phrases.items())[:5]]
        )

        summary_prompt = f"""
        MacBook Air Kullanıcı Yorumları Analizi:

        İSTATİSTİKLER:
        - Toplam Yorum: {len(df)}
        - Ortalama Puan: {df['Yıldız Sayısı'].mean():.1f}/5
        - Pozitif Yorum Oranı: {(len(df[df['sentiment_label'] == 'Pozitif']) / len(df) * 100):.1f}%

        SIKÇA KULLANILAN KELİMELER:
        Olumlu: {pos_features}
        Olumsuz: {neg_features}

        ÖRNEK OLUMLU YORUMLAR:
        {' '.join([f"• {yorum[:200]}..." for yorum in top_positive])}

        ÖRNEK OLUMSUZ YORUMLAR:
        {' '.join([f"• {yorum[:200]}..." for yorum in top_negative])}

        Lütfen bu veriler ışığında bu ürün için kısa ve öz bir değerlendirme yap.
        Özellikle kullanıcıların en çok beğendiği özellikler ve en sık dile getirilen sorunlara odaklan.
        Değerlendirmeyi 3 paragrafla sınırla ve somut örnekler kullan.
        """

        messages = [
            {
                "role": "system",
                "content": "Sen bir ürün yorumları analiz uzmanısın. Yorumları özetlerken nesnel ve açık ol.",
            },
            {"role": "user", "content": summary_prompt},
        ]

        outputs = self.summary_pipe(
            messages,
            max_new_tokens=512,
            eos_token_id=self.terminators,
            return_full_text=False,
            **self.sampling_params,
        )

        return outputs[0]["generated_text"]


def analyze_reviews(file_path):
    df = pd.read_csv(file_path)

    analyzer = ReviewAnalyzer()

    filtered_df = analyzer.filter_reviews(df)

    print("Sentiment analizi başlatılıyor...")
    analyzed_df = analyzer.analyze_sentiment(filtered_df)

    analyzed_df.to_csv(
        "sentiment_analyzed_reviews.csv", index=False, encoding="utf-8-sig"
    )
    print("Sentiment analizi tamamlandı ve kaydedildi.")

    print("\nÜrün özeti oluşturuluyor...")
    summary = analyzer.generate_summary(analyzed_df)

    with open("urun_ozeti.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print("\nÜrün Özeti:")
    print("-" * 50)
    print(summary)
    print("\nÖzet 'urun_ozeti.txt' dosyasına kaydedildi.")


if __name__ == "__main__":
    analyze_reviews("data/macbook_product_comments_with_ratings.csv")
