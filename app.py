import os
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from scrape.trendyol_scraper_origin import scrape_comments
from scripts.review_summarizer import analyze_reviews

st.set_page_config(page_title="Trendyol Yorum Analizi", layout="wide")


def create_sentiment_plot(df):
    """Creates a pie chart visualization for sentiment distribution"""
    sentiment_counts = df["sentiment_label"].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Duygu Analizi Dağılımı",
        color_discrete_map={
            "Pozitif": "#2ecc71",
            "Nötr": "#95a5a6",
            "Negatif": "#e74c3c",
        },
    )
    return fig


def create_star_plot(df):
    """Creates a bar chart visualization for star rating distribution"""
    star_counts = df["Yıldız Sayısı"].value_counts().sort_index()
    fig = px.bar(
        x=star_counts.index,
        y=star_counts.values,
        title="Yıldız Dağılımı",
        labels={"x": "Yıldız Sayısı", "y": "Yorum Sayısı"},
        color_discrete_sequence=["#f39c12"],
    )
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            ticktext=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
        )
    )
    return fig


if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

st.title("Trendyol Yorum Analizi")
st.markdown(
    """
Bu uygulama, Trendyol ürün sayfasındaki yorumları analiz eder ve özetler.

Kullanım:
1. Trendyol ürün yorumlar sayfasının URL'sini girin
2. Gemini API anahtarınızı girin
3. 'Analiz Et' butonuna tıklayın
"""
)

with st.form("analysis_form"):
    url = st.text_input(
        "Trendyol Ürün Yorumları URL",
        placeholder="https://www.trendyol.com/.../yorumlar",
    )
    api_key = st.text_input("Gemini API Anahtarı", type="password")
    submitted = st.form_submit_button("Analiz Et")

if submitted and url and api_key:
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Yorumlar çekiliyor...")
        progress_bar.progress(0.1)
        
        df = scrape_comments(url)

        if df is None or len(df) == 0:
            st.error("Yorumlar çekilirken bir hata oluştu veya hiç yorum bulunamadı.")
            st.stop()
        else:
            data_path = os.path.join("data", "product_comments.csv")
            df.to_csv(data_path, index=False, encoding="utf-8-sig")

            status_text.text("Yorumlar analiz ediliyor...")
            progress_bar.progress(0.4)
            summary, analyzed_df = analyze_reviews(data_path, api_key)

            status_text.text("Sonuçlar hazırlanıyor...")
            progress_bar.progress(0.7)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Toplam Yorum", len(df))
            with col2:
                st.metric("Ürün Değerlendirme Sayısı", len(analyzed_df))
            with col3:
                st.metric(
                    "Ortalama Puan", f"{analyzed_df['Yıldız Sayısı'].mean():.1f}⭐"
                )
            with col4:
                positive_ratio = (
                    len(analyzed_df[analyzed_df["sentiment_label"] == "Pozitif"])
                    / len(analyzed_df)
                    * 100
                )
                st.metric("Olumlu Yorum Oranı", f"%{positive_ratio:.1f}")

            removed_reviews = len(df) - len(analyzed_df)
            if removed_reviews > 0:
                st.info(
                    f"Not: Toplam {removed_reviews} adet kargo, teslimat ve satıcı ile ilgili yorum analiz dışı bırakılmıştır."
                )

            st.subheader("📝 Genel Değerlendirme")
            st.write(summary)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_sentiment_plot(analyzed_df), use_container_width=True
                )
            with col2:
                st.plotly_chart(create_star_plot(analyzed_df), use_container_width=True)

            progress_bar.progress(1.0)
            status_text.text("Analiz tamamlandı!")

    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
        st.session_state.analysis_started = False

url = "https://www.trendyol.com/ronic-nutrition/gainer-mass-ultimate-3000-g-kilo-almaya-yardimci-karbonhidrat-tozu-cikolata-aromali-p-37609782/yorumlar"
reviews_df = scrape_comments(url)

if reviews_df is not None:
    print(f"Toplam {len(reviews_df)} yorum çekildi")
    print(reviews_df)