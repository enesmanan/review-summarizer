import os
import time
import requests
import re

import pandas as pd
import plotly.express as px
import streamlit as st

#from scrape.trendyol_scraper import scrape_product_comments
from scripts.review_summarizer import analyze_reviews

if not os.path.exists("data"):
    os.makedirs("data")


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
        placeholder="ürünün linki",
    )
    api_key = st.text_input("Gemini API Anahtarı", type="password")
    submitted = st.form_submit_button("Analiz Et")

if submitted and url and api_key:
    st.session_state.analysis_started = True

if st.session_state.analysis_started:
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Yorumlar çekiliyor...")
        progress_bar.progress(0.1)
        def scrape_product_comments_v2(url):
            headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-US,en;q=0.9","cache-control": "max-age=0","upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (iPad; CPU OS 14_6_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/129.0 Mobile/15E148 Safari/605.1.15"}       
            # Regex ile product_id'yi alıyoruz
            match = re.search(r"-p-(\d+)", url)
            if match:
                product_id = match.group(1)
                print(f"Product ID: {product_id}")
            else:
                print("Product ID not found.")
            # API URL'si oluşturuluyor
            api_url = f"https://apigw.trendyol.com/discovery-web-websfxsocialreviewrating-santral/product-reviews-detailed?contentId={product_id}&page=1&order=DESC&orderBy=Score&channelId=1"
            # 3) Yorumları çekmek için fonksiyon
            def fetch_reviews(api_url, headers):
                all_reviews = []
                try:
                # İlk isteği yap ve totalPages değerini al
                    response = requests.get(api_url, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        total_pages = data["result"]["productReviews"]["totalPages"]
                        print(f"Toplam Sayfa: {total_pages}")
            
                        # İlk sayfadaki verileri ekle
                        all_reviews.extend(data["result"]["productReviews"]["content"])
            
                        # Kalan sayfaları döngüyle al
                        for page in range(2, total_pages + 1):
                            paginated_url = api_url.replace("page=1", f"page={page}")
                            response = requests.get(paginated_url, headers=headers)
                            if response.status_code == 200:
                                page_data = response.json()
                                all_reviews.extend(page_data["result"]["productReviews"]["content"])
                            else:
                                print(f"Sayfa {page} için istek başarısız oldu: {response.status_code}")
        
                    else:
                        print(f"İstek başarısız oldu: {response.status_code}")
                except Exception as e:
                    print(f"Bir hata oluştu: {e}")
    
                return all_reviews
            # Yorumları çek
            reviews = fetch_reviews(api_url, headers)
            # Artık tüm yorumlar reviews listesinde yer alıyor.

            # 4) Yorumları pandas DataFrame'e dönüştürme
            reviews_df = pd.DataFrame(reviews)
            reviews_df = reviews_df.rename(columns={"id": "Kullanıcı_id","userFullName": "Kullanıcı Adı","comment": "Yorum","lastModifiedDate": "Tarih","rate": "Yıldız Sayısı"})
            reviews_df = reviews_df[["Kullanıcı_id","Kullanıcı Adı","Yorum","Tarih","Yıldız Sayısı"]]
            return reviews_df
        df = scrape_product_comments_v2(url)
        #df = scrape_product_comments(url)
        if df is None or len(df) == 0:
            st.error("Yorumlar çekilemedi. URL'yi kontrol edin.")
            st.session_state.analysis_started = False
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
