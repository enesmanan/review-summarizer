import os
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def comprehensive_scroll(driver):
    """Sayfayi en alta kadar kaydir"""
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(90)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def scrape_product_comments(url):
    """
    Trendyol ürün yorumlarını çek ve DataFrame olarak döndür

    Parameters:
        url (str): Trendyol ürün yorumları sayfasının URL'si
    """
    try:
        # Chrome ayarlarını yapılandır
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")

        # WebDriver'ı başlat
        service = Service(
            r"C:\Users\enesm\OneDrive\Masaüstü\chromedriver-win64\chromedriver.exe"
        )
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # URL'ye git
        print(f"Sayfa yükleniyor: {url}")
        driver.get(url)

        # Çerezleri kabul et
        try:
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            ).click()
            print("Çerezler kabul edildi")
        except:
            print("Çerez bildirimi bulunamadı veya zaten kabul edilmiş")

        # Sayfayı kaydır
        print("Tüm yorumlar yükleniyor...")
        comprehensive_scroll(driver)

        # Yorumları topla
        print("Yorumlar toplanıyor...")
        comment_elements = driver.find_elements(
            By.XPATH,
            "/html/body/div[1]/div[4]/div/div/div/div/div[3]/div/div/div[3]/div[2]/div",
        )
        total_comments = len(comment_elements)
        print(f"Toplam {total_comments} yorum bulundu")

        # Veriyi topla
        data = []
        for i in range(1, total_comments + 1):
            try:
                username_xpath = f"/html/body/div[1]/div[4]/div/div/div/div/div[3]/div/div/div[3]/div[2]/div[{i}]/div[1]/div[2]/div[1]"
                username = driver.find_element(By.XPATH, username_xpath).text
            except:
                username = "N/A"

            try:
                comment_xpath = f"/html/body/div[1]/div[4]/div/div/div/div/div[3]/div/div/div[3]/div[2]/div[{i}]/div[2]/p"
                comment = driver.find_element(By.XPATH, comment_xpath).text
            except:
                comment = "N/A"

            try:
                date_xpath = f"/html/body/div[1]/div[4]/div/div/div/div/div[3]/div/div/div[3]/div[2]/div[{i}]/div[1]/div[2]/div[2]"
                date = driver.find_element(By.XPATH, date_xpath).text
            except:
                date = "N/A"

            star_xpath_base = f"/html/body/div[1]/div[4]/div/div/div/div/div[3]/div/div/div[3]/div[2]/div[{i}]/div[1]/div[1]/div"
            try:
                full_stars = driver.find_elements(
                    By.XPATH,
                    f"{star_xpath_base}/div[@class='star-w']/div[@class='full'][@style='width: 100%; max-width: 100%;']",
                )
                star_count = len(full_stars)
            except:
                star_count = 0

            data.append(
                {
                    "Kullanıcı_id": i,
                    "Kullanıcı Adı": username,
                    "Yorum": comment,
                    "Tarih": date,
                    "Yıldız Sayısı": star_count,
                }
            )

            if i % 10 == 0:
                print(f"İşlenen yorum: {i}/{total_comments}")

        driver.quit()

        # DataFrame oluştur
        df = pd.DataFrame(data)

        # Data klasörünü kontrol et ve oluştur
        if not os.path.exists("data"):
            os.makedirs("data")

        return df

    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return None
