from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import os

# Create data directory if it doesn't exist
data_directory = "data"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

def comprehensive_scroll(driver):
    # Scroll until no more new content is loaded
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # Wait for potential content loading
        
        # Calculate new scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        # Check if bottom has been reached
        if new_height == last_height:
            break
        
        last_height = new_height

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-notifications")

service = Service(r"C:\Users\enesm\OneDrive\Masaüstü\chromedriver-win64\chromedriver.exe")
driver = webdriver.Chrome(service=service, options=chrome_options)
driver.maximize_window()

#url = "https://www.trendyol.com/apple/macbook-air-m1-cip-8gb-256gb-ssd-macos-13-qhd-tasinabilir-bilgisayar-uzay-grisi-p-68042136/yorumlar?boutiqueId=638145&merchantId=968"
url = "https://www.trendyol.com/go-aksesuar/iphone-13-ve-14-uyumlu-kirmizi-kalpler-baski-desenli-silikon-kilif-p-853410575/yorumlar?boutiqueId=61&merchantId=266456"
driver.get(url)

WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler'))
).click()

comprehensive_scroll(driver)

comment_elements = driver.find_elements(By.XPATH, '/html/body/div[1]/div[4]/div/div/div/div/div[3]/div/div/div[3]/div[2]/div')
total_comments = len(comment_elements)

data = []
for i in range(1, total_comments + 1):
    kullanıcı_id = i
    try:
        username_xpath = f'/html/body/div[1]/div[4]/div/div/div/div/div[3]/div/div/div[3]/div[2]/div[{i}]/div[1]/div[2]/div[1]'
        username = driver.find_element(By.XPATH, username_xpath).text
    except:
        username = "N/A"

    try:
        comment_xpath = f'/html/body/div[1]/div[4]/div/div/div/div/div[3]/div/div/div[3]/div[2]/div[{i}]/div[2]/p'
        comment = driver.find_element(By.XPATH, comment_xpath).text
    except:
        comment = "N/A"

    try:
        date_xpath = f'/html/body/div[1]/div[4]/div/div/div/div/div[3]/div/div/div[3]/div[2]/div[{i}]/div[1]/div[2]/div[2]'
        date = driver.find_element(By.XPATH, date_xpath).text
    except:
        date = "N/A"

    star_xpath_base = f'/html/body/div[1]/div[4]/div/div/div/div/div[3]/div/div/div[3]/div[2]/div[{i}]/div[1]/div[1]/div'
    try:
        full_stars = driver.find_elements(By.XPATH, f"{star_xpath_base}/div[@class='star-w']/div[@class='full'][@style='width: 100%; max-width: 100%;']")
        star_count = len(full_stars)
    except:
        star_count = 0

    data.append({
        "Kullanıcı_id": kullanıcı_id,
        "Kullanıcı Adı": username,
        "Yorum": comment,
        "Tarih": date,
        "Yıldız Sayısı": star_count
    })

output_file = os.path.join(data_directory, 'macbook_product_comments_with_ratings.csv')
df = pd.DataFrame(data)
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Total {len(data)} comments extracted and saved to {output_file}")

driver.quit()