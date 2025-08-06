
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from bs4 import BeautifulSoup
import time

def scrape_yahoo_insider_trades_selenium(ticker="CRWD", min_trade_value=1_000_000):
    url = f"https://finance.yahoo.com/quote/{ticker}/insider-transactions"

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    driver.get(url)
    time.sleep(5)  # Allow time for JS to render

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    table = soup.find("table")
    if not table:
        print(f"No insider transaction table found for {ticker}.")
        return None

    rows = table.find_all("tr")[1:]
    data = []
    for row in rows:
        cols = [col.get_text(strip=True) for col in row.find_all("td")]
        if len(cols) == 7:
            data.append(cols)

    df = pd.DataFrame(data, columns=["Insider", "Relationship", "Date", "Transaction", "Shares", "Price", "Value"])
    df["Value ($)"] = df["Value"].str.replace("[$,]", "", regex=True).astype(float)
    df["Is Executive"] = df["Relationship"].str.contains("CEO|CFO|Chief|Director|President", case=False, na=False)
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    filtered = df[df["Value ($)"] >= min_trade_value].copy()
    return filtered

if __name__ == "__main__":
    df_filtered = scrape_yahoo_insider_trades_selenium("CRWD")
    if df_filtered is not None:
        print(df_filtered)
        df_filtered.to_csv("CRWD_Insider_Trades_Filtered.csv", index=False)
