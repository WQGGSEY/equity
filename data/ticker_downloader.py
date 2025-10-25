import requests
import pandas as pd
from io import StringIO
import sqlite3 as sql
import os
import re

ishares_russell_3000_url = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"

DB_DIR = "/Users/seongje/Desktop/project/domain shift lab/equity/db"

def init_download_ishares_russell_3000() -> pd.DataFrame:
    response = requests.get(ishares_russell_3000_url)
    data = response.content.decode('utf-8')
    with StringIO(data) as f:
        line = f.readlines()
        # print all lines for debugging
        is_started = False
        data_tuple = []
        for l in line:
            # just take in between \" \"
            # since there is number with `,`, parse with \",\" is not sufficient


            parsed_line = re.findall(r'"(.*?)"', l)
            if len(parsed_line) == 0 or parsed_line[0] == '-': is_started = False
            if is_started:
                ticker = parsed_line[0]
                sector = parsed_line[2]
                asset_type = parsed_line[3]
                exchage = parsed_line[10]

                if asset_type == 'Equity':
                    data_tuple.append((ticker, sector, exchage))
               

            if not is_started and len(parsed_line) >= 2: is_started = True
    insert_stocks_to_db(data_tuple)
    

def construct_list_db():
    conn = sql.connect(f"{DB_DIR}/equity.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history_russell_3000_list (
        ticker varchar(10) PRIMARY KEY UNIQUE,
        sector varchar(50),
        exchange varchar(50)
    )
    """)
    conn.commit()
    conn.close()

def insert_stocks_to_db(data: list[tuple]):
    conn = sql.connect(f"{DB_DIR}/equity.db")
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT OR IGNORE INTO history_russell_3000_list (ticker, sector, exchange)
    VALUES (?, ?, ?)
    """, data)
    conn.commit()
    conn.close()

def check_table_exists() -> bool:
    conn = sql.connect(f"{DB_DIR}/equity.db")
    cursor = conn.cursor()
    cursor.execute("""
    SELECT name FROM sqlite_master WHERE type='table' AND name='daily_prices';
    """)
    result = cursor.fetchone()
    conn.close()
    return result is not None

def check_recent_date() -> str:
    conn = sql.connect(f"{DB_DIR}/equity.db")
    cursor = conn.cursor()
    cursor.execute("""
    SELECT MAX(date) FROM daily_prices;
    """)
    result = cursor.fetchone()
    conn.close()
    if result and result[0]:
        return result[0]
    else:
        return ""


def main():
    if os.path.exists(DB_DIR) is False:
        os.makedirs(DB_DIR)
    construct_list_db()
    init_download_ishares_russell_3000()

if __name__ == "__main__":
    main()

        


