import os
import json
import time
import datetime
import random
import re
import requests
import pandas as pd
import yfinance as yf
import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# ==========================================
# 1. 設定 & 認証周り
# ==========================================

# 固定係数・基準値
CONST_NOPAT_RATE = 0.6       # M: NOPAT係数
CONST_PAYOUT_RATE = 0.4      # N: 擬似配当係数
CONST_MARKET_YIELD = 0.021   # V: 市場平均配当利回り (2.1%)

def get_config_from_env():
    """環境変数から設定JSONを読み込む"""
    keys_json = os.environ.get("GCP_KEYS")
    if not keys_json:
        raise ValueError("GCP_KEYS secret is missing.")
    return json.loads(keys_json)

def connect_gsheet(config):
    """Google Spreadsheetに接続"""
    creds_dict = config["gcp_credentials"]
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_url(config["spreadsheet_url"])
    worksheet = spreadsheet.worksheet(config["worksheet_name"])
    return worksheet

# ==========================================
# 2. スクレイピング & 通信ヘルパー
# ==========================================

def create_session():
    """リトライ付きセッション作成"""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=2, # バックオフを少し長めに
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

_HTTP_SESSION = create_session()

def is_market_open():
    """休日判定 (簡易版: 土日のみチェック)"""
    d = datetime.date.today()
    if d.weekday() >= 5:
        return False, "土日"
    return True, "稼働日"

def get_yahoo_jp_info(ticker_code):
    """Yahoo!ファイナンス(日本)から情報を取得 (エラー時はNoneを返す)"""
    code_only = str(ticker_code).replace(".T", "")
    url_div = f"https://finance.yahoo.co.jp/quote/{code_only}.T/dividend"
    url_prof = f"https://finance.yahoo.co.jp/quote/{code_only}.T/profile"
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    data = {"payout_ratio": None, "name": str(ticker_code), "sector": "-"}

    try:
        # ランダム待機 (サーバー負荷軽減)
        time.sleep(random.uniform(1.5, 3.0))

        # 1. 配当性向
        try:
            res = _HTTP_SESSION.get(url_div, headers=headers, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                th = soup.find("th", string=re.compile("配当性向"))
                if th:
                    td = th.find_next_sibling("td")
                    if td:
                        text = td.get_text(strip=True).replace("%", "")
                        if text and text not in ["-", "---"]:
                            data["payout_ratio"] = float(text)
        except:
            pass # ログ抑制
        
        # 2. 銘柄名・業種
        try:
            res_prof = _HTTP_SESSION.get(url_prof, headers=headers, timeout=10)
            if res_prof.status_code == 200:
                soup = BeautifulSoup(res_prof.text, "html.parser")
                title_tag = soup.find("title")
                if title_tag:
                    m = re.search(r'(.*?)【', title_tag.text)
                    if m: data["name"] = m.group(1).strip()
                
                TSE_SECTORS = [
                    "水産・農林業", "鉱業", "建設業", "食料品", "繊維製品", "パルプ・紙", "化学",
                    "医薬品", "石油・石炭製品", "ゴム製品", "ガラス・土石製品", "鉄鋼", "非鉄金属",
                    "金属製品", "機械", "電気機器", "輸送用機器", "精密機器", "その他製品",
                    "電気・ガス業", "陸運業", "海運業", "空運業", "倉庫・運輸関連業", "情報・通信業",
                    "卸売業", "小売業", "銀行業", "証券、商品先物取引業", "保険業",
                    "その他金融業", "不動産業", "サービス業"
                ]
                text_content = soup.get_text()
                for sec in TSE_SECTORS:
                    if sec in text_content:
                        data["sector"] = sec
                        break
        except:
            pass # ログ抑制

    except:
        pass # ログ抑制
    
    return data

# ==========================================
# 3. コアロジック
# ==========================================

def get_value(df, keys, date_col):
    if df.empty or date_col is None or date_col not in df.columns:
        return 0
    for key in keys:
        if key in df.index:
            val = df.loc[key, date_col]
            if not pd.isna(val):
                return float(val)
    return 0

def analyze_stock(ticker, current_price_cache):
    res = {
        "B_cost_ratio": None, "C_judge1": "不合格",
        "D_payout": None, "E_judge2": "不合格",
        "F_cagr": None, "G_judge3": "不合格",
        "H_cap": None, "I_shares": None, "J_equity": None, "K_op_income": None, "L_date": None,
        "M_nopat_k": CONST_NOPAT_RATE, "N_div_k": CONST_PAYOUT_RATE,
        "O_nopat": None, "P_pseudo_div": None, "Q_pseudo_roe": None,
        "R_roe_class": None, "S_7y_mult": None, "T_7y_div": None,
        "U_fut_yield": None, "V_mkt_yield": CONST_MARKET_YIELD,
        "W_upside": None, "X_price": None, "Y_target": None,
        "Z_final": "不合格",
        "AA_name": "", "AB_sector": ""
    }

    try:
        # Yahoo JP
        yj_data = get_yahoo_jp_info(ticker)
        res["AA_name"] = yj_data["name"]
        res["AB_sector"] = yj_data["sector"]
        
        # yfinance
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        fins = tk.financials
        bs = tk.balance_sheet
        
        if fins.empty or bs.empty:
            return format_result(res)

        # 現在株価
        current_price = current_price_cache.get(ticker)
        if not current_price:
            current_price = info.get("currentPrice") or info.get("previousClose")
        res["X_price"] = current_price

        dates = fins.columns
        if len(dates) == 0: return format_result(res)
        latest_date = dates[0]

        # --- Phase 1 ---
        # ① 営業費用売上比率
        revenue = get_value(fins, ['Total Revenue'], latest_date)
        op_income = get_value(fins, ['Operating Income', 'Operating Profit'], latest_date)
        cost = revenue - op_income
        
        pass_gate1 = False
        if revenue > 0 and op_income > 0 and cost > 0:
            ratio = revenue / cost
            res["B_cost_ratio"] = round(ratio, 2)
            if ratio >= 1.15:
                res["C_judge1"] = "合格"
                pass_gate1 = True
        
        # ② 配当性向
        payout = yj_data["payout_ratio"]
        if payout is not None:
            payout = payout / 100.0
        if payout is None:
            val = info.get("payoutRatio")
            if val is not None: payout = val
            
        pass_gate2 = False
        if payout is not None:
            res["D_payout"] = payout
            if 0.2 <= payout <= 0.6:
                res["E_judge2"] = "合格"
                pass_gate2 = True
        
        # ③ 増収
        pass_gate3 = False
        revs = []
        if len(dates) >= 4:
            for i in range(4):
                revs.append(get_value(fins, ['Total Revenue'], dates[i]))
            if all(r > 0 for r in revs):
                if revs[0] > revs[1] > revs[2] > revs[3]:
                    res["G_judge3"] = "合格"
                    pass_gate3 = True
                cagr = (revs[0] / revs[3]) ** (1/3) - 1
                res["F_cagr"] = cagr

        all_gates_passed = pass_gate1 and pass_gate2 and pass_gate3
        if not all_gates_passed:
            return format_result(res)

        # --- Phase 2 ---
        cap = info.get("marketCap")
        if not cap: return format_result(res)
        cap = cap / 100000000.0
        res["H_cap"] = cap

        shares = info.get("sharesOutstanding")
        if not shares: return format_result(res)
        res["I_shares"] = shares

        equity = get_value(bs, ['Total Stockholder Equity', 'Total Equity', 'Stockholders Equity'], bs.columns[0])
        if equity == 0: return format_result(res)
        equity = equity / 100000000.0
        res["J_equity"] = equity

        op_income = op_income / 100000000.0
        res["K_op_income"] = op_income
        res["L_date"] = str(latest_date.date())

        nopat = op_income * CONST_NOPAT_RATE
        res["O_nopat"] = nopat
        
        pseudo_div = nopat * CONST_PAYOUT_RATE
        res["P_pseudo_div"] = pseudo_div

        if equity > 0:
            pseudo_roe = nopat / equity
            res["Q_pseudo_roe"] = pseudo_roe
            
            roe_val = pseudo_roe
            mult = 1.5
            roe_cls = "10%未満"
            if roe_val >= 0.20:
                mult = 4.0
                roe_cls = "20%以上"
            elif roe_val >= 0.15:
                mult = 3.0
                roe_cls = "15-20%"
            elif roe_val >= 0.10:
                mult = 2.0
                roe_cls = "10-15%"
            
            res["R_roe_class"] = roe_cls
            res["S_7y_mult"] = mult
            
            fut_div = pseudo_div * mult
            res["T_7y_div"] = fut_div
            
            if cap > 0:
                fut_yield = fut_div / cap
                res["U_fut_yield"] = fut_yield
                upside = fut_yield / CONST_MARKET_YIELD
                res["W_upside"] = round(upside, 2)
                
                if current_price:
                    target = current_price * upside
                    res["Y_target"] = round(target, 0)
                    if upside >= 2.0:
                        res["Z_final"] = "合格"

    except Exception:
        pass # ログ抑制

    return format_result(res)

def format_result(r):
    row = [
        r["AA_name"], r["AB_sector"],
        r["B_cost_ratio"], r["C_judge1"],
        r["D_payout"], r["E_judge2"],
        r["F_cagr"], r["G_judge3"],
        r["X_price"], r["Y_target"], r["Z_final"],
        r["H_cap"], r["I_shares"], r["J_equity"], r["K_op_income"], r["L_date"],
        r["M_nopat_k"], r["N_div_k"],
        r["O_nopat"], r["P_pseudo_div"], r["Q_pseudo_roe"],
        r["R_roe_class"], r["S_7y_mult"], r["T_7y_div"],
        r["U_fut_yield"], r["V_mkt_yield"],
        r["W_upside"]
    ]
    cleaned_row = []
    for x in row:
        if x is None:
            cleaned_row.append("")
            continue
        if isinstance(x, float):
            if np.isinf(x) or np.isnan(x):
                cleaned_row.append("")
                continue
        cleaned_row.append(x)
    return cleaned_row

# ==========================================
# 4. メイン処理 (バッチ処理化)
# ==========================================

def main():
    open_flg, reason = is_market_open()
    if not open_flg:
        print(f"Market Closed: {reason}")
        return

    config = get_config_from_env()
    sheet = connect_gsheet(config)

    # ヘッダー (変更なし)
    headers = [
        "会社名", "業種",
        "①営業費用売上比率", "①判定",
        "②配当性向(%)", "②判定",
        "③売上高CAGR(%)", "③判定",
        "直近終値", "目標株価", "最終判定",
        "時価総額", "発行済株式数", "自己資本", "営業利益", "決算期",
        "NOPAT係数", "擬似配当係数",
        "NOPAT", "擬似配当", "擬似ROE(%)",
        "ROE区分", "7年後配当倍率", "7年後配当",
        "将来利回り(%)", "市場平均配当利回り",
        "上値目途(倍率)"
    ]
    sheet.update(range_name="B1:AB1", values=[headers])
    
    # 銘柄読み込み
    raw_tickers = sheet.col_values(1)[1:]
    tickers = []
    for t in raw_tickers:
        if t:
            t_str = str(t).strip()
            if not t_str.endswith(".T"): t_str += ".T"
            tickers.append(t_str)
    
    total_tickers = len(tickers)
    print(f"Total Tickers: {total_tickers}")
    
    # --- バッチ処理ロジック ---
    # 2800銘柄を 50件ずつの塊(Batch)にして処理・書き込みを行う
    BATCH_SIZE = 50
    
    # 全体の処理インデックス
    current_index = 0

    while current_index < total_tickers:
        # 今回処理するバッチの範囲
        end_index = min(current_index + BATCH_SIZE, total_tickers)
        batch_tickers = tickers[current_index:end_index]
        
        print(f"Processing batch: {current_index + 1} - {end_index} / {total_tickers}")

        # 1. バッチ分の株価一括取得
        price_cache = {}
        try:
            # yfinanceのdownloadログを抑制しつつ取得
            df_p = yf.download(batch_tickers, period="1d", group_by='ticker', threads=True, progress=False)
            for t in batch_tickers:
                try:
                    if len(batch_tickers) > 1:
                        price = df_p[t]['Close'].iloc[-1]
                    else:
                        price = df_p['Close'].iloc[-1]
                    price_cache[t] = float(price)
                except:
                    price_cache[t] = None
        except Exception:
            pass # ログ抑制

        # 2. バッチ分の分析 (並列処理)
        # サーバー負荷を考慮し workers は少なめに維持
        batch_results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(analyze_stock, t, price_cache): t for t in batch_tickers}
            
            for future in futures:
                t = futures[future]
                try:
                    res_list = future.result()
                    batch_results[t] = res_list
                except Exception:
                    batch_results[t] = [""] * 27

        # 3. バッチ分の結果を整形
        output_rows = []
        for t in batch_tickers:
            output_rows.append(batch_results.get(t, [""] * 27))

        # 4. スプレッドシートへ追記書き込み
        # 書き込み開始行: ヘッダ(1行) + 既に処理した行数 + 1(1-based index) => current_index + 2
        start_row = current_index + 2
        end_row = start_row + len(output_rows) - 1
        range_name = f"B{start_row}:AB{end_row}"
        
        try:
            sheet.update(range_name=range_name, values=output_rows)
            # API制限回避のための待機
            time.sleep(2) 
        except Exception as e:
            print(f"Sheet Update Error at batch {current_index}: {e}")

        # 次のバッチへ
        current_index += BATCH_SIZE
        
        # バッチ間にも少し待機を入れてサーバーを休ませる
        time.sleep(3)

    print("All processing completed.")

if __name__ == "__main__":
    main()
