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
# 2. スクレイピング & 通信ヘルパー (buhin.py統合版)
# ==========================================

def create_session():
    """リトライ付きセッション作成"""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
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
    """
    Yahoo!ファイナンス(日本)から以下を取得
    1. 配当性向 (優先)
    2. 銘柄名
    3. 業種
    """
    code_only = str(ticker_code).replace(".T", "")
    
    # 配当ページ
    url_div = f"https://finance.yahoo.co.jp/quote/{code_only}.T/dividend"
    # プロフィールページ(業種・社名用)
    url_prof = f"https://finance.yahoo.co.jp/quote/{code_only}.T/profile"
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    data = {
        "payout_ratio": None,
        "name": str(ticker_code),
        "sector": "-"
    }

    try:
        # スクレイピング検知回避のためのランダム待機
        time.sleep(random.uniform(1.0, 2.0))

        # --- 1. 配当性向の取得 ---
        try:
            res = _HTTP_SESSION.get(url_div, headers=headers, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                # "配当性向" という文字列を含むthを探し、その隣のtdを取得
                th = soup.find("th", string=re.compile("配当性向"))
                if th:
                    td = th.find_next_sibling("td")
                    if td:
                        text = td.get_text(strip=True).replace("%", "")
                        # ハイフン等は除外
                        if text and text != "-" and text != "---":
                            data["payout_ratio"] = float(text)
        except Exception as e:
            print(f"Yahoo JP Dividend Error {ticker_code}: {e}")
        
        # --- 2. 銘柄名・業種の取得 ---
        try:
            res_prof = _HTTP_SESSION.get(url_prof, headers=headers, timeout=10)
            if res_prof.status_code == 200:
                soup = BeautifulSoup(res_prof.text, "html.parser")
                
                # 社名
                title_tag = soup.find("title")
                if title_tag:
                    # <title>トヨタ自動車(株)【7203】...
                    m = re.search(r'(.*?)【', title_tag.text)
                    if m:
                        data["name"] = m.group(1).strip()
                
                # 業種 (YahooファイナンスのHTML構造から推測)
                # 東証33業種リストにある単語が含まれているかチェック
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
        except Exception as e:
            print(f"Yahoo JP Profile Error {ticker_code}: {e}")

    except Exception as e:
        print(f"Yahoo JP Scraping Critical Error {ticker_code}: {e}")
    
    return data

# ==========================================
# 3. コアロジック: 山本潤式モデル
# ==========================================

def get_value(df, keys, date_col):
    """
    buhin.py から移植・改良
    財務データDataFrameから特定の日付・キーの値を取得する
    """
    if df.empty or date_col is None:
        return 0
    
    # date_colがDataFrameに存在するかチェック
    if date_col not in df.columns:
        return 0

    for key in keys:
        if key in df.index:
            val = df.loc[key, date_col]
            if pd.isna(val):
                continue
            return float(val)
    return 0

def analyze_stock(ticker, current_price_cache):
    """
    1銘柄の分析を実行。
    戻り値: スプレッドシートの1行分（B列～Z列+参照データ）のリスト
    """
    # 結果格納用辞書 (初期値: 不合格)
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
        # 参照用
        "AA_name": "", "AB_sector": ""
    }

    try:
        # --- データ取得 ---
        # 1. Yahoo Japan (配当性向優先)
        yj_data = get_yahoo_jp_info(ticker)
        res["AA_name"] = yj_data["name"]
        res["AB_sector"] = yj_data["sector"]
        
        # 2. yfinance
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        
        # 財務データ (Annual)
        fins = tk.financials
        bs = tk.balance_sheet
        
        # データが空の場合は終了
        if fins.empty or bs.empty:
            return format_result(res)

        # 現在株価 (キャッシュ または info)
        current_price = current_price_cache.get(ticker)
        if not current_price:
            current_price = info.get("currentPrice") or info.get("previousClose")
        res["X_price"] = current_price

        # 最新の決算日を取得 (列名が日付になっている)
        dates = fins.columns
        if len(dates) == 0:
            return format_result(res)
        latest_date = dates[0] # 最新

        # ---------------------------
        # フェーズ1: 3つのゲート
        # ---------------------------

        # ① 営業費用売上比率 (売上 / (売上 - 営業利益))
        revenue = get_value(fins, ['Total Revenue'], latest_date)
        op_income = get_value(fins, ['Operating Income', 'Operating Profit'], latest_date)
        
        pass_gate1 = False
        # 0除算対策
        cost = revenue - op_income
        if revenue > 0 and op_income > 0 and cost > 0:
            ratio = revenue / cost
            res["B_cost_ratio"] = round(ratio, 2)
            if ratio >= 1.15:
                res["C_judge1"] = "合格"
                pass_gate1 = True
        
        # ② 配当性向 (YahooJP優先 -> yfinance)
        payout = yj_data["payout_ratio"]
        if payout is None:
            # yfinanceのpayoutRatioは小数(0.3など)で返る
            val = info.get("payoutRatio")
            if val is not None:
                payout = val * 100
            
        pass_gate2 = False
        if payout is not None:
            res["D_payout"] = round(payout, 2)
            if 20 <= payout <= 60:
                res["E_judge2"] = "合格"
                pass_gate2 = True
        else:
            # 取得できない場合は判定不能（不合格）
            pass
        
        # ③ 4年連続増収
        # 過去4期分のデータが必要
        pass_gate3 = False
        revs = []
        if len(dates) >= 4:
            for i in range(4):
                val = get_value(fins, ['Total Revenue'], dates[i])
                revs.append(val)
            
            # revs[0]が最新、revs[3]が4年前。全て0より大きいこと
            if all(r > 0 for r in revs):
                # t > t-1 > t-2 > t-3
                if revs[0] > revs[1] > revs[2] > revs[3]:
                    res["G_judge3"] = "合格"
                    pass_gate3 = True
                
                # CAGR計算
                cagr = (revs[0] / revs[3]) ** (1/3) - 1
                res["F_cagr"] = round(cagr * 100, 2)
        else:
            # データ不足
            pass

        # ゲート通過判定
        all_gates_passed = pass_gate1 and pass_gate2 and pass_gate3

        if not all_gates_passed:
            return format_result(res) # 計算せず終了

        # ---------------------------
        # フェーズ2: 11の計算
        # ---------------------------
        
        # 基礎データ
        # 時価総額
        cap = info.get("marketCap")
        if not cap:
            return format_result(res)
        res["H_cap"] = cap

        # 発行済株式数
        shares = info.get("sharesOutstanding")
        if not shares:
            return format_result(res)
        res["I_shares"] = shares

        # 自己資本 (複数のキーを試す)
        equity = get_value(bs, ['Total Stockholder Equity', 'Total Equity', 'Stockholders Equity'], bs.columns[0])
        if equity == 0:
            return format_result(res)
        res["J_equity"] = equity

        # 営業利益 (取得済み)
        res["K_op_income"] = op_income
        
        # 決算期
        res["L_date"] = str(latest_date.date())

        # 計算開始
        # O: NOPAT
        nopat = op_income * CONST_NOPAT_RATE
        res["O_nopat"] = nopat
        
        # P: 擬似配当
        pseudo_div = nopat * CONST_PAYOUT_RATE
        res["P_pseudo_div"] = pseudo_div

        # Q: 擬似ROE
        if equity > 0:
            pseudo_roe = nopat / equity
            res["Q_pseudo_roe"] = round(pseudo_roe * 100, 2)
            
            # R: ROE区分 & S: 7年後倍率
            roe_val = pseudo_roe * 100
            mult = 1.5
            roe_cls = "10%未満"
            
            if roe_val >= 20:
                mult = 4.0
                roe_cls = "20%以上"
            elif roe_val >= 15:
                mult = 3.0
                roe_cls = "15-20%"
            elif roe_val >= 10:
                mult = 2.0
                roe_cls = "10-15%"
            
            res["R_roe_class"] = roe_cls
            res["S_7y_mult"] = mult
            
            # T: 7年後配当
            fut_div = pseudo_div * mult
            res["T_7y_div"] = fut_div
            
            # U: 将来利回り
            if cap > 0:
                fut_yield = fut_div / cap
                res["U_fut_yield"] = round(fut_yield * 100, 2)
                
                # W: 上値目途
                upside = fut_yield / CONST_MARKET_YIELD
                res["W_upside"] = round(upside, 2)
                
                # Y: 目標株価
                if current_price:
                    target = current_price * upside
                    res["Y_target"] = round(target, 0)

                    # ---------------------------
                    # フェーズ3: 最終判定
                    # ---------------------------
                    if upside >= 2.0:
                        res["Z_final"] = "合格"

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        # エラー時は現状のresを返す（不合格扱い）

    return format_result(res)

def format_result(r):
    """辞書をスプレッドシート書き込み用のリスト(B列以降)に変換"""
    # 列順序: B -> Z, AA, AB
    row = [
        r["B_cost_ratio"], r["C_judge1"],
        r["D_payout"], r["E_judge2"],
        r["F_cagr"], r["G_judge3"],
        r["H_cap"], r["I_shares"], r["J_equity"], r["K_op_income"], r["L_date"],
        r["M_nopat_k"], r["N_div_k"],
        r["O_nopat"], r["P_pseudo_div"], r["Q_pseudo_roe"],
        r["R_roe_class"], r["S_7y_mult"], r["T_7y_div"],
        r["U_fut_yield"], r["V_mkt_yield"],
        r["W_upside"], r["X_price"], r["Y_target"],
        r["Z_final"],
        # 参照
        r["AA_name"], r["AB_sector"]
    ]
    # Noneを空文字に変換
    return [x if x is not None else "" for x in row]

# ==========================================
# 4. メイン処理
# ==========================================

def main():
    # 休日チェック
    open_flg, reason = is_market_open()
    if not open_flg:
        print(f"Skipping run: {reason}")
        return

    config = get_config_from_env()
    sheet = connect_gsheet(config)

    # --- ヘッダー書き込み (1行目: B列〜AB列) ---
    headers = [
        "①営業費用売上比率", "①判定",
        "②配当性向(%)", "②判定",
        "③売上高CAGR(%)", "③判定",
        "時価総額", "発行済株式数", "自己資本", "営業利益", "決算期",
        "NOPAT係数", "擬似配当係数",
        "NOPAT", "擬似配当", "擬似ROE(%)",
        "ROE区分", "7年後配当倍率", "7年後配当",
        "将来利回り(%)", "市場平均配当利回り",
        "上値目途(倍率)", "直近終値", "目標株価",
        "最終判定",
        "会社名", "業種"
    ]
    sheet.update(range_name="B1:AB1", values=[headers])
    
    # --- A列の銘柄コード読み込み & .T付与 ---
    raw_tickers = sheet.col_values(1)[1:] # 1行目はヘッダと仮定
    tickers = []
    for t in raw_tickers:
        if t:
            t_str = str(t).strip()
            # .T を自動付与
            if not t_str.endswith(".T"):
                t_str += ".T"
            tickers.append(t_str)
    
    print(f"Target Tickers: {len(tickers)}")
    
    # 現在株価の一括取得 (高速化)
    print("Downloading stock prices...")
    price_cache = {}
    if tickers:
        try:
            # yfinanceで一括DL
            df_p = yf.download(tickers, period="1d", group_by='ticker', threads=True, progress=False)
            for t in tickers:
                try:
                    if len(tickers) > 1:
                        price = df_p[t]['Close'].iloc[-1]
                    else:
                        price = df_p['Close'].iloc[-1]
                    price_cache[t] = float(price)
                except:
                    price_cache[t] = None
        except Exception as e:
            print(f"Bulk download failed: {e}")

    # 分析実行 (並列処理)
    print("Analyzing stocks...")
    results_map = {}
    
    # サーバー負荷とスクレイピング制限を考慮し、同時実行数は控えめに(3-5)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(analyze_stock, t, price_cache): t for t in tickers}
        
        for future in futures:
            t = futures[future]
            try:
                res_list = future.result()
                results_map[t] = res_list
                print(f"Done: {t}")
            except Exception as e:
                print(f"Failed: {t} {e}")
                results_map[t] = [""] * 27 # エラー時は空行

    # 結果をリストに整形 (元のA列の順番を守る)
    output_rows = []
    for t in tickers:
        output_rows.append(results_map.get(t, [""] * 27))
        
    # スプレッドシートへ一括書き込み
    # B2 (row=2, col=2) から書き込み開始
    if output_rows:
        range_name = f"B2:AB{2 + len(output_rows) - 1}"
        sheet.update(range_name=range_name, values=output_rows)
        print("Spreadsheet updated successfully.")

if __name__ == "__main__":
    main()
