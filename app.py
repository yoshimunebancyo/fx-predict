import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import pytz

st.set_page_config(page_title="FX 10分後予測AI", page_icon="📈", layout="centered")

def get_market_data():
    # ドル円、米国10年債利回り、日経平均の3つを取得
    tickers = ["JPY=X", "^TNX", "^N225"]
    df = yf.download(tickers, period="5d", interval="1m", progress=False)
    
    if df.empty:
        return None
        
    # MultiIndexの解消（仕様変更対策）
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
        
    # 列名の整理
    df = df.rename(columns={'JPY=X': 'USDJPY', '^TNX': 'US10Y', '^N225': 'NIKKEI'})
    
    # 日経平均などが閉まっている時間は直前の価格で埋める（重要）
    df = df.ffill()
    return df.dropna()

def add_advanced_features(df):
    # 1. テクニカル指標
    df['SMA_5'] = df['USDJPY'].rolling(window=5).mean()
    df['SMA_15'] = df['USDJPY'].rolling(window=15).mean()
    df['STD_10'] = df['USDJPY'].rolling(window=10).std()
    
    # 2. 各市場の変化率（モメンタム）
    df['Ret_USDJPY'] = df['USDJPY'].pct_change()
    df['Ret_US10Y'] = df['US10Y'].pct_change()
    df['Ret_NIKKEI'] = df['NIKKEI'].pct_change()
    
    # 3. ラグ特徴量（過去3分間の動きを記憶させる）
    for i in range(1, 4):
        df[f'Lag_{i}_USD'] = df['Ret_USDJPY'].shift(i)
        df[f'Lag_{i}_US10Y'] = df['Ret_US10Y'].shift(i)
        
    # 4. 時間帯特徴量（日本時間に変換して「時」と「分」を取得）
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df_jst = df.index.tz_convert('Asia/Tokyo')
    df['Hour'] = df_jst.hour
    df['Minute'] = df_jst.minute
    
    return df.dropna()

st.title("📈 FX 10分後予測AI (Pro版)")
st.write("ドル円、米国債利回り、日経平均の相関関係をAI（勾配ブースティング）が分析します。")

if st.button("最新データで予測を実行", type="primary"):
    with st.spinner("世界市場のデータを取得・ディープ分析中..."):
        
        df = get_market_data()
        
        if df is None:
            st.error("データの取得に失敗しました。時間をおいて再試行してください。")
        else:
            # 特徴量の作成
            df = add_advanced_features(df)
            
            # 目的変数：10分後の価格が現在より高ければ1、そうでなければ0
            df['Target'] = (df['USDJPY'].shift(-10) > df['USDJPY']).astype(int)
            
            # 未来がNaNになる直近10行を削除した学習用データ
            df_train = df.dropna()
            
            # AIに学習させる判断材料（特徴量リスト）
            features = [
                'USDJPY', 'US10Y', 'NIKKEI', 'SMA_5', 'SMA_15', 'STD_10', 
                'Ret_USDJPY', 'Ret_US10Y', 'Ret_NIKKEI', 
                'Lag_1_USD', 'Lag_2_USD', 'Lag_3_USD',
                'Lag_1_US10Y', 'Lag_2_US10Y', 'Lag_3_US10Y',
                'Hour', 'Minute'
            ]
            
            X = df_train[features]
            y = df_train['Target']
            
            # モデルの学習 (LightGBMと同等の強力なアルゴリズム)
            model = HistGradientBoostingClassifier(random_state=42, max_iter=100)
            model.fit(X, y)
            
            # 最新の状況を取得して予測
            latest_data = df.iloc[-1][features].values.reshape(1, -1)
            prediction = model.predict(latest_data)
            probability = model.predict_proba(latest_data)[0]
            
            # --- 結果の表示 ---
            st.divider()
            st.subheader("📊 予測結果")
            
            # 現在の価格を取得して数値化（Seriesエラー対策済み）
            current_usd = float(df['USDJPY'].iloc[-1])
            current_us10y = float(df['US10Y'].iloc[-1])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="現在のUSD/JPY", value=f"{current_usd:.3f} 円")
            with col2:
                st.metric(label="現在の米国10年債利回り", value=f"{current_us10y:.3f} %")
            
            up_prob = probability[1] * 100
            down_prob = probability[0] * 100
            
            st.write("")
            st.write("### 10分後の変動確率")
            st.progress(probability[1])
            
            col3, col4 = st.columns(2)
            with col3:
                st.info(f"🔺 上昇確率: **{up_prob:.1f}%**")
            with col4:
                st.warning(f"🔻 下降確率: **{down_prob:.1f}%**")
            
            st.write("")
            # 強いサインが出た時だけ強調表示（55%以上を基準とする）
            if up_prob >= 55.0:
                st.success("🤖 AI判定: **上昇優位（買いサイン）**")
            elif down_prob >= 55.0:
                st.error("🤖 AI判定: **下降優位（売りサイン）**")
            else:
                st.info("🤖 AI判定: **方向感なし（様子見推奨）**")
                
            st.caption("※確率が55%未満の時は「様子見」を推奨します。相場は他市場の急なニュースで変動する可能性があるため、自己責任でご活用ください。")
