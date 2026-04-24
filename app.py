import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import pytz

st.set_page_config(page_title="FX 先読みAI (Pro)", page_icon="📈", layout="centered")

def get_market_data():
    tickers = ["JPY=X", "^TNX", "^N225"]
    df = yf.download(tickers, period="5d", interval="1m", progress=False)
    
    if df.empty:
        return None
        
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
        
    df = df.rename(columns={'JPY=X': 'USDJPY', '^TNX': 'US10Y', '^N225': 'NIKKEI'})
    df = df.ffill()
    return df.dropna()

def add_advanced_features(df):
    df['SMA_5'] = df['USDJPY'].rolling(window=5).mean()
    df['SMA_15'] = df['USDJPY'].rolling(window=15).mean()
    df['SMA_60'] = df['USDJPY'].rolling(window=60).mean() # 1時間予測用に長期SMAを追加
    df['STD_10'] = df['USDJPY'].rolling(window=10).std()
    
    df['Ret_USDJPY'] = df['USDJPY'].pct_change()
    df['Ret_US10Y'] = df['US10Y'].pct_change()
    df['Ret_NIKKEI'] = df['NIKKEI'].pct_change()
    
    for i in range(1, 4):
        df[f'Lag_{i}_USD'] = df['Ret_USDJPY'].shift(i)
        df[f'Lag_{i}_US10Y'] = df['Ret_US10Y'].shift(i)
        
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df_jst = df.index.tz_convert('Asia/Tokyo')
    df['Hour'] = df_jst.hour
    df['Minute'] = df_jst.minute
    
    return df.dropna()

# 各タイムフレームの予測を行う関数
def predict_horizon(X_train, y_train, latest_features):
    model = HistGradientBoostingClassifier(random_state=42, max_iter=100)
    model.fit(X_train, y_train)
    probability = model.predict_proba(latest_features)[0]
    return probability[1] * 100, probability[0] * 100  # 上昇確率, 下降確率

st.title("📈 FX 先読みAI (Pro)")
st.write("10分後・30分後・1時間後の値動きをAIが並行分析します。")

if st.button("最新データで予測を実行", type="primary", use_container_width=True):
    with st.spinner("世界市場のデータを取得し、3つのAIモデルを構築中..."):
        
        df = get_market_data()
        
        if df is None:
            st.error("データの取得に失敗しました。")
        else:
            df = add_advanced_features(df)
            
            features = [
                'USDJPY', 'US10Y', 'NIKKEI', 'SMA_5', 'SMA_15', 'SMA_60', 'STD_10', 
                'Ret_USDJPY', 'Ret_US10Y', 'Ret_NIKKEI', 
                'Lag_1_USD', 'Lag_2_USD', 'Lag_3_USD',
                'Lag_1_US10Y', 'Lag_2_US10Y', 'Lag_3_US10Y',
                'Hour', 'Minute'
            ]
            
            # 最新の状況を抽出（予測に入力するため）
            latest_features = df[features].iloc[-1].values.reshape(1, -1)
            
            # 目的変数（10分、30分、60分後の正解データ）を作成
            df['Target_10'] = (df['USDJPY'].shift(-10) > df['USDJPY']).astype(int)
            df['Target_30'] = (df['USDJPY'].shift(-30) > df['USDJPY']).astype(int)
            df['Target_60'] = (df['USDJPY'].shift(-60) > df['USDJPY']).astype(int)
            
            # 学習データの作成（未来の答えがまだ出ていない直近60分を削る）
            df_train = df.dropna(subset=['Target_10', 'Target_30', 'Target_60'])
            X_train = df_train[features]
            
            # 各時間軸の予測を実行
            up_10, down_10 = predict_horizon(X_train, df_train['Target_10'], latest_features)
            up_30, down_30 = predict_horizon(X_train, df_train['Target_30'], latest_features)
            up_60, down_60 = predict_horizon(X_train, df_train['Target_60'], latest_features)
            
            # --- 結果の表示 ---
            st.divider()
            
            current_usd = float(df['USDJPY'].iloc[-1])
            current_us10y = float(df['US10Y'].iloc[-1])
            
            st.markdown(f"### 現在の価格: **{current_usd:.3f} 円**")
            st.caption(f"米国10年債利回り: {current_us10y:.3f}%")
            st.write("")
            
            # 総合トレンド判定（3つの時間軸がすべて同じ方向を示しているか）
            up_count = sum([up_10 >= 55, up_30 >= 55, up_60 >= 55])
            down_count = sum([down_10 >= 55, down_30 >= 55, down_60 >= 55])
            
            if up_count == 3:
                st.success("🔥 **総合AI判定: 強い上昇トレンド（全時間軸が一致）**")
            elif down_count == 3:
                st.error("📉 **総合AI判定: 強い下降トレンド（全時間軸が一致）**")
            elif up_count >= 2:
                st.info("🔺 **総合AI判定: やや上昇優位**")
            elif down_count >= 2:
                st.warning("🔻 **総合AI判定: やや下降優位**")
            else:
                st.write("🔄 **総合AI判定: 方向感なし（様子見推奨）**")
            
            st.write("")
            
            # スマホで縦並びになるように、3つのコンテナで表示
            with st.container():
                st.write("#### ⏱️ 10分後の予測")
                st.progress(up_10 / 100)
                st.write(f"🔺上昇: **{up_10:.1f}%** ｜ 🔻下降: **{down_10:.1f}%**")
                
            st.write("---")
            
            with st.container():
                st.write("#### ⏱️ 30分後の予測")
                st.progress(up_30 / 100)
                st.write(f"🔺上昇: **{up_30:.1f}%** ｜ 🔻下降: **{down_30:.1f}%**")

            st.write("---")
                
            with st.container():
                st.write("#### ⏱️ 1時間後 (60分後) の予測")
                st.progress(up_60 / 100)
                st.write(f"🔺上昇: **{up_60:.1f}%** ｜ 🔻下降: **{down_60:.1f}%**")
                
            st.write("")
            st.caption("※確率が55%以上の時に優位性があると判断します。1時間後など先の予測ほど、突発的なニュースの影響を受けやすくなる点にご注意ください。")
