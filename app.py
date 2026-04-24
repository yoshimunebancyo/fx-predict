import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# テクニカル指標を計算する関数
def add_technical_indicators(df):
    # 単純移動平均 (SMA)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_15'] = df['Close'].rolling(window=15).mean()
    
    # RSI (14期間)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ボラティリティ (標準偏差)
    df['STD_10'] = df['Close'].rolling(window=10).std()

    # 【追加】価格の変化率（モメンタム）
    df['Change'] = df['Close'].pct_change()
    
    # 【追加】ラグ特徴量（1分前〜5分前の変化率を学習させる）
    for i in range(1, 6):
        df[f'Lag_{i}'] = df['Change'].shift(i)
        
    # 【追加】ボリンジャーバンド (2σ)
    df['Upper'] = df['SMA_5'] + (df['Close'].rolling(window=5).std() * 2)
    df['Lower'] = df['SMA_5'] - (df['Close'].rolling(window=5).std() * 2)
    
    return df.dropna()

st.title("FX 10分後予測アプリ (USD/JPY)")
st.write("直近のデータとテクニカル指標から、10分後の値動きを確率的に予測します。")

if st.button("予測を実行する"):
    with st.spinner("データを取得・分析中..."):
        # 1. データの取得 (ドル円の1分足データを過去5日分)
        ticker = "JPY=X"
        df = yf.download(ticker, period="5d", interval="1m", progress=False)
        
        if df.empty:
            st.error("データの取得に失敗しました。")
        else:
            # --- 【追加】yfinanceの仕様変更対策（列の階層をシンプルな1階層に戻す） ---
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # -----------------------------------------------------------------

            # 2. 特徴量（予測の根拠）の作成
            df = add_technical_indicators(df)
            
            # 3. 目的変数（正解データ）の作成
            # 10分後（10行下）の終値が現在の終値より高ければ1（上昇）、そうでなければ0（下降）
            df['Target'] = (df['Close'].shift(-10) > df['Close']).astype(int)
            
            # 未来のデータがNaNになる直近10行を削除
            df_train = df.dropna()
            
            # 学習に使う特徴量に追加した指標をすべて追記
            features = ['Close', 'SMA_5', 'SMA_15', 'RSI', 'STD_10', 'Change', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Upper', 'Lower']
            X = df_train[features]
            y = df_train['Target']
            
            # 4. モデルの学習 (ランダムフォレスト)
            # ※本来は時系列分割等を行いますが、今回は簡易的な最新データ予測に特化しています
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # 5. 現在の状況を取得して予測
            latest_data = df.iloc[-1][features].values.reshape(1, -1)
            prediction = model.predict(latest_data)
            probability = model.predict_proba(latest_data)[0]
            
            # 結果の表示
            st.subheader("予測結果")
            
            # --- 修正箇所 ---
            current_price = df['Close'].iloc[-1]
            if isinstance(current_price, pd.Series):
                current_price = float(current_price.iloc[0]) # Seriesの場合は最初の要素を数値として取り出す
            else:
                current_price = float(current_price)
            # --------------
                
            st.write(f"現在の価格 (USD/JPY): **{current_price:.2f} 円**")
            
            up_prob = probability[1] * 100
            down_prob = probability[0] * 100
            
            st.write(f"10分後に上昇する確率: **{up_prob:.1f}%**")
            st.write(f"10分後に下降する確率: **{down_prob:.1f}%**")
            
            # プログレスバーで視覚化
            st.progress(probability[1])
            
            if prediction[0] == 1:
                st.success("統計的予測: **上昇トレンド優勢** 📈")
            else:
                st.warning("統計的予測: **下降トレンド優勢** 📉")
                
            st.caption("※このアプリは過去のデータに基づく統計的な確率を示すものであり、投資を助言・推奨するものではありません。実際の取引は自己責任で行ってください。")