import os
import json
import feedparser
import yfinance as yf
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import plotly.graph_objects as go
from flask import Flask, render_template_string


# 多隻股票分析
tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META"]
print(f"分析股票: {', '.join(tickers)}")

end_date = datetime.today().strftime('%Y-%m-%d')

# 儲存所有股票的結果
all_results = {}

# 通用新聞情緒分析函數
def get_weighted_sentiment(ticker):
    # 針對不同股票的RSS來源
    rss_sources = [
        f"https://news.google.com/rss/search?q={ticker}",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://feeds.marketwatch.com/marketwatch/StockPulse?symbol={ticker}",
        f"https://news.google.com/rss/search?q={ticker}+stock",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"  # Dow Jones
    ]
    
    all_articles = []
    
    # 重要性關鍵字權重
    importance_keywords = {
        'earnings': 3.0,
        'revenue': 2.5,
        'profit': 2.5,
        'guidance': 2.5,
        'AI': 2.0,
        'datacenter': 2.0,
        'gaming': 1.5,
        'chip': 1.5,
        'semiconductor': 1.5,
        'acquisition': 2.0,
        'partnership': 1.5,
        'breakthrough': 2.0,
        'crisis': 2.0,
        'lawsuit': 1.8,
        'regulation': 1.8
    }
    
    for rss_url in rss_sources:
        try:
            print(f"  Fetching {ticker} news from: {rss_url}")
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries:
                if 'link' in entry:
                    try:
                        title = entry.title.lower()
                        # 確保新聞與該股票相關
                        if ticker.lower() in title or any(keyword in title for keyword in [ticker.lower(), 'stock', 'market']):
                            analysis = TextBlob(entry.title)
                            sentiment_score = analysis.sentiment.polarity
                            
                            # 根據關鍵字計算重要性權重
                            importance_weight = 1.0
                            for keyword, weight in importance_keywords.items():
                                if keyword in title:
                                    importance_weight = max(importance_weight, weight)
                            
                            # 解析日期
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                date = datetime(*entry.published_parsed[:6])
                            else:
                                date = datetime.now()
                            
                            weighted_sentiment = sentiment_score * importance_weight
                            
                            all_articles.append({
                                "date": date,
                                "sentiment": sentiment_score,
                                "weighted_sentiment": weighted_sentiment,
                                "importance_weight": importance_weight,
                                "title": entry.title,
                                "source": rss_url
                            })
                            
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"  Error fetching from {rss_url}: {e}")
            continue
    
    return all_articles

# 分析每隻股票
for ticker in tickers:
    print(f"\n{'='*50}")
    print(f"分析股票: {ticker}")
    print(f"{'='*50}")
    
    try:
        # 下載股票資料
        stock_data = yf.download(ticker, start="2020-01-01", end=end_date)
        
        # 處理多層欄位名稱
        if stock_data.columns.nlevels > 1:
            stock_data.columns = stock_data.columns.get_level_values(0)
        
        # 確保Close資料是一維的
        close_series = stock_data['Close'].squeeze()
        
        # 計算技術指標
        stock_data['MA_5'] = SMAIndicator(close=close_series, window=5).sma_indicator()
        stock_data['MA_20'] = SMAIndicator(close=close_series, window=20).sma_indicator()
        stock_data['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
        
        #  將技術指標延遲1天
        stock_data['MA_5_lag'] = stock_data['MA_5'].shift(1)
        stock_data['MA_20_lag'] = stock_data['MA_20'].shift(1)
        stock_data['RSI_lag'] = stock_data['RSI'].shift(1)
        stock_data['Volume_lag'] = stock_data['Volume'].shift(1)
        
        stock_data = stock_data.bfill()
        
        # 獲取該股票的情緒分析
        print(f"Fetching news sentiment data for {ticker}...")
        articles = get_weighted_sentiment(ticker)
        print(f"Total articles collected for {ticker}: {len(articles)}")
        
        if articles:
            articles_df = pd.DataFrame(articles)
            articles_df.set_index('date', inplace=True)
            
            # 每日情緒彙總
            articles_daily = articles_df.groupby(articles_df.index.date).agg({
                'sentiment': 'mean',
                'weighted_sentiment': 'mean',
                'importance_weight': 'mean'
            })
            
            articles_daily['final_sentiment'] = articles_daily['weighted_sentiment']
            
        else:
            print(f"No articles found for {ticker}, using dummy sentiment data")
            articles_daily = pd.DataFrame({
                'final_sentiment': [0.0] * len(stock_data)
            }, index=stock_data.index)
        
        # 標準化情緒分數
        scaler = StandardScaler()
        if len(articles_daily) > 0:
            articles_daily['final_sentiment'] = scaler.fit_transform(articles_daily[['final_sentiment']])
        
        # 修正索引以便合併
        stock_data.index = pd.to_datetime(stock_data.index.date)
        articles_daily.index = pd.to_datetime(articles_daily.index)
        
        # 將情緒分數延遲1天以避免資料洩漏
        articles_daily['sentiment_lag'] = articles_daily['final_sentiment'].shift(1)
        
        combined_df = stock_data.join(articles_daily[['sentiment_lag']], how='left')
        combined_df = combined_df.bfill()
        
        #  使用延遲特徵避免資料洩漏
        X = combined_df[['sentiment_lag', 'Volume_lag', 'MA_5_lag', 'MA_20_lag', 'RSI_lag']]
        y = combined_df['Close']
        
        # 移除包含NaN的行
        X = X.dropna()
        y = y[X.index]
        y = y.dropna()
        
        # 確保有足夠的資料
        if len(X) < 50:
            print(f"Warning: Limited data available for {ticker}")
            continue
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 訓練模型
        models = {
            'LinearRegression': LinearRegression(),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=400),
            'DecisionTree': DecisionTreeRegressor(criterion='squared_error', max_depth=3, random_state=0),
            'RandomForest': RandomForestRegressor(criterion='squared_error', n_estimators=10, random_state=0, n_jobs=8)
        }
        
        model_results = {}
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            print(f'{model_name} score (train): {train_score:.4f}')
            print(f'{model_name} score (test): {test_score:.4f}')
            
            model_results[model_name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score
            }
        
        print(f"\n=== {ticker} 價格預測 ===")
        
        # 獲取最新資料點的特徵
        latest_sentiment = float(articles_daily['sentiment_lag'].iloc[-1]) if len(articles_daily) > 0 and not pd.isna(articles_daily['sentiment_lag'].iloc[-1]) else 0.0
        latest_volume = float(stock_data['Volume_lag'].iloc[-1])
        latest_ma_5 = float(stock_data['MA_5_lag'].iloc[-1])
        latest_ma_20 = float(stock_data['MA_20_lag'].iloc[-1])
        latest_rsi = float(stock_data['RSI_lag'].iloc[-1])
        
        print(f"使用前一日特徵:")
        print(f"Sentiment: {latest_sentiment:.4f}")
        print(f"Volume: {latest_volume:,.0f}")
        print(f"MA_5: {latest_ma_5:.2f}")
        print(f"MA_20: {latest_ma_20:.2f}")
        print(f"RSI: {latest_rsi:.2f}")
        
        X_new = pd.DataFrame({
            'sentiment_lag': [latest_sentiment],
            'Volume_lag': [latest_volume],
            'MA_5_lag': [latest_ma_5],
            'MA_20_lag': [latest_ma_20],
            'RSI_lag': [latest_rsi]
        })
        
        # 獲取當前實際價格進行比較
        current_close = float(stock_data['Close'].iloc[-1])
        
        print(f"\n當前收盤價: ${current_close:.2f}")
        
        predictions = {}
        for model_name, model_info in model_results.items():
            predicted_price = model_info['model'].predict(X_new)[0]
            change_percent = ((predicted_price - current_close) / current_close * 100)
            predictions[model_name] = {
                'predicted_price': predicted_price,
                'change_percent': change_percent
            }
            print(f"使用 {model_name} 預測價格: ${predicted_price:.2f} ({change_percent:+.2f}%)")
        
        # 儲存該股票的結果
        all_results[ticker] = {
            'current_price': current_close,
            'model_scores': model_results,
            'predictions': predictions,
            'latest_features': X_new.iloc[0].to_dict()
        }
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        continue

#  總結所有股票的預測結果
print(f"\n{'='*80}")
print("所有股票預測總結")
print(f"{'='*80}")

for ticker, results in all_results.items():
    print(f"\n{ticker} - 當前價格: ${results['current_price']:.2f}")
    print("-" * 50)
    
    # 顯示最佳模型（基於測試分數）
    best_model = max(results['model_scores'].items(), key=lambda x: x[1]['test_score'])
    best_model_name = best_model[0]
    best_prediction = results['predictions'][best_model_name]
    
    print(f"最佳模型: {best_model_name} (測試分數: {best_model[1]['test_score']:.4f})")
    print(f"預測價格: ${best_prediction['predicted_price']:.2f}")
    print(f"預期變化: {best_prediction['change_percent']:+.2f}%")
    
    # 顯示所有模型的平均預測
    avg_prediction = sum([pred['predicted_price'] for pred in results['predictions'].values()]) / len(results['predictions'])
    avg_change = ((avg_prediction - results['current_price']) / results['current_price'] * 100)
    print(f"平均預測: ${avg_prediction:.2f} ({avg_change:+.2f}%)")

print(f"\n{'='*80}")
print("分析完成！")
print(f"{'='*80}")

def make_json_safe(results_dict):
    safe_dict = {}
    for ticker, result in results_dict.items():
        safe_scores = {}
        for model_name, info in result['model_scores'].items():
            safe_scores[model_name] = {
                'train_score': info['train_score'],
                'test_score': info['test_score']
            }
        safe_dict[ticker] = {
            'current_price': result['current_price'],
            'model_scores': safe_scores,
            'predictions': result['predictions'],
            'latest_features': result['latest_features']
        }
    return safe_dict

# 儲存分析結果到 JSON
safe_results = make_json_safe(all_results)
with open("stock_results.json", "w") as f:
    json.dump(safe_results, f)

# ------------------ 啟動 Flask 伺服器 ------------------
app = Flask(__name__)

def generate_comparison_chart():
    tickers = []
    actual_prices = []
    predicted_prices = []
    for ticker, results in all_results.items():
        best_model = max(results['model_scores'].items(), key=lambda x: x[1]['test_score'])
        tickers.append(ticker)
        actual_prices.append(results['current_price'])
        predicted_prices.append(results['predictions'][best_model[0]]['predicted_price'])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=tickers, y=actual_prices, name='實際價格'))
    fig.add_trace(go.Bar(x=tickers, y=predicted_prices, name='預測價格'))
    fig.update_layout(title='股票實際 vs 預測價格', xaxis_title='股票代號', yaxis_title='價格 (USD)', barmode='group')
    return fig.to_html(full_html=False)

@app.route('/')
def index():
    chart_html = generate_comparison_chart()
    summary_html = ""
    for ticker, results in all_results.items():
        best_model = max(results['model_scores'].items(), key=lambda x: x[1]['test_score'])
        pred = results['predictions'][best_model[0]]

        # 計算平均預測
        preds = [p['predicted_price'] for p in results['predictions'].values()]
        avg_pred = sum(preds) / len(preds)
        avg_change = ((avg_pred - results['current_price']) / results['current_price']) * 100

        summary_html += f"""
        <h2>{ticker}</h2>
        <ul>
            <li>最新一筆價格: ${results['current_price']:.2f}</li>
            <li>預測價格 ({best_model[0]}): ${pred['predicted_price']:.2f}</li>
            <li>預期變化: {pred['change_percent']:+.2f}%</li>
        </ul>
        """

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>股票預測儀表板</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                background-color: #f0f2f5;
                color: #333;
                margin: 0;
                padding: 0;
            }
            h1 {
                background-color: #1f77b4;
                color: white;
                padding: 20px;
                margin: 0;
                text-align: center;
            }
            .container {
                padding: 20px;
                max-width: 1000px;
                margin: auto;
            }
            .card {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                padding: 20px;
            }
            ul {
                list-style-type: none;
                padding-left: 0;
            }
            li {
                margin: 8px 0;
            }
        </style>
    </head>
    <body>
        <h1>📈 股票預測儀表板</h1>
        <div class="container">
            {{ chart_html | safe }}
            <hr>
            {{ summary_html | safe }}
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, chart_html=chart_html, summary_html=summary_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
