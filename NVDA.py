import yfinance as yf
import feedparser
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

ticker = "NVDA"

end_date = datetime.today().strftime('%Y-%m-%d')


nvda_data = yf.download(ticker, start="2020-01-01", end=end_date)

nvda_data['MA_5'] = SMAIndicator(nvda_data['Close'], window=5).sma_indicator()
nvda_data['MA_20'] = SMAIndicator(nvda_data['Close'], window=20).sma_indicator()
nvda_data['RSI'] = RSIIndicator(nvda_data['Close'], window=14).rsi()

nvda_data = nvda_data.bfill()


rss_url = "https://news.google.com/rss/search?q=NVDA"

feed = feedparser.parse(rss_url)
articles = []

for entry in feed.entries:
    if 'link' in entry:
        url = entry.link
        try:
            analysis = TextBlob(entry.title)  
            sentiment_score = analysis.sentiment.polarity
            
            date = datetime(*entry.published_parsed[:6])
            articles.append({
                "date": date,
                "sentiment": sentiment_score
            })
        except Exception as e:
            print(f"无法处理 {url}: {e}")

articles_df = pd.DataFrame(articles)

articles_df.set_index('date', inplace=True)


articles_daily = articles_df.resample('D').sum()


scaler = StandardScaler()
articles_daily['sentiment'] = scaler.fit_transform(articles_daily[['sentiment']])

nvda_data.index = nvda_data.index.date
combined_df = nvda_data.join(articles_daily[['sentiment']], how='left')


combined_df = combined_df.bfill()


X = combined_df[['sentiment', 'Volume', 'MA_5', 'MA_20', 'RSI']]
y = combined_df['Close']


X = X.dropna()
y = y[X.index]  
y = y.dropna()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
print('LinearRegression score:', model.score(X_train, y_train))
print('LinearRegression score:', model.score(X_test, y_test))

gbr = GradientBoostingRegressor(n_estimators=400)
gbr.fit(X_train, y_train)
print('GradientBoostingRegressor score:', gbr.score(X_train, y_train))
print('GradientBoostingRegressor score:', gbr.score(X_test, y_test))

clf = DecisionTreeRegressor(criterion='squared_error',max_depth=3, random_state=0)
clf.fit(X_train, y_train)
print('DecisionTreeRegressor score:', clf.score(X_train, y_train))
print('DecisionTreeRegressor score:', clf.score(X_test, y_test))

forest = RandomForestRegressor(criterion='squared_error', n_estimators=10,random_state=0,n_jobs=8) 
forest.fit(X_train,y_train)
print('rf score:', forest.score(X_train, y_train))
print('rf score:', forest.score(X_test, y_test))


latest_data = yf.download(ticker, start="2024-08-20", end="2024-08-21")  
latest_volume = latest_data['Volume'].iloc[-1]  
latest_close = latest_data['Close'].iloc[-1]  

latest_ma_5 = SMAIndicator(nvda_data['Close'], window=5).sma_indicator().iloc[-1]
latest_ma_20 = SMAIndicator(nvda_data['Close'], window=20).sma_indicator().iloc[-1]
latest_rsi = RSIIndicator(nvda_data['Close'], window=14).rsi().iloc[-1]

latest_sentiment = articles_daily['sentiment'].iloc[-1]  

X_new = pd.DataFrame({
    'sentiment': [latest_sentiment],
    'Volume': [latest_volume],
    'MA_5': [latest_ma_5],
    'MA_20': [latest_ma_20],
    'RSI': [latest_rsi]
})

# 使用训练好的模型进行预测
predicted_close_lr = model.predict(X_new)
predicted_close_gbr = gbr.predict(X_new)
predicted_close_dtr = clf.predict(X_new)
predicted_close_rf = forest.predict(X_new)

# 输出预测结果
print(f"Predicted Close Price using Linear Regression: {predicted_close_lr[0]}")
print(f"Predicted Close Price using Gradient Boosting: {predicted_close_gbr[0]}")
print(f"Predicted Close Price using DecisionTreeRegressor: {predicted_close_dtr[0]}")
print(f"Predicted Close Price using rf: {predicted_close_rf[0]}")