# NVIDIA (NVDA) 股票價格預測

本程式使用機器學習模型來預測 NVIDIA (NVDA) 股票的收盤價。該模型整合了技術指標（移動平均線、相對強弱指數等）及新聞情緒分析，並使用多種回歸模型來進行預測。

## 功能特色
- 透過 `yfinance` 獲取 NVDA 股票的歷史數據。
- 計算技術指標，如 5 日與 20 日移動平均線 (MA) 以及相對強弱指數 (RSI)。
- 從 Google News 爬取 NVDA 相關新聞，並使用 `TextBlob` 進行情緒分析。
- 透過 `scikit-learn` 訓練多種機器學習模型，包括:
  - 線性回歸 (Linear Regression)
  - 梯度提升回歸 (Gradient Boosting Regressor)
  - 決策樹回歸 (Decision Tree Regressor)
  - 隨機森林回歸 (Random Forest Regressor)
- 預測未來某日的收盤價格。

## 安裝需求

請確保您的環境已安裝 Python 及以下相依套件，可使用 pip 安裝：

```sh
pip install yfinance feedparser pandas textblob scikit-learn ta
```

## 使用方式

1. 執行程式，系統會自動下載 NVDA 股票歷史數據，計算技術指標，並收集新聞情緒。
2. 訓練四種不同的機器學習回歸模型，並輸出各模型的準確度分數。
3. 使用最新數據進行未來股價預測，並輸出預測結果。

## 主要程式邏輯

1. **數據收集**
   - 透過 `yfinance` 下載 NVDA 股票的歷史數據。
   - 透過 `feedparser` 從 Google News 取得 NVDA 相關新聞，並使用 `TextBlob` 進行情緒分析。

2. **特徵工程**
   - 計算 5 日及 20 日移動平均線 (MA_5, MA_20)。
   - 計算相對強弱指數 (RSI)。
   - 使用 `StandardScaler` 標準化新聞情緒數據。

3. **機器學習建模**
   - 使用 `train_test_split` 將數據分為訓練集與測試集。
   - 訓練四種不同的機器學習回歸模型。
   - 評估各模型在訓練集及測試集上的表現。

4. **預測未來股價**
   - 提取最新數據作為模型輸入，並預測未來收盤價。

## 預測結果範例

```sh
LinearRegression score: 0.85
LinearRegression score: 0.78
GradientBoostingRegressor score: 0.95
GradientBoostingRegressor score: 0.89
DecisionTreeRegressor score: 0.82
DecisionTreeRegressor score: 0.76
rf score: 0.93
rf score: 0.87

Predicted Close Price using Linear Regression: 725.34
Predicted Close Price using Gradient Boosting: 738.92
Predicted Close Price using DecisionTreeRegressor: 710.45
Predicted Close Price using Random Forest: 730.12
```

## 注意事項
- 預測結果僅供參考，股市具有高度波動性，請謹慎操作。
- 若 `feedparser` 無法成功擷取新聞，請檢查 RSS 來源是否變更。
- 運行程式前，請確保 API 服務正常運作，避免數據擷取失敗。


