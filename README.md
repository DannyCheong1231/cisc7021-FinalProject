# SST-5 情感分類：Frozen LLM + MLP

本專案研究：保留預訓練語言模型的權重，只訓練小型分類頭，能否完成英文電影評論的五級情感分類。主要實驗使用 TinyLlama 1.1B，另保留 42M 課程模型作對照。

## 實驗架構

```text
電影評論
   ↓
Tokenizer：轉為 token IDs 與 attention mask
   ↓
凍結 LLM：抽取最後一層、最後一個非 padding token 的向量
   ↓
MLP 分類頭：LayerNorm → Linear → ReLU → Dropout → Linear
   ↓
五個情感類別的分數 → 預測類別／顯示等級
```

對 TinyLlama 1.1B 而言，每句評論的特徵向量為 **2048 維**，原分類頭的隱藏層為 **128 維**，輸出為 **5 類**。LLM 全程使用 eval / no_grad 並凍結參數；只有 LayerNorm 和分類頭參與訓練。

分類頭使用 cross-entropy loss 和 AdamW。特徵可以先儲存成 `features.pt`，後續直接重用，無須每次重新執行 LLM。

## 資料與評估流程

使用 Stanford 更新版 SST-5，保留原有切分，只取完整句子的根節點情感標籤：

| 資料 | 句數 | 用途 |
|---|---:|---|
| Train | 8,527 | 更新分類頭權重 |
| Validation | 1,095 | 選擇 epoch 及比較實驗設定 |
| Test | 2,203 | 評估選定模型 |

內部標籤 `0–4` 依序表示「非常負面、負面、中性、正面、非常正面」，顯示時對應 `1–5`。

每輪訓練後記錄 Train / Validation 指標，以 **validation macro-F1** 保存最佳 checkpoint，最後評估 test。主要評估指標為 accuracy、macro-F1、各類別指標及混淆矩陣。

## 各程式的作用

| 程式 | 功能 |
|---|---|
| `sst5.py` | 核心入口：資料下載與處理、模型載入、特徵抽取、分類頭訓練及新評論預測。支援從文字抽取特徵，或讀取已有特徵重新訓練。 |
| `verify_results.py` | 載入已儲存的分類頭與特徵，重新計算 test 預測，核對 CSV、標籤映射、accuracy、macro-F1 及混淆矩陣。 |
| `plot_results.py` | 讀取實驗紀錄，繪製 Train / Validation loss、accuracy、macro-F1，以及實驗比較圖。 |
| `test_sst5.py` | 檢查資料標籤、padding、LLM 凍結、分類頭更新、權重重載、特徵重用及正則化參數是否正確。 |

### `sst5.py` 的主要組件

| 組件 | 職責 |
|---|---|
| `parse_tree` / `load_sst` | 解析 Stanford 樹狀資料，取得句子及標籤。 |
| `load_encoder` / `FrozenEncoder` / `last_token` | 載入並凍結 LLM，選取最後一個有效 token 的隱藏狀態。 |
| `extract` | 批次將評論轉為特徵向量。 |
| `Head` | 定義分類頭；可調隱藏層寬度和 dropout，寬度為 0 時使用線性頭。 |
| `train` / `fit_head` | 準備特徵、訓練分類頭、依 validation 選擇 checkpoint，並儲存結果。 |
| `scores` / `evaluate` | 計算分類指標。 |
| `predict` | 將新評論轉為五類情感預測及機率。 |

## 資料夾與輸出檔案

| 項目 | 內容 |
|---|---|
| `data/` | 固定版本的 Stanford 原始資料。 |
| `runs/` | 各次實驗的設定、權重、特徵、指標、預測及比較報告。 |
| `figures/` | 已產生的部分模型比較圖。 |
| `head.pt` | 分類頭權重與實驗設定，不包含完整 LLM。 |
| `features.pt` | 凍結 LLM 產生的特徵向量及標籤。 |
| `metrics.json` | 逐輪訓練紀錄、最佳 epoch 及評估結果。 |
| `test_predictions.csv` | 每句 test 評論的標準答案與預測。 |
| `training.png` | 根據實驗紀錄產生的訓練曲線。 |
| `requirements.txt` / `requirements-lock.txt` | 套件需求及參考環境的版本紀錄。 |
| `requirements-visualization.txt` | 繪圖套件需求。 |

## 可修改的實驗項目

一般參數由 `sst5.py` 的命令列介面控制，包括 learning rate、epochs、seed、分類頭大小、dropout、weight decay 和 batch size。模型、模型版本及 token 長度在重新抽取特徵時指定；重用快取時則沿用其特徵設定。

修改分類頭架構可閱讀 `Head`；修改特徵選取方式可閱讀 `FrozenEncoder`；修改訓練流程可閱讀 `fit_head`。比較設定時應保持資料切分一致，並以 validation 選擇模型。

歷史對照結果保存在本地 `runs/`，包括 learning rate、分類頭大小及 dropout / weight decay 比較；此目錄預設不隨 Git 上傳。目前以單一 seed 為主，尚未驗證多 seed 穩定性及中文／粵語評論表現。
