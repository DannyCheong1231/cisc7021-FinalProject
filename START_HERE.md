# 從這裡開始：自己運行並理解 SST-5 實驗

本機專案：`C:\nlp_project`。Python 環境、模型快取及已訓練結果都已備好。
本機 `.venv` 的基礎 Python 現在位於 `C:\nlp_project\.python312`，
不再依賴 AppData 下 uv 管理的 Python 路徑。
以下指令在 Windows PowerShell 執行，不是貼進 Python 的 `>>>` 視窗。
打開開始功能表，搜尋 PowerShell 並開啟，先執行：

```powershell
cd C:\nlp_project
$env:PYTHONIOENCODING='utf-8'
```

## 第一步：自己重算 accuracy（不用重新訓練）

```powershell
.venv\Scripts\python.exe verify_results.py --run runs/sst5-1.1b
```

你會看到：

```text
PASS: loaded saved head and recomputed all 2203 test predictions.
PASS: CSV predictions, 0-4 to 1-5 mapping, confusion matrix and metrics agree.
PASS: checkpoint selected by validation macro-F1: epoch 15.
Correct: 1196/2203; accuracy: 0.542896; macro-F1: 0.527522
```

accuracy = 判對句數 / 全部句數 = 1196 / 2203 = 54.29%。
此指令真的載入分類頭並計算預測，不只是讀取 metrics.json 印出數字。
但 LLM 的輸出使用已儲存的 features.pt，不會重新跑 LLM。
最後兩行英文說明的也是這個範圍。

比較 42M：

```powershell
.venv\Scripts\python.exe verify_results.py --run runs/sst5-42m-curves
```

預期：`Correct: 814/2203; accuracy: 0.369496; macro-F1: 0.313476`。

指令各部分的意思：

| 部分 | 意思 |
|---|---|
| `.venv\Scripts\python.exe` | 使用這個專案已裝好依賴的 Python |
| `verify_results.py` | 執行重新驗證結果的程式 |
| `--run` | 指定哪個實驗資料夾 |
| `runs/sst5-1.1b` | 1.1B 這次實驗的儲存位置 |

## 第二步：從一筆結果理解 accuracy

用 Excel 或文字編輯器打開：

`C:\nlp_project\runs\sst5-1.1b\test_predictions.csv`

每列是一句 test 影評：

| 欄位 | 意思 |
|---|---|
| text | 評論內容 |
| gold_id | 資料集提供的標準答案（0–4） |
| predicted_id | 模型預測（0–4） |
| gold_level | 標準答案的顯示等級（1–5） |
| predicted_level | 預測的顯示等級（1–5） |

兩個 level 相同就是判對，不同就是判錯。不是用模型自己生成的答案當標準答案。
1–5 的順序為：非常負面、負面、中性、正面、非常正面。
亦可以不載入模型，獨立在 PowerShell 計算 CSV 的 accuracy：

```powershell
$sstRows = Import-Csv .\runs\sst5-1.1b\test_predictions.csv
$sstCorrect = @($sstRows | Where-Object { $_.gold_id -eq $_.predicted_id }).Count
"Correct: $sstCorrect / $($sstRows.Count)"
"Accuracy: {0:P2}" -f ($sstCorrect / $sstRows.Count)
```

這只驗證 CSV 中的算術；第一步還會核對權重重新計算的預測與 CSV 是否一致。

## 第三步：理解模型架構與資料流

```text
Stanford SST-5 樹狀資料
    └─ 每棵樹取完整句子和根節點的情感標籤
        ├─ train：8527 句 → 用來更新分類頭
        ├─ validation：1095 句 → 選哪一輪的分類頭
        └─ test：2203 句 → 最後量度 accuracy / macro-F1

每句文字 → tokenizer → token IDs / attention mask
         → 凍結 Llama（不更新權重）
         → 最後一層、最後一個有效 token 的向量
         → LayerNorm → Linear(128) → ReLU → Dropout → Linear(5)
         → 5 個分數 → 最大分數的位置就是預測 ID
```

42M 每句輸出 512 維向量，1.1B 為 2048 維，所以兩者分類頭的輸入大小不同。
訓練時用 cross-entropy 將這 5 個分數與真實標籤比較，再用 AdamW 更新分類頭。
LLM 用 eval / no_grad 且所有參數 requires_grad=False；optimizer 只接收分類頭參數。

每個 epoch = 分類頭把 train 資料看完一次。每輪結束評估 validation，
共跑 30 輪，保留 validation macro-F1 最高的一輪（1.1B 為第 15 輪）。
test 不用於更新權重或選 epoch。

## 第四步：知道要看哪些檔案

| 檔案 | 功能 |
|---|---|
| sst5.py | 核心程式：資料讀取、LLM、MLP、訓練、輸入評論預測 |
| verify_results.py | 讀取已儲存分類頭與特徵，重新計算 test 預測及指標 |
| plot_results.py | 用逐輪紀錄產生圖表 |
| test_sst5.py | 9 個程式行為測試，例如 padding、凍結、特徵重用及正則化設定；不代表 accuracy 高 |
| runs/.../head.pt | 分類頭權重及設定，不包含完整 LLM |
| runs/.../features.pt | LLM 已抽取的向量及對應標籤 |
| runs/.../metrics.json | 每輪 loss / accuracy / F1 及最後結果 |
| runs/.../test_predictions.csv | 每筆 test 的標準答案與預測 |
| figures/ | 圖表 PNG |

LLM 預訓練權重由 Hugging Face 下載到使用者快取。
重算 test 時只需 head.pt + features.pt；從新文字推論時仍需 LLM 及 tokenizer。

若閱讀程式，建議按順序找：
`load_sst` / `parse_tree` → `load_encoder` → `FrozenEncoder` → `Head` → `train` → `scores`。

## 第五步：自己輸入影評（從文字重新跑 LLM）

```powershell
.venv\Scripts\python.exe sst5.py predict --checkpoint runs/sst5-1.1b/head.pt --text "The movie was wonderful!"
```

這一步會經過 tokenizer、LLM 和分類頭，輸出 label_id、level、sentiment 及機率。
單句若沒有標準答案，只能看預測，不能量度 accuracy。
資料是英文影評，本次沒有驗證中文或粵語。

## 第六步：自己從頭重新訓練

目前建議重用 1.1B 已儲存特徵，只訓練 MLP（預設 lr 0.0001、seed 7021、30 epochs）：

```powershell
.venv\Scripts\python.exe sst5.py train --features runs/sst5-1.1b/features.pt
```

結果自動儲存至新的 `runs/train-日期時間` 目錄，結尾會顯示對應的驗證指令。
若要重新執行 LLM 抽取特徵，使用下列不帶 `--features` 的指令。

建議先用較快的 42M，觀察每輪輸出：

```powershell
.venv\Scripts\python.exe sst5.py train --output runs/my-42m
.venv\Scripts\python.exe verify_results.py --run runs/my-42m
```

想重跑 1.1B：

```powershell
.venv\Scripts\python.exe sst5.py train --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --revision 59f6f375b26bde864a6ca194a9a3044570490064 --feature-batch-size 8 --output runs/my-1.1b
.venv\Scripts\python.exe verify_results.py --run runs/my-1.1b
```

過程依序顯示 Extracting train / validation / test（此時只抽特徵，不用 test 更新權重），
然後輸出 epoch 1–30 的訓練紀錄，最後顯示最佳 epoch 的 test 結果。
請使用新的 output 目錄；已有 head.pt 時會拒絕覆蓋。不同硬體／函式庫版本可能產生數值差異。

## 第七步：查看可視化

```powershell
.venv\Scripts\python.exe plot_results.py
Invoke-Item .\figures\training_1.1B.png
```

繪圖程式目前固定讀取 runs/sst5-42m-curves 及 runs/sst5-1.1b，
不會自動把 my-42m 等新實驗加入圖中；要比較自訂實驗，可修改 plot_results.py 的 runs 字典。
藍線為 train、橙線為 validation，虛線是按 validation macro-F1 選定的 epoch。
train 變好但 validation 不變好，表示過擬合。
