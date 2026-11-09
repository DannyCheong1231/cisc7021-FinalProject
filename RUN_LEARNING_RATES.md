# 固定 learning rate 0.0001：自己如何運行

在 Windows 開啟 **PowerShell**，逐行執行：

```powershell
cd C:\nlp_project
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe sst5.py train --features runs/sst5-1.1b/features.pt
```

**統一程式碼位置：** `C:\nlp_project\sst5.py`。原獨立比較程式已移除。
此指令用 1.1B 已儲存特徵、現有 MLP、固定 seed 7021，
只訓練 learning rate **0.0001**，共 30 epochs，不重新下載或執行 LLM。
每次從頭初始化分類頭，不是接續上一個實驗的權重。

畫面會輸出一組逐輪紀錄，最後顯示 test 結果、儲存位置及驗證指令。
每次自動建立 `runs/train-日期時間/`，保留以前結果；亦可用 `--output` 指定新目錄。
預設值的修改位置：

```python
# sst5.py：兩種訓練方式共用的 learning rate 預設值
train_parser.add_argument('--lr', type=float, default=1e-4)
```

`1e-4` 與 `0.0001` 是同一個數值。命令列明確傳入的參數會覆蓋預設值。

新電腦先按 README 建立 Python 環境並安裝 requirements.txt；本機環境已具備依賴。

## 歷史 learning-rate 對照結果（保留作紀錄）

儲存位置：`C:\nlp_project\runs\lr-sweep-seed7021`。

| Learning rate | 選定 epoch | Validation accuracy | Validation macro-F1 |
|---|---:|---:|---:|
| 0.001 | 15 | 52.05% | 0.5060 |
| 0.0003 | 13 | 53.24% | 0.5129 |
| **0.0001** | **24** | **52.88%** | **0.5243** |

每個候選均用 validation macro-F1 選 epoch，再以該分數選 learning rate。
因此選定 **0.0001**；它不是表中 validation accuracy 最高的設定。
選定後才評估此候選的 test：

- 判對：**1209 / 2203**。
- Test accuracy：**54.88%**。
- Test macro-F1：**0.5401**。
- 相對原 LR 0.001 的 test accuracy 54.29%，提高約 **0.59 個百分點**。

未選中的 learning rates 不計算 test 指標，避免用 test 反覆選參數。
圖表全部為 validation 比較。降低 learning rate 明顯減緩 validation loss 後期上升，
但這次 test 改善幅度小，只是單一 seed 的初步結果。
LR 0.001 的 30 輪紀錄與原 1.1B 實驗完全一致，作為重現核對。

## 查看圖表及原始數字

```powershell
Invoke-Item .\runs\lr-sweep-seed7021\learning_rate_comparison.png
Import-Csv .\runs\lr-sweep-seed7021\comparison.csv | Format-Table
```

同時比較每個 learning rate 的 train loss 與 validation loss（只讀紀錄，不重新訓練）：

```powershell
.venv\Scripts\python.exe plot_results.py --sweep runs/lr-sweep-seed7021
Invoke-Item .\runs\lr-sweep-seed7021\train_validation_losses.png
```

三個子圖使用相同 y 軸刻度。train loss 是訓練過程 batch 平均（含 dropout），
validation loss 是輪末 eval 模式；灰色虛線表示按 validation macro-F1 選定的 epoch，
不是 validation loss 最低的位置。

圖中三條線是三個 learning rates；三個子圖依次是 validation loss、accuracy、macro-F1。
各候選目錄的 `training.png` 另有 train / validation 曲線。
`summary.json` 記錄所有候選分數及選定結果。

## 自己重新核對選定模型的 accuracy

```powershell
.venv\Scripts\python.exe verify_results.py --run runs/lr-sweep-seed7021/lr-0.0001
```

預期：`Correct: 1209/2203; accuracy: 0.548797; macro-F1: 0.540085`。
若重跑到新目錄，程式最後會印出對應的 verify 指令，直接複製即可。

## 用新模型輸入影評

```powershell
.venv\Scripts\python.exe sst5.py predict --checkpoint runs/lr-sweep-seed7021/lr-0.0001/head.pt --text "The movie was wonderful!"
```

`sst5.py` 的 `fit_head` 是兩種訓練方式共用的分類頭訓練程式。
先按 validation macro-F1 選 epoch，再評估選定 checkpoint 的 test 表現。
歷史 sweep 結果保留原樣；現在不再執行多 learning-rate 比較。

## 合併後的重現驗證

已執行以下指令，結果與原 LR 0.0001 實驗一致（epoch 24，accuracy 54.88%）：

```powershell
.venv\Scripts\python.exe sst5.py train --features runs/sst5-1.1b/features.pt --output runs/sst5-1.1b-merged
.venv\Scripts\python.exe verify_results.py --run runs/sst5-1.1b-merged
.venv\Scripts\python.exe plot_results.py --run runs/sst5-1.1b-merged
```

該結果目錄已存在，若再次訓練請換新的 `--output`，或省略此參數讓程式自動建立目錄。
繪圖指令只讀取紀錄，會產生該實驗的 `training.png`。
