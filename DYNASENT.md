# DynaSent：42M 凍結 LLM＋MLP

來源：[DynaSent 官方資料](https://github.com/cgpotts/dynasent)，固定版本 `707b6208978bfe7ff251a0c3263c9f0552c27b48` 的 v1.1 zip，下載時核對 SHA-256。

Round 1 為 Yelp 原句，Round 2 為人工撰寫的挑戰句。使用 `gold_label`，排除 mixed／無共識，未使用附帶模型預測或星級作標籤。完整使用可用的 93,553 筆 train、4,320 筆 validation、4,320 筆 test，保留官方切分，不重新隨機切分。SST-dev revalidation 檔不加入本次實驗。

模型為 `rzzhan/tiny-llama-stories-42m`，revision `eefdbd1577163cb01d906066bbe9176da71f86a1`。凍結 LLM，讀取最後一層最後一個有效 token 特徵；訓練 LayerNorm → Linear(128) → ReLU → Dropout(0.3) → Linear(3)。內部標籤：0=負面、1=中性、2=正面。

已完成的首次實驗設定：30 epochs、learning rate 0.0001、weight decay 0.1、seed 7021、head batch 64、max token length 128。目前程式預設及下列重跑指令已改為 200 epochs。合併兩輪 train 訓練，以合併 dev 的 macro-F1 保存最佳模型；只用選定模型報告合併及分輪 test。兩輪 test 分別有 3,600／720 筆，合併結果較受 Round 1 影響。

在 `C:\nlp_project` 的 PowerShell 中，重新執行一次完整實驗（自動建立新輸出目錄）：

```powershell
.venv\Scripts\python.exe sst5.py train --dataset dynasent --model rzzhan/tiny-llama-stories-42m --revision eefdbd1577163cb01d906066bbe9176da71f86a1 --epochs 200 --lr 0.0001 --hidden-size 128 --dropout 0.3 --weight-decay 0.1 --seed 7021 --feature-batch-size 128
```

查看本次已執行實驗的結果及圖表：

```powershell
.venv\Scripts\python.exe verify_results.py --run runs/dynasent-42m
.venv\Scripts\python.exe plot_results.py --run runs/dynasent-42m
```

重新訓練分類頭可重用新資料集的特徵：

```powershell
.venv\Scripts\python.exe sst5.py train --dataset dynasent --features runs/dynasent-42m/features.pt --epochs 200 --lr 0.0001
```

輸出位置為 `runs/dynasent-42m/`：`metrics.json` 包含訓練曲線、最佳 epoch、test 及 test_by_round；`head.pt` 是分類頭；`features.pt` 是特徵；`test_predictions.csv` 是逐句預測。`gold_level`／`predicted_level` 在三分類實驗只代表 label+1 的 1–3 編碼，並非 SST-5 五級強度。

這是不同資料及三分類任務，無法用其 accuracy 高於 SST-5 就判斷原資料有問題。仍以各自的 train/validation 曲線和驗證指標判斷泛化；本次只有一個 seed。

本次已完成：最佳 epoch 23，test accuracy 58.19%、macro-F1 0.5681。過擬合分析及官方切分的文字重合稽核見 [本次報告](runs/dynasent-42m/REPORT.md)。

200 epochs 重跑另存於 `runs/dynasent-42m-200epochs/`，前 30 輪紀錄與首次實驗一致。最佳 epoch 85，test accuracy 58.50%、macro-F1 0.5712；loss 圖及過擬合判斷見 [200 輪報告](runs/dynasent-42m-200epochs/REPORT.md)。查看此實驗時，將上面 verify／plot 指令的 `--run` 改為 `runs/dynasent-42m-200epochs`。

1.1B、30 epochs 實驗另存於 `runs/dynasent-1.1b-30epochs/`，最佳 epoch 25，test accuracy 76.60%、macro-F1 0.7651；比較圖、指令及限制見 [1.1B 報告](runs/dynasent-1.1b-30epochs/REPORT.md)。本次以命令列指定模型及輪數，主程式預設仍為42M／200 epochs。
