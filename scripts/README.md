# PBS Scripts for GULP Experiments

這個目錄包含用於在 PBS 叢集上運行 GULP 實驗的腳本。

## 文件說明

- `run_cifar10_experiments.pbs`: 運行 CIFAR-10 實驗 (單個 seed)
- `run_cifar100_experiments.pbs`: 運行 CIFAR-100 實驗 (單個 seed)  
- `run_table3_complete.pbs`: 運行完整的 Table 3 實驗 (5 個 seeds)

## 使用方法

### 1. 準備環境

確保你的虛擬環境已經安裝了所需的依賴：

```bash
source /scratch/rp06/sl5952/PKB/.venv/bin/activate
pip install -r requirements.txt
```

### 2. 提交單個實驗

```bash
# CIFAR-10 實驗
qsub scripts/run_cifar10_experiments.pbs

# CIFAR-100 實驗  
qsub scripts/run_cifar100_experiments.pbs
```

### 3. 提交完整的 Table 3 實驗

```bash
qsub scripts/run_table3_complete.pbs
```

### 4. 檢查作業狀態

```bash
qstat -u $USER
```

### 5. 收集結果

實驗完成後，使用結果收集腳本：

```bash
python3 collect_results.py --exp_dir ./experiments --save
```

## 腳本配置

### 資源需求
- **GPU**: 1x A100 
- **CPU**: 16 cores
- **Memory**: 32GB
- **時間**: 
  - 單個實驗: 1-3 小時
  - 完整 Table 3: 12 小時

### 環境設定
- CUDA 12.6.2
- Python virtual environment: `/scratch/rp06/sl5952/PKB/.venv`
- 禁用 BFloat16: `TORCH_DISABLE_BFLOAT16=1`

## 實驗參數

### CIFAR-10 (ResNet-18)
- Dataset: CIFAR-10 (50k train, 10k test)
- Model: ResNet-18
- Epochs: 100
- Batch size: 128
- Learning rate: 0.1
- Optimizer: SGD (momentum=0.9, weight_decay=5e-4)
- Scheduler: Cosine annealing

### CIFAR-100 (Wide-ResNet-50-2)
- Dataset: CIFAR-100 (50k train, 10k test)
- Model: Wide-ResNet-50-2
- Epochs: 100
- Batch size: 128
- Learning rate: 0.1
- Optimizer: SGD (momentum=0.9, weight_decay=5e-4)
- Scheduler: Cosine annealing

### GULP 參數
- α (alpha): 1.2
- A (amplitude): 0.25
- μ (mu): 1.0
- σ (sigma): 0.5

## 輸出文件

### 訓練日誌
- `CIFAR10_[ACTIVATION]_seed[SEED].log`
- `CIFAR100_[ACTIVATION]_seed[SEED].log`

### 實驗結果
```
experiments/
├── cifar10_resnet18_relu_20240816_142030_seed42/
│   ├── config.json
│   ├── results.json
│   ├── best_model.pth
│   └── final_model.pth
├── table3_results.csv        # 原始結果
├── table3_summary.csv        # 統計摘要
└── table3_formatted.json     # Table 3 格式
```

## 故障排除

### 常見問題

1. **BFloat16 錯誤**: 已在腳本中設定 `TORCH_DISABLE_BFLOAT16=1`
2. **CUDA 記憶體不足**: 可以降低 batch size
3. **模組載入失敗**: 檢查 CUDA 版本和虛擬環境

### 監控作業

```bash
# 查看作業輸出
qcat [JOB_ID]

# 查看錯誤輸出  
qcat -e [JOB_ID]

# 取消作業
qdel [JOB_ID]
```

## 預期結果

完整的 Table 3 實驗將產生：

```
TABLE 3: CIFAR-10/100 accuracy ± std (n=5 seeds)

CIFAR10:
| Model | ReLU | GELU | SiLU | GULP |
|-------|------|------|------|------|
| resnet18 | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX |

CIFAR100:
| Model | ReLU | GELU | SiLU | GULP |
|-------|------|------|------|------|
| wide_resnet50_2 | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX |
```

預計 GULP 在兩個數據集上都會略優於其他激活函數。
