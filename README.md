# GULP: Gated Unimodal Loudness Pulse

自訂的 GULP activation function 與參數掃描工具。

## 安裝需求

需要 Python 3.9+ 以及套件：

```bash
pip install torch matplotlib
```

## 單一函數繪圖

```bash
python -m src.act.gulp
```

會在專案根目錄下產生 `gulp_activation.png`。

## 參數掃描 (批次繪圖)

使用 `gulp_param_sweep.py` 可一次產生多組參數對應的 activation 圖：

```bash
python -m src.act.gulp_param_sweep \
	--alpha 0.5,1.0,1.5 \
	--bump_amp 0.0,0.25,0.5 \
	--mu 0.0,1.0 \
	--sigma 0.5,1.0 \
	--out-dir docs/imgs --grid
```

參數說明：

* --alpha  逗號分隔的 alpha 值列表
* --bump_amp  bump 振幅列表
* --mu  高斯 bump 的中心列表
* --sigma  高斯標準差列表
* --x-range  x 範圍 (兩個數字)
* --points  取樣點數 (預設 500)
* --out-dir 圖片輸出目錄 (預設 docs/imgs)
* --prefix  圖片檔名前綴 (預設 gulp)
* --grid  額外輸出彙總網格圖
* --no-individual  若指定則不輸出單張圖片，只輸出網格

輸出檔名格式：

```text
gulp_a{alpha}_b{bump}_mu{mu}_s{sigma}.png
```

例如： `gulp_a1_b0p25_mu1_s0p5.png`。

## 未來可能改進

* 加入結果的統計或數值特徵 (例如曲率、最大值) 的 CSV 匯出。
* 加入互動式介面 (e.g. Gradio) 動態調參。
