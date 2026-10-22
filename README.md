# Student Signals — HW2 多元線性回歸網站

這是一個可直接部署至 GitHub Pages 的靜態網站，呈現 CRISP-DM 分析流程、三種線性回歸模型比較、RFE 特徵選擇與 95% 預測區間。

網站亦包含瀏覽器端互動分析工具：

- 下拉選單切換兩種 Problem（內建學生成績／自訂 CSV）
- 可輸入公開 CSV URL 或上傳本機 CSV
- 自動辨識數值欄位與選擇預測目標
- 執行單純或多元線性回歸
- 依 Pearson 相關係數進行特徵選擇
- 顯示 MAE、RMSE、R² 與實際值／預測值散佈圖

## 本機預覽

因瀏覽器不允許從 `file://` 讀取相對路徑資料，請在專案根目錄啟動靜態伺服器，例如：

```powershell
npx serve .
```

接著開啟終端機顯示的網址。

## GitHub Pages 部署

1. 將整個專案推送到 GitHub repository。
2. 到 repository 的 **Settings → Pages**。
3. Source 選擇 **Deploy from a branch**。
4. Branch 選擇 `main`，資料夾選擇 `/ (root)`，再按 Save。

數分鐘後即可由 GitHub Pages 網址瀏覽。

## 主要檔案

- `index.html`：網站內容與結構
- `styles.css`：響應式視覺設計
- `app.js`：CSV 讀取、特徵選擇、線性回歸與圖表
- `7115056039_hw2.ipynb`：完整分析 Notebook
- `7115056039_hw2.pdf`：作業報告
- `data/StudentPerformanceFactors.csv`：原始資料集

> 遠端 CSV 必須允許瀏覽器跨網域存取（CORS）。若來源阻擋，可先下載 CSV，再使用網站內的檔案上傳功能。
