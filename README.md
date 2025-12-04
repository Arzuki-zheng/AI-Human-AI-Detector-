# AI-Human-AI-Detector-
AIoT-HW5 辨識AI/Human文本 7114056186 陳鉦元
這是一個基於 Hugging Face Transformers 與 Streamlit 建立的 AI/Human 文本分類工具，用於辨識輸入的文本是由人工撰寫還是由大型語言模型（如 ChatGPT）生成。

Demo 連結
Streamlit App: 點此查看線上 Demo (請替換成你自己的 Streamlit Cloud 部署網址)

專案介紹
本專案實現了一個簡單的前端介面，使用者可以輸入一段英文文本，系統會透過預訓練的 RoBERTa 模型進行分析，並即時回饋該文本由 AI 生成的機率與由人類撰寫的機率。

主要功能
文本輸入框：提供一個讓使用者自由貼上文本的區域。

即時分析：點擊按鈕後，立即呼叫後端模型進行推論。

機率顯示：以百分比的量化指標顯示 AI 與 Human 的可能性。

視覺化圖表：透過進度條與長條圖，直觀地呈現信賴度分佈。

技術棧 (Tech Stack)
開發語言：Python 3.9+

後端模型：transformers (Hugging Face)

預訓練模型: Hello-SimpleAI/chatgpt-detector-roberta，一個在 AI/Human 文本上微調過的 RoBERTa 模型。

前端框架：streamlit

核心依賴：torch (PyTorch), scipy

本地運行 (Local Setup)
依照以下步驟在你的本地端電腦運行此專案。

複製專案庫 (Clone Repository)

bash
git clone [你的 GitHub Repository 網址]
cd [專案資料夾名稱]
安裝依賴套件 (Install Dependencies)
建議在虛擬環境中安裝，以避免套件版本衝突。

bash
pip install -r requirements.txt
啟動 Streamlit App

bash
streamlit run app.py
啟動後，瀏覽器會自動開啟 http://localhost:8501 顯示應用程式介面。
