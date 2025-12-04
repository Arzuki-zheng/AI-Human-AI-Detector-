import streamlit as st
import pandas as pd
from transformers import pipeline

# 設定頁面配置
st.set_page_config(page_title="AI Text Detector", page_icon="🤖")

# 1. 載入模型 (使用 @st.cache_resource 避免每次重跑模型)
@st.cache_resource
def load_model():
    # 使用 OpenAI 官方的 RoBERTa 檢測器（更穩定）
    classifier = pipeline(
        "text-classification",
        model="openai-community/roberta-base-openai-detector",
        truncation=True,
        max_length=512
    )
    return classifier

# 2. UI 介面設計 (參考 justdone.com 風格)
st.title("🤖 AI Content Detector")
st.markdown("AIoT_HW5 Q1 AI/HUMAN 文本辨識器")
st.markdown("**學號**: 7114056186  **姓名**: 陳鉦元")
st.markdown("### Check if your text is written by **Human** or **AI**")
st.markdown("Paste your text below to analyze:")

# 文本輸入框
user_input = st.text_area("Input Text", height=200, placeholder="Type or paste content here...")

if st.button("Analyze Text"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                classifier = load_model()
                outputs = classifier(
                    user_input[:512],
                    truncation=True,
                    max_length=512,
                    return_all_scores=True
                )
            
                results = outputs[0]  # 這裡才取內層 list
            
                ai_score = 0.0
                human_score = 0.0
            
                for res in results:
                    label = res["label"].lower()
                    score = float(res["score"])
                    if "fake" in label or "ai" in label or "chatgpt" in label:
                        ai_score = score
                    else:
                        human_score = score
            
                # 計算百分比
                total = ai_score + human_score or 1.0
                ai_percent = (ai_score / total) * 100
                human_percent = (human_score / total) * 100

                # 5. 顯示結果
                st.markdown("---")
                st.subheader("Analysis Result")
                
                # 使用 Columns 顯示大數字
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("AI Generated Probability", f"{ai_percent:.1f}%")
                with col2:
                    st.metric("Human Written Probability", f"{human_percent:.1f}%")
                
                # 進度條視覺化
                st.write("### Confidence Distribution")
                st.progress(int(ai_percent)/100, text=f"AI Confidence: {ai_percent:.1f}%")
                
                # 簡單的長條圖（已修正）
                chart_data = pd.DataFrame({
                    "Category": ["AI", "Human"], 
                    "Probability": [ai_percent, human_percent]
                })
                st.bar_chart(chart_data, x="Category", y="Probability")
                
                # 判斷結論
                if ai_percent > 60:
                    st.error("🚨 This text is likely **AI-Generated**.")
                elif human_percent > 60:
                    st.success("✅ This text is likely **Human-Written**.")
                else:
                    st.info("🤔 The result is **Mixed/Uncertain**.")
            
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

# 頁尾
st.markdown("---")
st.caption("Powered by Hugging Face Transformers & Streamlit | Model: openai-community/roberta-base-openai-detector")
