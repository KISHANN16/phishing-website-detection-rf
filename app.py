import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from url_feature_extractor import predict_url, extract_features_from_url

# Page Configuration
st.set_page_config(
    page_title="PhishingGuard AI | Enterprise Threat Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise SaaS Design System CSS
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    
    <style>
    /* Base Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Top Enterprise Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
    }
    
    .status-pill {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    
    /* SaaS KPI Card */
    .saas-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    
    .saas-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .saas-lbl {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-top: 2px;
    }
    
    /* Result Badges */
    .badge-safe {
        background-color: #065f46;
        color: #ecfdf5;
        border: 1px solid #059669;
        padding: 14px 24px;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        display: inline-block;
    }
    
    .badge-danger {
        background-color: #991b1b;
        color: #fef2f2;
        border: 1px solid #dc2626;
        padding: 14px 24px;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b1329;
        border-right: 1px solid #1e293b;
    }
    
    /* Enterprise Footer */
    .enterprise-footer {
        border-top: 1px solid #1e293b;
        padding: 16px 0px;
        margin-top: 40px;
        color: #64748b;
        font-size: 0.85rem;
        display: flex;
        justify-content: space-between;
    }
    </style>
""", unsafe_allow_html=True)

# Top Hero Render
st.markdown("""
    <div class="hero-banner">
        <div>
            <div class="hero-title">🛡️ PhishingGuard AI</div>
            <div class="hero-subtitle">Automated Phishing Threat Detection & Domain Telemetry Engine</div>
        </div>
        <div>
            <span class="status-pill">● MODEL ACTIVE | 98% ACCURACY</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Clean Sidebar - Architecture & Metrics
st.sidebar.markdown("### 📊 Model Architecture")
st.sidebar.markdown("""
<div style="background: #1e293b; padding: 14px; border-radius: 8px; border: 1px solid #334155; font-size: 0.85rem; color: #cbd5e1;">
    <p style="margin-bottom: 6px;"><b>Algorithm:</b> Random Forest</p>
    <p style="margin-bottom: 6px;"><b>Trees Count:</b> 100 Decision Trees</p>
    <p style="margin-bottom: 6px;"><b>Training Dataset:</b> 11,055 Samples</p>
    <p style="margin-bottom: 0px;"><b>Features Count:</b> 30 Security Metrics</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Model Metrics")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    st.markdown("""
        <div class="saas-card">
            <div class="saas-val">98.0%</div>
            <div class="saas-lbl">Accuracy</div>
        </div>
    """, unsafe_allow_html=True)
with col_s2:
    st.markdown("""
        <div class="saas-card">
            <div class="saas-val">0.98</div>
            <div class="saas-lbl">F1-Score</div>
        </div>
    """, unsafe_allow_html=True)

col_s3, col_s4 = st.sidebar.columns(2)
with col_s3:
    st.markdown("""<br><div class="saas-card"><div class="saas-val">0.98</div><div class="saas-lbl">Precision</div></div>""", unsafe_allow_html=True)
with col_s4:
    st.markdown("""<br><div class="saas-card"><div class="saas-val">0.98</div><div class="saas-lbl">Recall</div></div>""", unsafe_allow_html=True)

# Load Dataset
dataset_file = "kaggle_phishing_dataset_with_urls.csv"
urls_df = pd.read_csv(dataset_file) if os.path.exists(dataset_file) else None

# Main Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Live Threat Scanner", 
    "📂 Benchmark Dataset (11,055 Records)", 
    "📊 Performance & Analytics"
])

with tab1:
    st.markdown("### URL Security Scanner")
    st.write("Analyze any website URL to extract structural telemetry and classify threat probability.")
    
    url_input = st.text_input("Enter Target Domain / URL:", value="http://fake-lagin.com", placeholder="e.g. http://fake-lagin.com, https://google.com")
    
    if st.button("Scan Target URL", type="primary") or url_input:
        model_file = "output/random_forest_phishing_model.joblib"
        if not os.path.exists(model_file):
            st.error("Model file missing. Please train model first.")
        else:
            try:
                res = predict_url(url_input, model_path=model_file)
                st.markdown("---")
                
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    st.markdown("#### Threat Classification")
                    if res['prediction'] == "PHISHING":
                        st.markdown('<div class="badge-danger">⚠️ MALICIOUS PHISHING SITE</div>', unsafe_allow_html=True)
                        st.error(f"High Security Risk! Phishing Probability: **{res['phishing_probability']}%**")
                    else:
                        st.markdown('<div class="badge-safe">✅ LEGITIMATE (SAFE SITE)</div>', unsafe_allow_html=True)
                        st.success(f"Domain verified clean. Confidence: **{res['confidence']}%**")
                
                with res_col2:
                    st.markdown("#### Risk Assessment")
                    st.progress(res['phishing_probability'] / 100.0)
                    st.write(f"- **Phishing Risk Score:** `{res['phishing_probability']}%`")
                    st.write(f"- **Legitimate Score:** `{res['legitimate_probability']}%`")
                    st.write(f"- **Evaluated Model:** `Random Forest Classifier`")

                st.markdown("---")
                st.markdown("#### Extracted Security Feature Telemetry")
                
                feat_dict = res['features']
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                col_f1.metric("IP Address in URL", "Detected ⚠️" if feat_dict['Having_IP_Address'] else "None ✅")
                col_f2.metric("URL Length", f"{feat_dict['URL_Length']} chars")
                col_f3.metric("Number of Dots", f"{feat_dict['Number_of_Dots']} dots")
                col_f4.metric("Subdomains", f"{feat_dict['Number_of_Subdomains']} subs")
                
                col_f5, col_f6, col_f7, col_f8 = st.columns(4)
                col_f5.metric("Shortening Service", "Used ⚠️" if feat_dict['Short_URL'] else "None ✅")
                col_f6.metric("Prefix / Suffix (-)", "Present ⚠️" if feat_dict['Prefix_Suffix'] else "None ✅")
                col_f7.metric("HTTPS Protocol", "Valid ✅" if feat_dict['HTTPS_Types'] == 2 else "Suspicious ⚠️")
                col_f8.metric("Estimated Domain Age", f"{feat_dict['Domain_Age_Days']} days")

                with st.expander("Inspection: View All 30 Telemetry Attributes"):
                    st.dataframe(pd.DataFrame([feat_dict]).T, use_container_width=True)

            except Exception as e:
                st.error(f"Analysis failed: {e}")

with tab2:
    st.markdown("### Benchmark Dataset Explorer (11,055 Records)")
    st.write("Browse or search records from the benchmark Phishing Dataset to test the Random Forest model live.")
    
    if urls_df is not None:
        col_sch, col_flt = st.columns([3, 1])
        with col_sch:
            search_q = st.text_input("Search Domain / URL in Dataset:", placeholder="e.g. google, paypal, login, bit.ly")
        with col_flt:
            flt_type = st.selectbox("Category Filter:", ["All Records", "Phishing (1)", "Legitimate (0)"])
            
        filtered_df = urls_df.copy()
        if flt_type == "Phishing (1)":
            filtered_df = filtered_df[filtered_df['Target_Class'] == 1]
        elif flt_type == "Legitimate (0)":
            filtered_df = filtered_df[filtered_df['Target_Class'] == 0]
            
        if search_q:
            filtered_df = filtered_df[filtered_df['Website_URL'].str.contains(search_q, case=False, na=False)]

        st.markdown(f"**Showing {len(filtered_df)} matching records out of 11,055 total records:**")
        
        c_sel1, c_sel2 = st.columns([2, 3])
        with c_sel1:
            rec_id = st.number_input("Select Record ID:", min_value=1, max_value=len(urls_df), value=int(filtered_df.iloc[0]['Record_ID']) if len(filtered_df)>0 else 1)
            selected_row = urls_df[urls_df['Record_ID'] == rec_id].iloc[0]
            st.info(f"**Dataset True Label:** `{selected_row['Label']}`")
        with c_sel2:
            st.markdown("📋 **Copy Selected Target URL:**")
            st.code(selected_row['Website_URL'], language=None)

        if st.button("Evaluate Selected Record", type="primary"):
            res = predict_url(selected_row['Website_URL'], model_path="output/random_forest_phishing_model.joblib")
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Model Classification:**")
                if res['prediction'] == "LEGITIMATE":
                    st.markdown('<div class="badge-safe">✅ LEGITIMATE (0)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-danger">⚠️ PHISHING (1)</div>', unsafe_allow_html=True)
            with c2:
                st.write("**Model Confidence:**")
                st.metric("Confidence Score", f"{res['confidence']}%")
                st.metric("Actual Label", selected_row['Label'])

        st.dataframe(filtered_df, use_container_width=True, height=350)
    else:
        st.warning("Dataset file missing.")

with tab3:
    st.markdown("### Model Building & Comparative Performance")
    
    models_data = {
        "Algorithm": ["Logistic Regression", "Decision Tree", "Random Forest (Selected)", "Support Vector Machine", "Gradient Boosting"],
        "Accuracy": [0.98, 0.94, 0.98, 0.74, 0.98],
        "Precision": [0.98, 0.94, 0.98, 0.73, 0.98],
        "Recall": [0.98, 0.93, 0.98, 0.65, 0.98],
        "F1-Score": [0.98, 0.93, 0.98, 0.69, 0.98]
    }
    st.table(pd.DataFrame(models_data).style.highlight_max(axis=0, color='#1e3a8a', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']))
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Random Forest Confusion Matrix")
        if os.path.exists("output/confusion_matrix.png"):
            st.image("output/confusion_matrix.png", use_container_width=True)
            
    with col_b:
        st.markdown("#### Top 10 Feature Importance (Correlation)")
        if os.path.exists("output/top_10_features.png"):
            st.image("output/top_10_features.png", use_container_width=True)

# Professional Footer
st.markdown("""
    <div class="enterprise-footer">
        <div>© 2026 PhishingGuard AI Threat Engine. All rights reserved.</div>
        <div>Built with Python, Scikit-Learn (Random Forest) & Streamlit</div>
    </div>
""", unsafe_allow_html=True)
