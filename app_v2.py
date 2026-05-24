import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="InsureIQ — Health Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
ALPHA     = 0.45
INR_RATE  = 94
COST_FACTOR = 0.12
PREMIUM_RATE = 0.15

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp { background-color: #080C14; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0D1321 0%, #111827 100%);
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }
    .sidebar-section {
        margin: 1rem 0 0.4rem 0;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #475569 !important;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 0.3rem;
    }
    .result-card {
        background: #0F1929;
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 1.6rem 1.4rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .card-claim::before  { background: linear-gradient(90deg, #3B82F6, #06B6D4); }
    .card-low::before    { background: linear-gradient(90deg, #10B981, #34D399); }
    .card-medium::before { background: linear-gradient(90deg, #F59E0B, #FCD34D); }
    .card-high::before   { background: linear-gradient(90deg, #EF4444, #F97316); }
    .card-premium::before{ background: linear-gradient(90deg, #8B5CF6, #EC4899); }
    .card-monthly::before{ background: linear-gradient(90deg, #6366F1, #8B5CF6); }
    .card-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.5rem;
    }
    .card-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        font-weight: 400;
        line-height: 1.1;
        color: #F1F5F9;
    }
    .card-sub {
        font-size: 0.78rem;
        color: #475569;
        margin-top: 0.4rem;
    }
    .tier-pill {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    .tier-low    { background:#022C22; color:#34D399; border:1px solid #064E3B; }
    .tier-medium { background:#1C1100; color:#FCD34D; border:1px solid #451A03; }
    .tier-high   { background:#1F0505; color:#F87171; border:1px solid #450A0A; }
    .result-banner {
        background: linear-gradient(135deg, #0F1929 0%, #111827 100%);
        border: 1px solid #1E293B;
        border-radius: 20px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }
    .banner-icon  { font-size: 2.6rem; }
    .banner-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        color: #F1F5F9;
        margin: 0;
    }
    .banner-sub   { font-size: 0.85rem; color: #64748B; margin-top: 0.2rem; }
    .page-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #F1F5F9;
        margin-bottom: 0.2rem;
    }
    .page-subtitle {
        color: #475569;
        font-size: 0.9rem;
        margin-bottom: 1.8rem;
    }
    .info-box {
        background: #0D1321;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        color: #64748B;
        font-size: 0.85rem;
        line-height: 1.7;
    }
    .profile-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.6rem;
    }
    .profile-item {
        background: #0D1321;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
    }
    .profile-key   { font-size: 0.68rem; color:#475569; text-transform:uppercase; letter-spacing:0.08em; }
    .profile-val   { font-size: 0.95rem; color:#CBD5E1; font-weight:500; margin-top:0.15rem; }
    .streamlit-expanderHeader { color: #64748B !important; font-size: 0.85rem !important; }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3B82F6, #6366F1) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.03em !important;
        transition: opacity 0.2s !important;
        margin-top: 0.5rem;
    }
    .stButton > button:hover { opacity: 0.88 !important; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

BASE_FEATURES = [
    "age", "bmi",
    "systolic_bp", "diastolic_bp", "cholesterol_level",
    "diabetes", "heart_disease",
    "past_claims_log",
    "sex", "smoker",
    "exercise_level", "diet_quality",
    "alcohol_consumption", "stress_level",
]

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    return (
        joblib.load(f"{MODEL_DIR}/risk_model.pkl"),
        joblib.load(f"{MODEL_DIR}/claim_model.pkl"),
        joblib.load(f"{MODEL_DIR}/charges_model.pkl"),
    )

def compute_predictions(risk_m, claim_m, charges_m, raw_inputs: dict) -> dict:
    inp = dict(raw_inputs)
    inp["past_claims_log"] = float(np.log1p(inp.pop("past_claims")))

    X = pd.DataFrame([inp])[BASE_FEATURES]

    risk_score  = float(np.clip(risk_m.predict(X)[0],    0.0, 1.0))
    claim_prob  = float(np.clip(claim_m.predict(X)[0],   0.0, 1.0))
    base_usd    = float(charges_m.predict(X)[0])

    adj_usd = base_usd * (1 + ALPHA * risk_score)
    adj_inr = adj_usd * INR_RATE * COST_FACTOR
    yearly_premium  = adj_inr * PREMIUM_RATE
    monthly_premium = yearly_premium / 12

    tier = ("Low"    if risk_score < 0.33 else
            "Medium" if risk_score < 0.66 else "High")

    return {
        "risk_score"      : risk_score,
        "claim_prob"      : claim_prob,
        "base_usd"        : base_usd,
        "adj_usd"         : adj_usd,
        "adj_inr"         : adj_inr,
        "yearly_premium"  : yearly_premium,
        "monthly_premium" : monthly_premium,
        "loading_pct"     : ALPHA * risk_score * 100,
        "tier"            : tier,
    }

def claim_arc_chart(value: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.8, 2.2), facecolor="#0F1929")
    ax.set_facecolor("#0F1929")

    t = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(t), np.sin(t), color="#1E293B", lw=16, solid_capstyle="round")

    if value < 0.33:
        arc_col = "#34D399"
    elif value < 0.66:
        arc_col = "#FCD34D"
    else:
        arc_col = "#F87171"

    fill = np.linspace(np.pi, np.pi - value * np.pi, 300)
    ax.plot(np.cos(fill), np.sin(fill), color=arc_col, lw=16, solid_capstyle="round")

    ax.text(0, 0.08, f"{value:.0%}",
            ha="center", va="center", fontsize=26, fontweight="bold", color="#F1F5F9")
    ax.text(0, -0.28, "Claim Probability",
            ha="center", va="center", fontsize=9, color="#64748B")

    ax.text(-1.15, -0.05, "0%",  ha="center", fontsize=8, color="#334155")
    ax.text( 1.15, -0.05, "100%",ha="center", fontsize=8, color="#334155")

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.45, 1.15)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig

def card(css_class: str, label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="card-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="result-card {css_class}">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
        {sub_html}
    </div>"""

def profile_item(key: str, val: str) -> str:
    return f"""
    <div class="profile-item">
        <div class="profile-key">{key}</div>
        <div class="profile-val">{val}</div>
    </div>"""

st.markdown('<div class="page-title">🏥 InsureIQ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">AI-powered health insurance risk assessment '
    '& premium estimation</div>',
    unsafe_allow_html=True
)

try:
    risk_m, claim_m, charges_m = load_models()
    models_ready = True
except Exception as e:
    st.error(
        f"⚠️ **Models not found.** "
        f"Run `python insurance_pipeline_v2.py` first to train and save models.\n\n`{e}`"
    )
    models_ready = False

if models_ready:
    with st.sidebar:
        st.markdown("## Patient Details")

        st.markdown('<div class="sidebar-section">Demographics</div>',
                    unsafe_allow_html=True)
        age = st.slider("Age", min_value=18, max_value=64, value=35, step=1)
        sex = st.selectbox("Biological Sex", ["male", "female"])

        st.markdown('<div class="sidebar-section">Physical Health</div>',
                    unsafe_allow_html=True)
        bmi       = st.slider("BMI",            15.0, 55.0, 27.0, step=0.1,
                              help="Body Mass Index: weight(kg) / height²(m)")
        sys_bp    = st.slider("Systolic BP (mmHg)",   90, 180, 122,
                              help="Upper blood-pressure number")
        dias_bp   = st.slider("Diastolic BP (mmHg)", 60, 120,  80,
                              help="Lower blood-pressure number")
        chol      = st.slider("Cholesterol (mg/dL)", 120, 300, 190)
        diabetes      = st.checkbox("Diabetes",      value=False)
        heart_disease = st.checkbox("Heart Disease", value=False)

        st.markdown('<div class="sidebar-section">Lifestyle</div>',
                    unsafe_allow_html=True)
        smoker   = st.selectbox("Smoking Status",
                                ["no", "yes"],
                                format_func=lambda x: "Non-smoker" if x == "no" else "Smoker")
        exercise = st.selectbox("Exercise Level",   ["high", "medium", "low"])
        diet     = st.selectbox("Diet Quality",     ["good", "average", "poor"])
        alcohol  = st.selectbox("Alcohol Consumption", ["none", "moderate", "high"])
        stress   = st.selectbox("Stress Level",     ["low", "medium", "high"])

        st.markdown('<div class="sidebar-section">Claims History</div>',
                    unsafe_allow_html=True)
        past_claims = st.slider("Past Claims (count)", 0, 10, 1,
                                help="Number of claims filed previously")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍  Predict Risk & Premium", type="primary")

    if predict_btn:
        raw_inp = {
            "age"          : age,
            "sex"          : sex,
            "bmi"          : bmi,
            "smoker"       : smoker,
            "exercise_level"    : exercise,
            "diet_quality"      : diet,
            "alcohol_consumption": alcohol,
            "stress_level"      : stress,
            "systolic_bp"       : sys_bp,
            "diastolic_bp"      : dias_bp,
            "cholesterol_level" : chol,
            "diabetes"          : int(diabetes),
            "heart_disease"     : int(heart_disease),
            "past_claims"       : past_claims,
        }
        res = compute_predictions(risk_m, claim_m, charges_m, raw_inp)

        tier      = res["tier"]
        tier_col  = {"Low": "#34D399", "Medium": "#FCD34D", "High": "#F87171"}[tier]
        tier_emoji= {"Low": "🟢",      "Medium": "🟡",      "High": "🔴"}[tier]
        card_tier = {"Low": "card-low","Medium":"card-medium","High":"card-high"}[tier]

        st.markdown(f"""
        <div class="result-banner">
            <div class="banner-icon">{tier_emoji}</div>
            <div>
                <div class="banner-title">{tier} Risk Profile</div>
                <div class="banner-sub">
                    Based on {len(BASE_FEATURES)} clinical &amp; lifestyle features
                    &nbsp;·&nbsp; Risk score: {res['risk_score']:.3f}
                    &nbsp;·&nbsp; Loading: +{res['loading_pct']:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_arc, col_tier, col_yr, col_mo = st.columns([1.6, 1, 1.3, 1.3])

        with col_arc:
            st.pyplot(claim_arc_chart(res["claim_prob"]), use_container_width=True)

        with col_tier:
            st.markdown(
                card(card_tier, "Risk Tier",
                     tier,
                     f"Score: {res['risk_score']:.3f}"),
                unsafe_allow_html=True
            )

        with col_yr:
            st.markdown(
                card("card-premium", "Yearly Premium",
                     f"₹{res['yearly_premium']:,.0f}",
                     f"Risk-adjusted · INR"),
                unsafe_allow_html=True
            )

        with col_mo:
            st.markdown(
                card("card-monthly", "Monthly Premium",
                     f"₹{res['monthly_premium']:,.0f}",
                     f"= Yearly ÷ 12"),
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("📐  How is the premium calculated?"):
            st.markdown(f"""
**Step-by-step breakdown:**

| Step | Formula | Value |
|------|---------|-------|
| 1. Predict base cost | ML model (charges) | **${res['base_usd']:,.2f}** |
| 2. Risk loading | Base × (1 + {ALPHA} × {res['risk_score']:.3f}) | **${res['adj_usd']:,.2f}** |
| 3. Convert to INR | USD × {INR_RATE} × {COST_FACTOR} (India healthcare scaling) | **₹{res['adj_inr']:,.0f}** |
| 4. Annual premium | INR cost × {PREMIUM_RATE} (premium rate) | **₹{res['yearly_premium']:,.0f}** |
| 5. Monthly premium | Annual ÷ 12 | **₹{res['monthly_premium']:,.0f}** |

**α = {ALPHA}** risk-loading rationale:
- Risk < 0.33 (Low)    → up to **+{ALPHA*0.33*100:.1f}%** loading
- Risk 0.33–0.66 (Medium) → **+{ALPHA*0.33*100:.1f}% to +{ALPHA*0.66*100:.1f}%** loading
- Risk > 0.66 (High)   → **+{ALPHA*0.66*100:.1f}% to +{ALPHA*100:.1f}%** loading
            """)

        st.divider()

        st.markdown("#### 📋 Patient Profile")
        profile_items = [
            ("Age",           f"{age} years"),
            ("Sex",           sex.title()),
            ("BMI",           f"{bmi:.1f}"),
            ("Systolic BP",   f"{sys_bp} mmHg"),
            ("Diastolic BP",  f"{dias_bp} mmHg"),
            ("Cholesterol",   f"{chol} mg/dL"),
            ("Diabetes",      "Yes" if diabetes else "No"),
            ("Heart Disease", "Yes" if heart_disease else "No"),
            ("Smoking",       "Smoker" if smoker == "yes" else "Non-smoker"),
            ("Exercise",      exercise.title()),
            ("Diet",          diet.title()),
            ("Stress",        stress.title()),
            ("Alcohol",       alcohol.title()),
            ("Past Claims",   str(past_claims)),
            ("log(1+Claims)", f"{np.log1p(past_claims):.4f}"),
        ]
        grid_html = '<div class="profile-grid">'
        for k, v in profile_items:
            grid_html += profile_item(k, v)
        grid_html += "</div>"
        st.markdown(grid_html, unsafe_allow_html=True)

    else:
        c1, c2, c3 = st.columns(3)
        model_cards = [
            ("🎯", "Risk Score Model",
             "R² = 0.9380", "RMSE = 0.046", "RandomForest · 200 trees · max_depth 8"),
            ("📊", "Claim Probability",
             "ROC-AUC = 0.981", "Accuracy = 92.9%", "RandomForest · 300 trees · max_depth 10"),
            ("💵", "Charges Model",
             "R² = 0.8412", "RMSE = ₹4,965", "RandomForest · 400 trees · max_depth 12"),
        ]
        for col, (icon, title, m1, m2, note) in zip([c1, c2, c3], model_cards):
            with col:
                st.markdown(f"""
                <div class="result-card card-claim" style="text-align:left;padding:1.4rem;">
                    <div style="font-size:1.8rem;margin-bottom:0.5rem;">{icon}</div>
                    <div style="font-weight:600;color:#CBD5E1;font-size:0.95rem;
                                margin-bottom:0.6rem;">{title}</div>
                    <div style="color:#34D399;font-size:0.85rem;font-weight:600;">{m1}</div>
                    <div style="color:#34D399;font-size:0.85rem;font-weight:600;
                                margin-bottom:0.5rem;">{m2}</div>
                    <div style="color:#334155;font-size:0.75rem;">{note}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong style="color:#94A3B8;">How to use InsureIQ</strong><br><br>
            1. Fill in the patient's demographics and physical health details in the sidebar.<br>
            2. Set lifestyle factors (smoking, exercise, diet, alcohol, stress).<br>
            3. Enter the number of past insurance claims.<br>
            4. Click <strong>Predict Risk & Premium</strong> to get an instant assessment.<br><br>
            <em style="color:#334155;">Note: past_claims is log-transformed internally (log1p) before 
            being passed to the model — consistent with training pipeline v2.</em>
        </div>
        """, unsafe_allow_html=True)