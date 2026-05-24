import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import (mean_squared_error, r2_score,
                             roc_auc_score, classification_report,
                             mean_absolute_error)

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib, os, json

DATA_PATH  = "insurance_extended_v2.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

ALPHA=0.45
SEED=42

CATEGORICAL = ["sex", "smoker",
               "exercise_level", "diet_quality",
               "alcohol_consumption", "stress_level"]

NUMERICAL   = ["age", "bmi",
               "systolic_bp", "diastolic_bp", "cholesterol_level",
               "diabetes", "heart_disease",
               "past_claims_log"]


BASE_FEATURES = NUMERICAL + CATEGORICAL


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "age_grp" in df.columns:
        df = df.drop(columns=["age_grp"])

    df["past_claims_log"] = np.log1p(df["past_claims"])
    return df


def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2):
    X = df[BASE_FEATURES]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=SEED)


def train_risk_model(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  MODEL 1 — Risk Score Regression")
    print("="*60)
    X_tr, X_te, y_tr, y_te = split_data(df, "risk_score")
    prep = build_preprocessor(CATEGORICAL, NUMERICAL)
    model = Pipeline([
        ("prep", prep),
        ("rf",   RandomForestRegressor(
                     n_estimators=200,
                     max_depth=8,
                     min_samples_leaf=8,
                     max_features="sqrt",
                     random_state=SEED, n_jobs=-1))
    ])


    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")


    joblib.dump(model, f"{MODEL_DIR}/risk_model.pkl")
    return model, X_tr, X_te, y_tr, y_te, y_pred, {"rmse":rmse,"mae":mae,"r2":r2}


def train_claim_model(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  MODEL 2 — Claim Probability (Regression + AUC Eval)")
    print("="*60)
    X_tr, X_te, y_tr, y_te = split_data(df, "claim_probability")
    prep = build_preprocessor(CATEGORICAL, NUMERICAL)


    model = Pipeline([
        ("prep", prep),
        ("rf",   RandomForestRegressor(
                     n_estimators=300,
                     max_depth=10,
                     min_samples_leaf=5,
                     max_features="sqrt",
                     random_state=SEED, n_jobs=-1))
    ])


    model.fit(X_tr, y_tr)
    y_pred_prob = model.predict(X_te)
    y_pred_prob = np.clip(y_pred_prob, 0, 1)
    threshold   = float(np.median(y_tr))
    y_bin_true  = (y_te  > threshold).astype(int)
    y_bin_pred  = (y_pred_prob > threshold).astype(int)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred_prob))
    mae  = mean_absolute_error(y_te, y_pred_prob)
    r2   = r2_score(y_te, y_pred_prob)
    auc  = roc_auc_score(y_bin_true, y_pred_prob)


    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  ROC-AUC (binarised @ {threshold:.3f}) : {auc:.4f}")
    print("\n  Classification Report (threshold={:.3f}):".format(threshold))
    print(classification_report(y_bin_true, y_bin_pred,
                                target_names=["Low","High"], digits=3))
    joblib.dump(model, f"{MODEL_DIR}/claim_model.pkl")
    return model, X_te, y_te, y_pred_prob, {"rmse":rmse,"mae":mae,"r2":r2,"auc":auc}


def train_charges_model(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  MODEL 3 — Insurance Charges Regression")
    print("="*60)
    X_tr, X_te, y_tr, y_te = split_data(df, "charges")
    prep = build_preprocessor(CATEGORICAL, NUMERICAL)
    model = Pipeline([
        ("prep", prep),
        ("rf",   RandomForestRegressor(
                     n_estimators=400,
                     max_depth=12,
                     min_samples_leaf=3,
                     max_features="sqrt",
                     random_state=SEED, n_jobs=-1))
    ])


    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  R²   : {r2:.4f}")
    joblib.dump(model, f"{MODEL_DIR}/charges_model.pkl")
    return model, X_tr, X_te, y_tr, y_te, y_pred, {"rmse":rmse,"mae":mae,"r2":r2}


def optimise_premium(predicted_charges: float,
                     predicted_risk: float,
                     alpha: float = ALPHA) -> dict:
    loading    = alpha * predicted_risk
    adj_prem   = predicted_charges * (1 + loading)
    profit_est = adj_prem - predicted_charges
    tier = ("Low" if predicted_risk < 0.33 else
            "Medium" if predicted_risk < 0.66 else "High")
    return {
        "base_premium"        : round(predicted_charges, 2),
        "risk_loading_pct"    : round(loading * 100, 2),
        "adjusted_premium"    : round(adj_prem, 2),
        "estimated_profit"    : round(profit_est, 2),
        "risk_tier"           : tier,
    }


def predict_new_customer(risk_model, claim_model, charges_model,
                         customer: dict) -> dict:
    inp = dict(customer)
    inp["past_claims_log"] = np.log1p(inp.get("past_claims", 0))
    X = pd.DataFrame([inp])[BASE_FEATURES]
    risk   = float(np.clip(risk_model.predict(X)[0],   0, 1))
    claim  = float(np.clip(claim_model.predict(X)[0],  0, 1))
    charge = float(charges_model.predict(X)[0])
    prem   = optimise_premium(charge, risk)
    return {
        "predicted_risk_score"       : round(risk,  4),
        "predicted_claim_probability": round(claim, 4),
        "predicted_base_charges"     : round(charge, 2),
        **prem,
    }


def compute_importance(pipeline, X_test, y_test, n_repeats=10, label=""):
    prep      = pipeline.named_steps["prep"]
    rf        = pipeline.named_steps["rf"]
    Xt        = prep.transform(X_test)
    num_names = NUMERICAL
    cat_names = list(prep.named_transformers_["cat"]
                     .get_feature_names_out(CATEGORICAL))
    all_names = num_names + cat_names
    result = permutation_importance(rf, Xt, y_test,
                                    n_repeats=n_repeats,
                                    random_state=SEED, n_jobs=-1)
    imp_df = (pd.DataFrame({"feature": all_names,
                             "importance": result.importances_mean,
                             "std": result.importances_std})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True))
    print(f"\n  Top-10 Permutation Importances — {label}")
    print(imp_df.head(10).to_string(index=False))
    return imp_df


def build_dashboard(df, risk_res, charges_res, imp_risk, imp_charges):
    BG, PANEL, TEXT, GRID = '#0F1117','#1A1D27','#E0E0E0','#2A2D3A'
    AC = ['#4FC3F7','#81C784','#FFB74D','#F06292','#CE93D8','#80DEEA']


    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    fig.suptitle("Health Insurance ML Pipeline — Model Dashboard",
                 fontsize=22, fontweight="bold", color="white", y=0.99)
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.52, wspace=0.42)


    def sax(ax, title):
        ax.set_facecolor(PANEL); ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=7)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.grid(color=GRID, lw=0.5, alpha=0.6)


    metrics = [
        ("Risk  R²",           f"{risk_res['metrics']['r2']:.4f}",    AC[0]),
        ("Risk  RMSE",         f"{risk_res['metrics']['rmse']:.4f}",  AC[1]),
        ("Charges  R²",        f"{charges_res['metrics']['r2']:.4f}", AC[2]),
        ("Charges  RMSE",      f"${charges_res['metrics']['rmse']:,.0f}", AC[3]),
    ]


    for i,(lbl,val,col) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(col+"22"); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.axis("off")
        for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(2)
        ax.text(0.5,0.62,val,  ha="center",va="center",fontsize=24,
                fontweight="bold",color=col,transform=ax.transAxes)
        ax.text(0.5,0.25,lbl,  ha="center",va="center",fontsize=10,
                color=TEXT,transform=ax.transAxes)


    ax = fig.add_subplot(gs[1, 0:2])
    y_te_r, y_pr_r = risk_res["y_test"], risk_res["y_pred"]
    ax.scatter(y_te_r, y_pr_r, alpha=0.35, s=12, color=AC[0])
    lims = [min(y_te_r.min(), y_pr_r.min()), max(y_te_r.max(), y_pr_r.max())]
    ax.plot(lims, lims, "w--", lw=1.2, label="Perfect fit")
    sax(ax, "Risk Score — Actual vs Predicted"); ax.set_xlabel("Actual", color=TEXT, fontsize=8)
    ax.set_ylabel("Predicted", color=TEXT, fontsize=8); ax.legend(fontsize=7,facecolor=PANEL,labelcolor=TEXT)

    ax = fig.add_subplot(gs[1, 2:4])
    y_te_c, y_pr_c = charges_res["y_test"], charges_res["y_pred"]
    ax.scatter(y_te_c, y_pr_c, alpha=0.35, s=12, color=AC[2])
    lims2 = [min(y_te_c.min(), y_pr_c.min()), max(y_te_c.max(), y_pr_c.max())]
    ax.plot(lims2, lims2, "w--", lw=1.2, label="Perfect fit")
    sax(ax, "Charges — Actual vs Predicted"); ax.set_xlabel("Actual ($)", color=TEXT, fontsize=8)
    ax.set_ylabel("Predicted ($)", color=TEXT, fontsize=8); ax.legend(fontsize=7,facecolor=PANEL,labelcolor=TEXT)

    ax = fig.add_subplot(gs[2, 0:2])
    top = imp_risk.head(12)
    bars = ax.barh(top["feature"][::-1], top["importance"][::-1], color=AC[0], edgecolor=PANEL)
    ax.errorbar(top["importance"][::-1], top["feature"][::-1],
                xerr=top["std"][::-1], fmt="none", color=TEXT, capsize=3, lw=1)
    sax(ax, "Feature Importance — Risk Score (Permutation)")
    ax.set_xlabel("Mean Δ accuracy", color=TEXT, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)

    ax = fig.add_subplot(gs[2, 2:4])
    top2 = imp_charges.head(12)
    ax.barh(top2["feature"][::-1], top2["importance"][::-1], color=AC[2], edgecolor=PANEL)
    ax.errorbar(top2["importance"][::-1], top2["feature"][::-1],
                xerr=top2["std"][::-1], fmt="none", color=TEXT, capsize=3, lw=1)
    sax(ax, "Feature Importance — Charges (Permutation)")
    ax.set_xlabel("Mean Δ accuracy", color=TEXT, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)


    ax = fig.add_subplot(gs[3, 0:2])
    risks    = np.linspace(0, 1, 300)
    base_prem = 8000


    for a, col, lbl in [(0.30, AC[1],"α=0.30"),(0.45, AC[0],"α=0.45 (chosen)"),(0.60, AC[3],"α=0.60")]:
        ax.plot(risks, base_prem*(1+a*risks), color=col, lw=2, label=lbl)
    ax.axvline(0.33, color="white", ls=":", lw=1, alpha=0.5)
    ax.axvline(0.66, color="white", ls=":", lw=1, alpha=0.5)
    ax.text(0.16, 11500, "Low\nRisk", ha="center", color=TEXT, fontsize=8)
    ax.text(0.50, 11500, "Medium\nRisk", ha="center", color=TEXT, fontsize=8)
    ax.text(0.83, 11500, "High\nRisk", ha="center", color=TEXT, fontsize=8)
    sax(ax, "Premium Loading Curve by Risk Score (base=$8,000)")
    ax.set_xlabel("Risk Score", color=TEXT, fontsize=8)
    ax.set_ylabel("Adjusted Premium ($)", color=TEXT, fontsize=8)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)


    ax = fig.add_subplot(gs[3, 2:4])
    residuals = np.array(y_te_c) - np.array(y_pr_c)
    ax.hist(residuals, bins=40, color=AC[4], edgecolor=PANEL, linewidth=0.3)
    ax.axvline(0, color="white", lw=1.5, ls="--")
    ax.axvline(residuals.mean(), color=AC[3], lw=1.5, ls="--",
               label=f"Mean={residuals.mean():,.0f}")
    sax(ax, "Residuals Distribution — Charges Model")
    ax.set_xlabel("Residual ($)", color=TEXT, fontsize=8)
    ax.set_ylabel("Count", color=TEXT, fontsize=8)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)


    plt.savefig("ml_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
    print("\n  Dashboard saved → ml_dashboard.png")


if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  HEALTH INSURANCE ML PIPELINE")
    print("█"*60)


    df = load_data(DATA_PATH)
    print(f"\n  Dataset: {df.shape[0]} rows × {df.shape[1]} cols")


    risk_model,   X_tr_r, X_te_r, y_tr_r, y_te_r, y_pr_r, m1 = train_risk_model(df)
    claim_model,  X_te_cl, y_te_cl, y_pr_cl, m2                = train_claim_model(df)
    charges_model,X_tr_c, X_te_c, y_tr_c, y_te_c, y_pr_c, m3  = train_charges_model(df)


    print("\n" + "="*60)
    print("  FEATURE IMPORTANCES (Permutation)")
    print("="*60)
    imp_risk    = compute_importance(risk_model,    X_te_r, y_te_r, label="Risk Score")
    imp_charges = compute_importance(charges_model, X_te_c, y_te_c, label="Charges")

    risk_res    = {"metrics": m1, "y_test": y_te_r, "y_pred": y_pr_r}
    charges_res = {"metrics": m3, "y_test": y_te_c, "y_pred": y_pr_c}

    build_dashboard(df, risk_res, charges_res, imp_risk, imp_charges)


    new_customer = {
        "age": 45, "sex": "male", "bmi": 31.5,
        "smoker": "yes",
        "exercise_level": "low", "diet_quality": "poor",
        "alcohol_consumption": "moderate",
        "systolic_bp": 148, "diastolic_bp": 92,
        "cholesterol_level": 240,
        "diabetes": 1, "heart_disease": 0,
        "stress_level": "high",
        "past_claims": 3,
    }

    print("\n" + "="*60)
    print("  DEMO PREDICTION — New Customer")
    print("="*60)
    result = predict_new_customer(risk_model, claim_model, charges_model, new_customer)
    for k, v in result.items():
        label = k.replace("_", " ").title()
        val   = f"${v:,.2f}" if "charge" in k or "premium" in k or "profit" in k else v
        print(f"  {label:35s}: {val}")


    summary = {"risk": m1, "claim": m2, "charges": m3, "alpha": ALPHA}
    with open(f"{MODEL_DIR}/metrics_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Models saved → {MODEL_DIR}/")
    print("  Pipeline complete ✓")