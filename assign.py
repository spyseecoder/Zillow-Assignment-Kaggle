import warnings
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# optional boosters imported below inside try/except blocks

# assign.py
# GitHub Copilot
# Entry-point script to run a Zillow logerror modeling pipeline.
# Place this file at the project root. Expects a data/ folder with the CSV/XLSX files.

warnings.filterwarnings("ignore")

import statsmodels.api as sm

# Optional imports (try/except for environments where not installed)
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

# Optional plotting libs
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

# ---------------------
# Configuration
# ---------------------
ROOT = Path(".")
DATA_DIR = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS_DIR = ROOT / "models"

for d in (OUTPUTS, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------
# Utilities
# ---------------------
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    # compute RMSE in a way that's compatible with older/newer sklearn versions
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)}

# ---------------------
# Data Loading & Merge
# ---------------------
def load_data():
    # Look for property and train files in DATA_DIR first, then fall back to project ROOT.
    search_dirs = [DATA_DIR, ROOT]

    # Helper to try reading a file by suffix
    def _read_property_file(p: Path):
        if p.suffix.lower() in [".csv"]:
            return pd.read_csv(p, low_memory=False)
        if p.suffix.lower() in [".parquet"]:
            return pd.read_parquet(p)
        if p.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(p)
        # fallback
        return pd.read_csv(p, low_memory=False)

    props = []
    prop_patterns = ["properties_*.csv", "properties*.csv", "properties_*.parquet", "properties*.parquet", "properties_*.xlsx", "properties*.xlsx", "properties.csv"]
    for sd in search_dirs:
        if not sd.exists():
            continue
        for pat in prop_patterns:
            for p in sd.glob(pat):
                try:
                    props.append(_read_property_file(p))
                    print(f"Found property file: {p}")
                except Exception as e:
                    print(f"Failed to read property file {p}: {e}")
    if not props:
        present_data = []
        for sd in search_dirs:
            if sd.exists():
                present_data.extend([f.name for f in sd.iterdir()])
        raise FileNotFoundError(
            "No properties file found. Expected files like properties_2016.csv or properties_2017.csv in data/ or project root.\n"
            f"Files present: {present_data}"
        )

    properties = pd.concat(props, axis=0, ignore_index=True).drop_duplicates(subset="parcelid")
    print(f"Loaded properties: {properties.shape}")

    # Train files
    def _read_train_file(p: Path):
        if p.suffix.lower() in [".csv"]:
            # try to parse transactiondate if present
            try:
                return pd.read_csv(p, parse_dates=["transactiondate"])
            except Exception:
                return pd.read_csv(p)
        if p.suffix.lower() in [".gz"]:
            try:
                return pd.read_csv(p, compression="gzip", parse_dates=["transactiondate"])
            except Exception:
                return pd.read_csv(p, compression="gzip")
        # fallback
        try:
            return pd.read_csv(p, parse_dates=["transactiondate"])
        except Exception:
            return pd.read_csv(p)

    trains = []
    train_patterns = ["train_*.csv", "train*.csv", "train_*.csv.gz", "train*.csv.gz"]
    for sd in search_dirs:
        if not sd.exists():
            continue
        for pat in train_patterns:
            for t in sd.glob(pat):
                try:
                    trains.append(_read_train_file(t))
                    print(f"Found train file: {t}")
                except Exception as e:
                    print(f"Failed to read train file {t}: {e}")
    if not trains:
        present_data = []
        for sd in search_dirs:
            if sd.exists():
                present_data.extend([f.name for f in sd.iterdir()])
        raise FileNotFoundError(
            "No train file found. Expected files like train_2016_v2.csv or train_2017.csv in data/ or project root.\n"
            f"Files present: {present_data}"
        )

    train = pd.concat(trains, axis=0, ignore_index=True)
    print(f"Loaded train: {train.shape}")

    # Merge on parcelid
    df = train.merge(properties, how="left", on="parcelid")
    print(f"Merged dataset: {df.shape}")

    # Ensure target exists: prefer train.logerror, else compute if Zestimate & SalePrice present
    if "logerror" not in df.columns:
        if {"Zestimate", "SalePrice"}.issubset(df.columns):
            df["logerror"] = np.log(df["Zestimate"]) - np.log(df["SalePrice"])
        else:
            raise ValueError("No logerror and cannot compute it from Zestimate/SalePrice.")
    return df

# ---------------------
# Preprocessing & Feature Engineering
# ---------------------
def preprocess_and_engineer(df, max_cat_unique=200):
    df = df.copy()
    # Basic feature: transaction month/year
    if "transactiondate" in df.columns:
        df["transaction_month"] = df["transactiondate"].dt.month
        df["transaction_year"] = df["transactiondate"].dt.year
    # Built age
    if "yearbuilt" in df.columns:
        df["age"] = df["transaction_year"] - df["yearbuilt"]
        df["age"] = df["age"].clip(lower=0)
    # Total square feet proxy
    sqft_cols = ["calculatedfinishedsquarefeet", "finishedsquarefeet12", "finishedsquarefeet13"]
    for c in sqft_cols:
        if c not in df.columns:
            continue
    df["sqft"] = df[sqft_cols].bfill(axis=1).iloc[:, 0] if any(c in df.columns for c in sqft_cols) else np.nan

    # Drop columns with >90% missing or constant
    thresh = int(0.1 * len(df))
    good_cols = [c for c in df.columns if df[c].nunique() > 1 and df[c].count() > thresh]
    # Ensure we don't duplicate the target column if it's already in good_cols
    if "logerror" in df.columns:
        sel_cols = [c for c in good_cols if c != "logerror"] + ["logerror"]
    else:
        sel_cols = good_cols
    df = df[sel_cols]
    # Separate features
    target = "logerror"
    features = df.columns.drop([target] if target in df.columns else [])
    # Simple numeric/categorical split
    num_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]
    # Reduce high-cardinality categoricals by dropping if too many unique
    cat_cols = [c for c in cat_cols if df[c].nunique() <= max_cat_unique]
    # Impute numeric with median
    for c in num_cols:
        median = df[c].median()
        df[c] = df[c].fillna(median)
    # Impute categorical with 'missing' and label encode
    encoders = {}
    for c in cat_cols:
        df[c] = df[c].fillna("missing").astype(str)
        le = LabelEncoder()
        try:
            df[c] = le.fit_transform(df[c])
            encoders[c] = le
        except Exception:
            # fallback: factorize
            df[c], _ = pd.factorize(df[c])
    # Final feature list
    final_features = num_cols + cat_cols
    # Scaling for linear models (we will scale when needed)
    return df, final_features, encoders

# ---------------------
# Models: statsmodels OLS/RLM/GLM and sklearn + boosters
# ---------------------
def train_and_evaluate(df, features):
    X = df[features].copy()
    y = df["logerror"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    results = {}

    # Standardize for linear regressions
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # 1) OLS
    X_ols = sm.add_constant(pd.DataFrame(X_train_scaled, columns=features))
    model_ols = sm.OLS(y_train.values, X_ols).fit()
    pred_ols = model_ols.predict(sm.add_constant(pd.DataFrame(X_test_scaled, columns=features)))
    results["OLS"] = evaluate_model(y_test, pred_ols)
    joblib.dump(model_ols, MODELS_DIR / "ols_model.pkl")

    # 2) RLM (Robust)
    try:
        rlm = sm.RLM(y_train.values, X_ols, M=sm.robust.norms.HuberT()).fit()
        pred_rlm = rlm.predict(sm.add_constant(pd.DataFrame(X_test_scaled, columns=features)))
        results["RLM"] = evaluate_model(y_test, pred_rlm)
        joblib.dump(rlm, MODELS_DIR / "rlm_model.pkl")
    except Exception:
        results["RLM"] = None

    # 3) GLM (Gaussian family)
    try:
        glm = sm.GLM(y_train.values, X_ols, family=sm.families.Gaussian()).fit()
        pred_glm = glm.predict(sm.add_constant(pd.DataFrame(X_test_scaled, columns=features)))
        results["GLM"] = evaluate_model(y_test, pred_glm)
        joblib.dump(glm, MODELS_DIR / "glm_model.pkl")
    except Exception:
        results["GLM"] = None

    # 4) Ridge
    ridge = Ridge(random_state=RANDOM_STATE)
    ridge.fit(X_train_scaled, y_train)
    pred_ridge = ridge.predict(X_test_scaled)
    results["Ridge"] = evaluate_model(y_test, pred_ridge)
    joblib.dump(ridge, MODELS_DIR / "ridge_model.joblib")

    # 5) Lasso
    lasso = Lasso(random_state=RANDOM_STATE, max_iter=5000)
    lasso.fit(X_train_scaled, y_train)
    pred_lasso = lasso.predict(X_test_scaled)
    results["Lasso"] = evaluate_model(y_test, pred_lasso)
    joblib.dump(lasso, MODELS_DIR / "lasso_model.joblib")

    # 6) ElasticNet
    en = ElasticNet(random_state=RANDOM_STATE, max_iter=5000)
    en.fit(X_train_scaled, y_train)
    pred_en = en.predict(X_test_scaled)
    results["ElasticNet"] = evaluate_model(y_test, pred_en)
    joblib.dump(en, MODELS_DIR / "elasticnet_model.joblib")

    # 7) Random Forest (no scaling)
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results["RandomForest"] = evaluate_model(y_test, pred_rf)
    joblib.dump(rf, MODELS_DIR / "rf_model.joblib")

    # 8) XGBoost
    if xgb is not None:
        # ensure labels are 1-d numpy arrays (avoid passing DataFrame)
        try:
            y_train_arr = y_train.values.ravel()
        except Exception:
            y_train_arr = np.asarray(y_train).ravel()
        try:
            y_test_arr = y_test.values.ravel()
        except Exception:
            y_test_arr = np.asarray(y_test).ravel()
        # ensure numpy arrays with matching first-dimension lengths
        X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        X_test_np = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
        if X_train_np.shape[0] != y_train_arr.shape[0]:
            # try resetting indices to align
            try:
                X_train = X_train.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_train_arr = np.asarray(y_train).ravel()
                X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
            except Exception:
                raise ValueError(f"X_train rows ({X_train_np.shape[0]}) != y_train length ({y_train_arr.shape[0]}) and could not realign")
        dtrain = xgb.DMatrix(X_train_np, label=y_train_arr)
        dtest = xgb.DMatrix(X_test_np, label=y_test_arr)
        params = {"objective": "reg:squarederror", "seed": RANDOM_STATE, "verbosity": 0}
        xgb_model = xgb.train(params, dtrain, num_boost_round=300)
        pred_xgb = xgb_model.predict(dtest)
        results["XGBoost"] = evaluate_model(y_test, pred_xgb)
        xgb_model.save_model(str(MODELS_DIR / "xgb_model.json"))
    else:
        # xgboost not installed: use sklearn GradientBoostingRegressor as a fallback
        try:
            print("xgboost not available, training GradientBoostingRegressor fallback for XGBoost slot...")
            gb = GradientBoostingRegressor(n_estimators=300, random_state=RANDOM_STATE)
            gb.fit(X_train, y_train)
            pred_gb = gb.predict(X_test)
            results["XGBoost"] = evaluate_model(y_test, pred_gb)
            joblib.dump(gb, MODELS_DIR / "xgb_fallback_model.joblib")
        except Exception as e:
            print(f"Fallback XGBoost training failed: {e}")
            results["XGBoost"] = None

    # 9) LightGBM
    if lgb is not None:
        # LightGBM also expects 1-d label array
        try:
            y_train_arr = y_train.values.ravel()
        except Exception:
            y_train_arr = np.asarray(y_train).ravel()
        # Prepare numpy arrays and ensure lengths match
        X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        # Check alignment
        if X_train_np.shape[0] != y_train_arr.shape[0]:
            try:
                X_train = X_train.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_train_arr = np.asarray(y_train).ravel()
                X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
            except Exception:
                # give a clear error message
                raise ValueError(f"LightGBM label length ({y_train_arr.shape[0]}) does not match number of data rows ({X_train_np.shape[0]}).\n"
                                 f"Try checking for missing target values or misaligned indices.")
        ltrain = lgb.Dataset(X_train_np, label=y_train_arr)
        lparams = {"objective": "regression", "metric": "rmse", "seed": RANDOM_STATE}
        lgb_model = lgb.train(lparams, ltrain, num_boost_round=300)
        pred_lgb = lgb_model.predict(X_test)
        results["LightGBM"] = evaluate_model(y_test, pred_lgb)
        lgb_model.save_model(str(MODELS_DIR / "lgb_model.txt"))
    else:
        # lightgbm not installed: use sklearn GradientBoostingRegressor as a fallback
        try:
            print("lightgbm not available, training GradientBoostingRegressor fallback for LightGBM slot...")
            gb2 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=RANDOM_STATE)
            gb2.fit(X_train, y_train)
            pred_gb2 = gb2.predict(X_test)
            results["LightGBM"] = evaluate_model(y_test, pred_gb2)
            joblib.dump(gb2, MODELS_DIR / "lgb_fallback_model.joblib")
        except Exception as e:
            print(f"Fallback LightGBM training failed: {e}")
            results["LightGBM"] = None

    # Collect predictions for reporting
    preds = {
        "OLS": pred_ols,
        "RLM": pred_rlm if 'pred_rlm' in locals() else None,
        "GLM": pred_glm if 'pred_glm' in locals() else None,
        "Ridge": pred_ridge,
        "Lasso": pred_lasso,
        "ElasticNet": pred_en,
        "RandomForest": pred_rf,
        "XGBoost": pred_xgb if 'pred_xgb' in locals() else None,
        "LightGBM": pred_lgb if 'pred_lgb' in locals() else None,
    }

    # Save predictions for the best model (by MAE)
    valid_models = {k: v for k, v in results.items() if v is not None}
    best_model = min(valid_models.items(), key=lambda kv: kv[1]["mae"])[0]
    print(f"Best model by MAE: {best_model} -> {valid_models[best_model]}")
    # Save metrics
    (OUTPUTS / "metrics.json").write_text(json.dumps(results, indent=2))
    # Return also test split and predictions for plotting
    return results, best_model, X_test, y_test, preds, rf

# ---------------------
# Main execution
# ---------------------
def main():
    print("Loading data...")
    df = load_data()
    print("Preprocessing & feature engineering...")
    df, features, encoders = preprocess_and_engineer(df)
    print(f"Using {len(features)} features.")

    print("Training models...")
    results, best_model, X_test, y_test, preds, rf_model = train_and_evaluate(df, features)
    print("Results saved to outputs/ and models/")
    # Create plots (if matplotlib available)
    try:
        plot_reports(df, features, X_test, y_test, preds, rf_model, best_model)
    except Exception as e:
        print(f"Plotting failed: {e}")


def plot_reports(df, features, X_test, y_test, preds, rf_model, best_model):
    """Create and save diagnostic plots to OUTPUTS/plots.

    Plots saved:
      - logerror distribution (logerror_dist.png)
      - RandomForest feature importances (rf_feature_importance.png)
      - Predicted vs Actual for best model (pred_vs_actual_{model}.png)
      - Residuals histogram for best model (residuals_{model}.png)
    """
    plots_dir = OUTPUTS / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if plt is None:
        print("matplotlib not available, skipping plots.")
        return

    if sns is not None:
        sns.set(style="whitegrid")

    # 1) logerror distribution
    try:
        plt.figure(figsize=(8, 4))
        if "logerror" in df.columns:
            if sns is not None:
                sns.histplot(df["logerror"].dropna(), bins=60, kde=True)
            else:
                plt.hist(df["logerror"].dropna(), bins=60)
        plt.title("Logerror distribution")
        plt.xlabel("logerror")
        plt.tight_layout()
        plt.savefig(plots_dir / "logerror_dist.png")
        plt.close()
    except Exception as e:
        print(f"Failed to plot logerror distribution: {e}")

    # 2) RandomForest feature importances
    if rf_model is not None:
        try:
            importances = rf_model.feature_importances_
            fi = pd.Series(importances, index=features).sort_values(ascending=False).head(30)
            plt.figure(figsize=(8, min(12, 0.25 * len(fi))))
            if sns is not None:
                sns.barplot(x=fi.values, y=fi.index)
            else:
                plt.barh(fi.index, fi.values)
            plt.title("RandomForest feature importances (top 30)")
            plt.xlabel("importance")
            plt.tight_layout()
            plt.savefig(plots_dir / "rf_feature_importance.png")
            plt.close()
        except Exception as e:
            print(f"Failed to plot RF importances: {e}")

    # 3) Predicted vs Actual & residuals for best model
    try:
        if best_model in preds and preds[best_model] is not None:
            y_pred = preds[best_model]
            y_true = y_test.values if hasattr(y_test, "values") else np.array(y_test)

            # scatter actual vs predicted
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true, y_pred, alpha=0.3, s=8)
            vmin = min(y_true.min(), np.min(y_pred))
            vmax = max(y_true.max(), np.max(y_pred))
            plt.plot([vmin, vmax], [vmin, vmax], "r--")
            plt.xlabel("Actual logerror")
            plt.ylabel("Predicted logerror")
            plt.title(f"Actual vs Predicted ({best_model})")
            plt.tight_layout()
            plt.savefig(plots_dir / f"pred_vs_actual_{best_model}.png")
            plt.close()

            # residuals
            resid = y_true - np.array(y_pred)
            plt.figure(figsize=(8, 4))
            if sns is not None:
                sns.histplot(resid, bins=60, kde=True)
            else:
                plt.hist(resid, bins=60)
            plt.title(f"Residuals ({best_model})")
            plt.xlabel("residual")
            plt.tight_layout()
            plt.savefig(plots_dir / f"residuals_{best_model}.png")
            plt.close()
    except Exception as e:
        print(f"Failed to plot predictions/residuals: {e}")

    print(f"Saved plots to {plots_dir}")

    # 4) Region-based heatmap (mean logerror by region x month)
    try:
        # choose available region column
        region_candidates = ["regionidzip", "regionidcity", "regionidcounty"]
        region_col = None
        for c in region_candidates:
            if c in df.columns:
                region_col = c
                break

        if region_col is None:
            print("No region column found (regionidzip/regionidcity/regionidcounty). Skipping region heatmap.")
            return

        # require transaction_month; if not present try to extract
        if "transaction_month" not in df.columns and "transactiondate" in df.columns:
            df["transaction_month"] = pd.to_datetime(df["transactiondate"]).dt.month

        month_col = "transaction_month" if "transaction_month" in df.columns else None

        # limit to top N regions by count to keep heatmap readable
        top_n = 20
        region_counts = df[region_col].value_counts().nlargest(top_n).index.tolist()
        sub = df[df[region_col].isin(region_counts)].copy()

        if month_col is not None:
            pivot = sub.pivot_table(index=region_col, columns=month_col, values="logerror", aggfunc="mean")
            # sort rows by overall mean
            pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)
            plt.figure(figsize=(12, max(6, 0.3 * pivot.shape[0])))
            sns.heatmap(pivot, cmap="coolwarm", center=0, annot=False)
            plt.title(f"Mean logerror by {region_col} (top {top_n}) vs Month")
            plt.ylabel(region_col)
            plt.xlabel("Month")
            plt.tight_layout()
            fn = plots_dir / f"region_heatmap_{region_col}.png"
            plt.savefig(fn)
            plt.close()
            print(f"Saved region heatmap to {fn}")
        else:
            # single-column heatmap: region vs mean logerror
            agg = sub.groupby(region_col)["logerror"].mean().sort_values(ascending=False)
            plt.figure(figsize=(8, max(6, 0.25 * len(agg))))
            sns.heatmap(agg.to_frame(), cmap="coolwarm", center=0, annot=True)
            plt.title(f"Mean logerror by {region_col} (top {top_n})")
            plt.ylabel(region_col)
            plt.tight_layout()
            fn = plots_dir / f"region_heatmap_{region_col}.png"
            plt.savefig(fn)
            plt.close()
            print(f"Saved region heatmap to {fn}")
    except Exception as e:
        print(f"Failed to create region heatmap: {e}")

if __name__ == "__main__":
    main()