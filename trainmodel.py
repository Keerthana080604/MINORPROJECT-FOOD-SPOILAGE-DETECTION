import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
import joblib
import os

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("   FreshSense — Enhanced Model Training Pipeline")
print("=" * 60)

DATASET = "final_food_monitoring_with_thresholds (1).csv"
df = pd.read_csv(DATASET)
print(f"\n[✓] Dataset loaded → {df.shape[0]} records, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. NORMALISE COLUMN NAMES & LABEL
# ─────────────────────────────────────────────
df.rename(columns={
    "name":                  "Food_Item",
    "type":                  "Category",
    "temp_c":                "Temperature_C",
    "humidity_pct":          "Humidity_%",
    "spoilage_status":       "Condition_Class_Raw",
    "CO2_ppm":               "CO2_ppm",
    "Ethylene_ppm":          "Ethylene_ppm",
    "NH3_ppm":               "NH3_ppm",
    "H2S_ppm":               "H2S_ppm",
    "VOC_ppm":               "VOC_ppm",
    "O2_%":                  "O2_%",
    "Transit_Duration_hrs":  "Transit_Hours",
    "min_optimal_temp":      "Opt_Temp_Min_C",
    "max_optimal_temp":      "Opt_Temp_Max_C",
    "min_optimal_humidity":  "Opt_Humidity_Min_%",
    "max_optimal_humidity":  "Opt_Humidity_Max_%",
    "shelf_life":            "Shelf_Life_Days",
}, inplace=True)

df["Initial_Shelf_Life_hrs"] = df["Shelf_Life_Days"] * 24

# Normalise spoilage labels
label_map = {
    "OK":      "Safe",
    "Fresh":   "Safe",
    "WARN":    "Warning",
    "Warning": "Warning",
    "ALERT":   "Spoiled",
    "Spoiled": "Spoiled",
}
df["Condition_Class"] = df["Condition_Class_Raw"].map(label_map)
df.dropna(subset=["Condition_Class"], inplace=True)

print(f"[✓] Label distribution:")
print(df["Condition_Class"].value_counts().to_string())

# ─────────────────────────────────────────────
# 3. ENHANCED FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[*] Engineering advanced features...")

# Temperature & Humidity deviations
df["temp_deviation"] = df.apply(
    lambda r: max(0, r["Temperature_C"] - r["Opt_Temp_Max_C"])
            + max(0, r["Opt_Temp_Min_C"] - r["Temperature_C"]),
    axis=1,
)

df["humidity_deviation"] = df.apply(
    lambda r: max(0, r["Humidity_%"] - r["Opt_Humidity_Max_%"])
            + max(0, r["Opt_Humidity_Min_%"] - r["Humidity_%"]),
    axis=1,
)

# Normalized deviations (scaled by range)
df["temp_range"] = df["Opt_Temp_Max_C"] - df["Opt_Temp_Min_C"] + 1e-6
df["humidity_range"] = df["Opt_Humidity_Max_%"] - df["Opt_Humidity_Min_%"] + 1e-6
df["temp_deviation_norm"] = df["temp_deviation"] / df["temp_range"]
df["humidity_deviation_norm"] = df["humidity_deviation"] / df["humidity_range"]

# Temperature factor with improved formula
df["temp_factor"] = 1 + 0.08 * (df["Temperature_C"] - df["Opt_Temp_Max_C"]).clip(lower=0) \
                      + 0.04 * (df["Opt_Temp_Min_C"] - df["Temperature_C"]).clip(lower=0)

# Humidity stress factor
df["humidity_factor"] = 1 + 0.03 * df["humidity_deviation"]

# Combined environmental stress
df["env_stress"] = df["temp_factor"] * df["humidity_factor"]

# Shelf life calculations
df["hours_consumed"] = df["Transit_Hours"] * df["env_stress"]
df["Remaining_Shelf_Life_hrs"] = (
    df["Initial_Shelf_Life_hrs"] - df["hours_consumed"]
).clip(lower=0)

df["shelf_life_pct"] = df["Remaining_Shelf_Life_hrs"] / (df["Initial_Shelf_Life_hrs"] + 1e-6)
df["transit_ratio"] = df["Transit_Hours"] / (df["Initial_Shelf_Life_hrs"] + 1e-6)

# Gas-based features
# CO2 levels
df["CO2_level"] = pd.cut(
    df["CO2_ppm"],
    bins=[0, 100, 250, 500, float("inf")],
    labels=[0, 1, 2, 3],
).astype(float)

# Ethylene levels (critical for fruit ripening)
df["Ethylene_level"] = pd.cut(
    df["Ethylene_ppm"],
    bins=[0, 0.1, 1.0, 10, float("inf")],
    labels=[0, 1, 2, 3],
).astype(float)

# Log transforms for skewed distributions
for col in ["CO2_ppm", "NH3_ppm", "H2S_ppm", "Ethylene_ppm", "VOC_ppm"]:
    df[f"log_{col}"] = np.log1p(df[col])

# Square root transforms (alternative for moderate skew)
for col in ["CO2_ppm", "VOC_ppm"]:
    df[f"sqrt_{col}"] = np.sqrt(df[col])

# Gas ratios (interaction features)
df["NH3_to_H2S_ratio"] = df["NH3_ppm"] / (df["H2S_ppm"] + 1e-6)
df["VOC_to_CO2_ratio"] = df["VOC_ppm"] / (df["CO2_ppm"] + 1e-6)
df["Ethylene_to_O2_ratio"] = df["Ethylene_ppm"] / (df["O2_%"] + 1e-6)

# Composite gas index (weighted sum)
df["gas_spoilage_index"] = (
    0.3 * df["log_NH3_ppm"] +
    0.3 * df["log_H2S_ppm"] +
    0.2 * df["log_Ethylene_ppm"] +
    0.2 * df["log_VOC_ppm"]
)

# Oxygen depletion indicator
df["O2_depletion"] = 21 - df["O2_%"]  # Normal air is ~21% O2

# Interaction features
df["temp_x_humidity"] = df["Temperature_C"] * df["Humidity_%"]
df["temp_x_transit"] = df["Temperature_C"] * df["Transit_Hours"]
df["humidity_x_transit"] = df["Humidity_%"] * df["Transit_Hours"]
df["stress_x_transit"] = df["env_stress"] * df["Transit_Hours"]

# Polynomial features for critical variables
df["temp_squared"] = df["Temperature_C"] ** 2
df["humidity_squared"] = df["Humidity_%"] ** 2
df["transit_squared"] = df["Transit_Hours"] ** 2

# Risk score
df["spoilage_risk_score"] = (
    0.4 * df["temp_deviation_norm"] +
    0.2 * df["humidity_deviation_norm"] +
    0.2 * df["transit_ratio"] +
    0.2 * (1 - df["shelf_life_pct"])
)

print("[✓] Feature engineering complete")

# ─────────────────────────────────────────────
# 4. LABEL ENCODING
# ─────────────────────────────────────────────
le_cat = LabelEncoder()
le_label = LabelEncoder()

df["category_enc"] = le_cat.fit_transform(df["Category"])
le_label.fit(df["Condition_Class"])

print(f"[✓] Categories   : {list(le_cat.classes_)}")
print(f"[✓] Label classes: {list(le_label.classes_)}")

# ─────────────────────────────────────────────
# 5. DEFINE FEATURES
# ─────────────────────────────────────────────
FEATURES = [
    # Core measurements
    "Temperature_C", "Humidity_%",
    "CO2_ppm", "Ethylene_ppm", "NH3_ppm", "H2S_ppm", "VOC_ppm", "O2_%",
    "Transit_Hours",
    
    # Optimal ranges
    "Opt_Temp_Min_C", "Opt_Temp_Max_C",
    "Opt_Humidity_Min_%", "Opt_Humidity_Max_%",
    
    # Shelf life features
    "Initial_Shelf_Life_hrs", "Remaining_Shelf_Life_hrs",
    "shelf_life_pct", "transit_ratio",
    
    # Deviations
    "temp_deviation", "humidity_deviation",
    "temp_deviation_norm", "humidity_deviation_norm",
    
    # Environmental factors
    "temp_factor", "humidity_factor", "env_stress",
    
    # Gas features
    "CO2_level", "Ethylene_level",
    "log_CO2_ppm", "log_NH3_ppm", "log_H2S_ppm", "log_Ethylene_ppm", "log_VOC_ppm",
    "sqrt_CO2_ppm", "sqrt_VOC_ppm",
    "O2_depletion",
    
    # Gas ratios
    "NH3_to_H2S_ratio", "VOC_to_CO2_ratio", "Ethylene_to_O2_ratio",
    "gas_spoilage_index",
    
    # Interactions
    "temp_x_humidity", "temp_x_transit", "humidity_x_transit", "stress_x_transit",
    
    # Polynomial
    "temp_squared", "humidity_squared", "transit_squared",
    
    # Risk score
    "spoilage_risk_score",
    
    # Category
    "category_enc",
]

X = df[FEATURES].copy()
y_enc = le_label.transform(df["Condition_Class"])

# Handle any infinite or NaN values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

print(f"[✓] Total features: {len(FEATURES)}")

# ─────────────────────────────────────────────
# 6. FEATURE SCALING
# ─────────────────────────────────────────────
print("\n[*] Scaling features...")
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURES, index=X.index)

# ─────────────────────────────────────────────
# 7. HANDLE CLASS IMBALANCE WITH SMOTE
# ─────────────────────────────────────────────
print("\n[*] Applying SMOTE for class balancing...")
print(f"Original class distribution: {np.bincount(y_enc)}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_enc)

print(f"After SMOTE: {np.bincount(y_resampled)}")

# ─────────────────────────────────────────────
# 8. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
print(f"\n[✓] Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 9. RANDOM FOREST WITH OPTIMIZED PARAMS
# ─────────────────────────────────────────────
print("\n[*] Training Random Forest …")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"[✓] Random Forest Accuracy : {rf_acc * 100:.2f}%")
print(classification_report(
    y_test, rf.predict(X_test),
    target_names=le_label.classes_, zero_division=0,
))

# ─────────────────────────────────────────────
# 10. XGBOOST WITH OPTIMIZED PARAMS
# ─────────────────────────────────────────────
print("[*] Training XGBoost …")
n_classes = len(le_label.classes_)
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
print(f"[✓] XGBoost Accuracy       : {xgb_acc * 100:.2f}%")
print(classification_report(
    y_test, xgb.predict(X_test),
    target_names=le_label.classes_, zero_division=0,
))

# ─────────────────────────────────────────────
# 11. LIGHTGBM (ADDITIONAL MODEL)
# ─────────────────────────────────────────────
print("[*] Training LightGBM …")
lgbm = LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgbm.fit(X_train, y_train)
lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))
print(f"[✓] LightGBM Accuracy      : {lgbm_acc * 100:.2f}%")
print(classification_report(
    y_test, lgbm.predict(X_test),
    target_names=le_label.classes_, zero_division=0,
))

# ─────────────────────────────────────────────
# 12. GRADIENT BOOSTING
# ─────────────────────────────────────────────
print("[*] Training Gradient Boosting …")
gb = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"[✓] Gradient Boost Accuracy: {gb_acc * 100:.2f}%")
print(classification_report(
    y_test, gb.predict(X_test),
    target_names=le_label.classes_, zero_division=0,
))

# ─────────────────────────────────────────────
# 13. WEIGHTED ENSEMBLE (OPTIMIZED WEIGHTS)
# ─────────────────────────────────────────────
rf_proba = rf.predict_proba(X_test)
xgb_proba = xgb.predict_proba(X_test)
lgbm_proba = lgbm.predict_proba(X_test)
gb_proba = gb.predict_proba(X_test)

# Weighted ensemble based on individual accuracies
total_acc = rf_acc + xgb_acc + lgbm_acc + gb_acc
w_rf = rf_acc / total_acc
w_xgb = xgb_acc / total_acc
w_lgbm = lgbm_acc / total_acc
w_gb = gb_acc / total_acc

ensemble_proba = (
    w_rf * rf_proba +
    w_xgb * xgb_proba +
    w_lgbm * lgbm_proba +
    w_gb * gb_proba
)
ensemble_preds = np.argmax(ensemble_proba, axis=1)
ens_acc = accuracy_score(y_test, ensemble_preds)

print(f"\n[✓] Weighted Ensemble Accuracy: {ens_acc * 100:.2f}%")
print(f"    Weights → RF:{w_rf:.3f} | XGB:{w_xgb:.3f} | LGBM:{w_lgbm:.3f} | GB:{w_gb:.3f}")
print(classification_report(
    y_test, ensemble_preds,
    target_names=le_label.classes_, zero_division=0,
))

# ─────────────────────────────────────────────
# 14. CONFUSION MATRIX
# ─────────────────────────────────────────────
print("\nConfusion Matrix (Ensemble):")
cm = confusion_matrix(y_test, ensemble_preds)
print(cm)

# ─────────────────────────────────────────────
# 15. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n[*] Top 20 Feature Importances (Random Forest):")
feature_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(20).to_string(index=False))

# ─────────────────────────────────────────────
# 16. SAVE ALL MODELS & ARTIFACTS
# ─────────────────────────────────────────────
joblib.dump(rf, "rf_model_v3.pkl")
joblib.dump(xgb, "xgb_model_v3.pkl")
joblib.dump(lgbm, "lgbm_model_v3.pkl")
joblib.dump(gb, "gb_model_v3.pkl")
joblib.dump(le_cat, "le_cat_v3.pkl")
joblib.dump(le_label, "le_label_v3.pkl")
joblib.dump(scaler, "scaler_v3.pkl")
joblib.dump(FEATURES, "features_v3.pkl")
joblib.dump({
    'rf': w_rf,
    'xgb': w_xgb,
    'lgbm': w_lgbm,
    'gb': w_gb
}, "ensemble_weights_v3.pkl")

print("\n" + "=" * 60)
print("   PKL Files Generated:")
for f in ["rf_model_v3.pkl", "xgb_model_v3.pkl", "lgbm_model_v3.pkl", "gb_model_v3.pkl",
          "le_cat_v3.pkl", "le_label_v3.pkl", "scaler_v3.pkl", 
          "features_v3.pkl", "ensemble_weights_v3.pkl"]:
    if os.path.exists(f):
        size_kb = os.path.getsize(f) / 1024
        print(f"   ✓  {f:<32} {size_kb:>7.1f} KB")

print("=" * 60)
print(f"\nModel Performance Summary:")
print(f"   RF       : {rf_acc * 100:>5.2f}%")
print(f"   XGBoost  : {xgb_acc * 100:>5.2f}%")
print(f"   LightGBM : {lgbm_acc * 100:>5.2f}%")
print(f"   GradBoost: {gb_acc * 100:>5.2f}%")
print(f"   Ensemble : {ens_acc * 100:>5.2f}%")
print("=" * 60)
print("\n[✓] Training complete. All models saved.\n")