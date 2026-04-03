import os
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

# ─────────────────────────────────────────────
# FOOD KNOWLEDGE BASE  (derived from final_food_monitoring_with_thresholds)
# shelf_life is in DAYS; opt_temp in °C; opt_humidity in %
# ─────────────────────────────────────────────
FOOD_DATA = {
    # ── Fruits ───────────────────────────────────────────────────────────
    "Apple":             {"category": "Pome Fruit",     "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Banana":            {"category": "Tropical Fruit", "shelf_life_days": 6,    "opt_temp": (13,  15),  "opt_humidity": (85, 95)},
    "Mango":             {"category": "Tropical Fruit", "shelf_life_days": 7,    "opt_temp": (13,  15),  "opt_humidity": (85, 95)},
    "Orange":            {"category": "Citrus",         "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Papaya":            {"category": "Tropical Fruit", "shelf_life_days": 7,    "opt_temp": (12,  15),  "opt_humidity": (85, 95)},
    "Grapes":            {"category": "Berry",          "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Watermelon":        {"category": "Melon",          "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Strawberry":        {"category": "Berry",          "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Strawberries":      {"category": "Berry",          "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    # ── Vegetables ───────────────────────────────────────────────────────
    "Tomato":            {"category": "Vegetable",      "shelf_life_days": 7,    "opt_temp": (5,   10),  "opt_humidity": (90, 97)},
    "Leafy Greens":      {"category": "Vegetable",      "shelf_life_days": 5,    "opt_temp": (2,   6),   "opt_humidity": (90, 98)},
    "Bell Pepper":       {"category": "Vegetable",      "shelf_life_days": 7,    "opt_temp": (4,   8),   "opt_humidity": (90, 95)},
    "Cucumber":          {"category": "Vegetable",      "shelf_life_days": 7,    "opt_temp": (5,   10),  "opt_humidity": (90, 95)},
    "Onion":             {"category": "Root Veg",       "shelf_life_days": 30,   "opt_temp": (4,   10),  "opt_humidity": (80, 90)},
    "Potato":            {"category": "Root Veg",       "shelf_life_days": 30,   "opt_temp": (4,   10),  "opt_humidity": (80, 90)},
    # ── Dairy ────────────────────────────────────────────────────────────
    "Whole Milk":        {"category": "Dairy",          "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Yogurt":            {"category": "Dairy",          "shelf_life_days": 13,   "opt_temp": (2,   6),   "opt_humidity": (79, 88)},
    "Cheese":            {"category": "Dairy",          "shelf_life_days": 10,   "opt_temp": (2,   6),   "opt_humidity": (79, 88)},
    "Cheddar Cheese":    {"category": "Dairy",          "shelf_life_days": 13,   "opt_temp": (2,   6),   "opt_humidity": (79, 88)},
    "Butter":            {"category": "Dairy",          "shelf_life_days": 13,   "opt_temp": (2,   6),   "opt_humidity": (79, 88)},
    "Paneer":            {"category": "Dairy",          "shelf_life_days": 13,   "opt_temp": (2,   6),   "opt_humidity": (79, 88)},
    "Eggs":              {"category": "Poultry",        "shelf_life_days": 18,   "opt_temp": (3,   8),   "opt_humidity": (70, 80)},
    # ── Meat & Seafood ───────────────────────────────────────────────────
    "Chicken":           {"category": "Poultry",        "shelf_life_days": 5,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Chicken Breast":    {"category": "Poultry",        "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Beef Steak":        {"category": "Meat",           "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Lamb Chops":        {"category": "Meat",           "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Pork Loin":         {"category": "Meat",           "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Pork Ribs":         {"category": "Meat",           "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Shrimp":            {"category": "Seafood",        "shelf_life_days": 2,    "opt_temp": (0,   2),   "opt_humidity": (80, 90)},
    "Fish Fillet":       {"category": "Seafood",        "shelf_life_days": 2,    "opt_temp": (0,   2),   "opt_humidity": (80, 90)},
    "Fresh Salmon":      {"category": "Seafood",        "shelf_life_days": 7,    "opt_temp": (0,   2),   "opt_humidity": (80, 90)},
    # ── Bakery & Processed ───────────────────────────────────────────────
    "White Bread":       {"category": "Bread",          "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Whole Wheat Bread": {"category": "Bread",          "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Croissant":         {"category": "Pastry",         "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Chocolate Cake":    {"category": "Pastry",         "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Cookies":           {"category": "Snack",          "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Potato Chips":      {"category": "Snack",          "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    # ── Ready Meals & Preserves ──────────────────────────────────────────
    "Cooked Rice":       {"category": "Ready Meal",     "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Pasta Salad":       {"category": "Ready Meal",     "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Sandwich":          {"category": "Ready Meal",     "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Cheese Pizza":      {"category": "Ready Meal",     "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Tomato Sauce":      {"category": "Preserve",       "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
    "Fruit Jam":         {"category": "Preserve",       "shelf_life_days": 7,    "opt_temp": (2,   6),   "opt_humidity": (80, 90)},
}

# ─────────────────────────────────────────────
# LOAD V3 MODELS
# ─────────────────────────────────────────────
BASE = os.path.dirname(__file__)

rf       = joblib.load(os.path.join(BASE, "rf_model_v3.pkl"))
xgb      = joblib.load(os.path.join(BASE, "xgb_model_v3.pkl"))
le_cat   = joblib.load(os.path.join(BASE, "le_cat_v3.pkl"))
le_label = joblib.load(os.path.join(BASE, "le_label_v3.pkl"))
scaler   = joblib.load(os.path.join(BASE, "scaler_v3.pkl"))
FEATURES = joblib.load(os.path.join(BASE, "features_v3.pkl"))

print("[OK] All v3 models loaded.")
print(f"[OK] Label classes: {list(le_label.classes_)}")

# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)


def build_feature_vector(food_type, temperature, humidity,
                         co2_ppm, ethylene_ppm, nh3_ppm,
                         h2s_ppm, voc_ppm, o2_pct, transit_hrs):
    """Build the EXACT same feature vector that trainmodel.py used (v3 schema)."""
    meta = FOOD_DATA[food_type]

    opt_temp_min, opt_temp_max = meta["opt_temp"]
    opt_hum_min,  opt_hum_max  = meta["opt_humidity"]
    shelf_life_days            = meta["shelf_life_days"]
    initial_shelf_hrs          = shelf_life_days * 24  # days → hours

    # ── Deviations ──────────────────────────────────────────────────────
    temp_deviation = (
        max(0, temperature - opt_temp_max) +
        max(0, opt_temp_min - temperature)
    )
    humidity_deviation = (
        max(0, humidity - opt_hum_max) +
        max(0, opt_hum_min - humidity)
    )

    # Normalized deviations
    temp_range     = (opt_temp_max - opt_temp_min) + 1e-6
    humidity_range = (opt_hum_max  - opt_hum_min)  + 1e-6
    temp_deviation_norm     = temp_deviation     / temp_range
    humidity_deviation_norm = humidity_deviation / humidity_range

    # ── Environmental factors ────────────────────────────────────────────
    temp_factor     = (1
                       + 0.08 * max(0, temperature - opt_temp_max)
                       + 0.04 * max(0, opt_temp_min - temperature))
    humidity_factor = 1 + 0.03 * humidity_deviation
    env_stress      = temp_factor * humidity_factor

    # ── Shelf life ───────────────────────────────────────────────────────
    hours_consumed        = transit_hrs * env_stress
    remaining_shelf_hrs   = max(0.0, initial_shelf_hrs - hours_consumed)
    shelf_life_pct        = remaining_shelf_hrs / (initial_shelf_hrs + 1e-6)
    transit_ratio         = transit_hrs          / (initial_shelf_hrs + 1e-6)

    # ── Gas binning ──────────────────────────────────────────────────────
    # CO2 level: 0=<100, 1=100-250, 2=250-500, 3=>500
    if   co2_ppm <= 100:  co2_level = 0.0
    elif co2_ppm <= 250:  co2_level = 1.0
    elif co2_ppm <= 500:  co2_level = 2.0
    else:                 co2_level = 3.0

    # Ethylene level: 0=<0.1, 1=0.1-1.0, 2=1.0-10, 3=>10
    if   ethylene_ppm <= 0.1:  ethylene_level = 0.0
    elif ethylene_ppm <= 1.0:  ethylene_level = 1.0
    elif ethylene_ppm <= 10:   ethylene_level = 2.0
    else:                      ethylene_level = 3.0

    # ── Log / sqrt transforms ────────────────────────────────────────────
    log_co2_ppm      = np.log1p(co2_ppm)
    log_nh3_ppm      = np.log1p(nh3_ppm)
    log_h2s_ppm      = np.log1p(h2s_ppm)
    log_ethylene_ppm = np.log1p(ethylene_ppm)
    log_voc_ppm      = np.log1p(voc_ppm)
    sqrt_co2_ppm     = np.sqrt(co2_ppm)
    sqrt_voc_ppm     = np.sqrt(voc_ppm)

    # Oxygen depletion
    o2_depletion = 21.0 - o2_pct

    # ── Gas ratios ───────────────────────────────────────────────────────
    nh3_to_h2s_ratio      = nh3_ppm      / (h2s_ppm  + 1e-6)
    voc_to_co2_ratio      = voc_ppm      / (co2_ppm  + 1e-6)
    ethylene_to_o2_ratio  = ethylene_ppm / (o2_pct   + 1e-6)

    gas_spoilage_index = (
        0.3 * log_nh3_ppm +
        0.3 * log_h2s_ppm +
        0.2 * log_ethylene_ppm +
        0.2 * log_voc_ppm
    )

    # ── Interaction & polynomial features ───────────────────────────────
    temp_x_humidity    = temperature * humidity
    temp_x_transit     = temperature * transit_hrs
    humidity_x_transit = humidity    * transit_hrs
    stress_x_transit   = env_stress  * transit_hrs

    temp_squared     = temperature  ** 2
    humidity_squared = humidity     ** 2
    transit_squared  = transit_hrs  ** 2

    # ── Risk score ───────────────────────────────────────────────────────
    spoilage_risk_score = (
        0.4 * temp_deviation_norm +
        0.2 * humidity_deviation_norm +
        0.2 * transit_ratio +
        0.2 * (1 - shelf_life_pct)
    )

    # ── Category encoding ────────────────────────────────────────────────
    try:
        category_enc = int(le_cat.transform([meta["category"]])[0])
    except ValueError:
        category_enc = 0

    # ── Assemble in EXACT trainmodel.py FEATURES order ──────────────────
    row = {
        "Temperature_C":          temperature,
        "Humidity_%":             humidity,
        "CO2_ppm":                co2_ppm,
        "Ethylene_ppm":           ethylene_ppm,
        "NH3_ppm":                nh3_ppm,
        "H2S_ppm":                h2s_ppm,
        "VOC_ppm":                voc_ppm,
        "O2_%":                   o2_pct,
        "Transit_Hours":          transit_hrs,
        "Opt_Temp_Min_C":         opt_temp_min,
        "Opt_Temp_Max_C":         opt_temp_max,
        "Opt_Humidity_Min_%":     opt_hum_min,
        "Opt_Humidity_Max_%":     opt_hum_max,
        "Initial_Shelf_Life_hrs": initial_shelf_hrs,
        "Remaining_Shelf_Life_hrs": remaining_shelf_hrs,
        "shelf_life_pct":         shelf_life_pct,
        "transit_ratio":          transit_ratio,
        "temp_deviation":         temp_deviation,
        "humidity_deviation":     humidity_deviation,
        "temp_deviation_norm":    temp_deviation_norm,
        "humidity_deviation_norm": humidity_deviation_norm,
        "temp_factor":            temp_factor,
        "humidity_factor":        humidity_factor,
        "env_stress":             env_stress,
        "CO2_level":              co2_level,
        "Ethylene_level":         ethylene_level,
        "log_CO2_ppm":            log_co2_ppm,
        "log_NH3_ppm":            log_nh3_ppm,
        "log_H2S_ppm":            log_h2s_ppm,
        "log_Ethylene_ppm":       log_ethylene_ppm,
        "log_VOC_ppm":            log_voc_ppm,
        "sqrt_CO2_ppm":           sqrt_co2_ppm,
        "sqrt_VOC_ppm":           sqrt_voc_ppm,
        "O2_depletion":           o2_depletion,
        "NH3_to_H2S_ratio":       nh3_to_h2s_ratio,
        "VOC_to_CO2_ratio":       voc_to_co2_ratio,
        "Ethylene_to_O2_ratio":   ethylene_to_o2_ratio,
        "gas_spoilage_index":     gas_spoilage_index,
        "temp_x_humidity":        temp_x_humidity,
        "temp_x_transit":         temp_x_transit,
        "humidity_x_transit":     humidity_x_transit,
        "stress_x_transit":       stress_x_transit,
        "temp_squared":           temp_squared,
        "humidity_squared":       humidity_squared,
        "transit_squared":        transit_squared,
        "spoilage_risk_score":    spoilage_risk_score,
        "category_enc":           category_enc,
    }

    x_raw = pd.DataFrame([row])[FEATURES]

    # Apply the same RobustScaler used during training
    x_scaled = scaler.transform(x_raw)
    x_scaled = pd.DataFrame(x_scaled, columns=FEATURES)

    return x_scaled, remaining_shelf_hrs, initial_shelf_hrs, temp_deviation


def make_prediction(food_type, temperature, humidity,
                    co2_ppm, ethylene_ppm, nh3_ppm,
                    h2s_ppm, voc_ppm, o2_pct, transit_hrs):
    """Run RF + XGBoost ensemble and return a unified result dict."""
    x, remaining_shelf, initial_shelf, temp_dev = build_feature_vector(
        food_type, temperature, humidity,
        co2_ppm, ethylene_ppm, nh3_ppm, h2s_ppm, voc_ppm, o2_pct, transit_hrs
    )

    # Random Forest
    rf_pred_idx   = rf.predict(x)[0]
    rf_proba      = rf.predict_proba(x)[0]
    rf_label      = le_label.inverse_transform([rf_pred_idx])[0]
    rf_confidence = float(rf_proba[rf_pred_idx]) * 100

    # XGBoost
    xgb_pred_idx   = int(xgb.predict(x)[0])
    xgb_proba      = xgb.predict_proba(x)[0]
    xgb_label      = le_label.inverse_transform([xgb_pred_idx])[0]
    xgb_confidence = float(xgb_proba[xgb_pred_idx]) * 100

    # Ensemble: RF 50% + XGBoost 50%
    weights     = 0.5 * rf_proba + 0.5 * xgb_proba
    final_idx   = int(np.argmax(weights))
    final_label = le_label.classes_[final_idx]
    
    # Map 'Spoiled' to 'Critical' for user perception
    if final_label == "Spoiled":
        final_label = "Critical"
        
    confidence  = float(weights[final_idx]) * 100

    # Spoilage progress: % of shelf life consumed
    spoilage_pct = round((1 - remaining_shelf / initial_shelf) * 100, 1)
    spoilage_pct = max(0, min(100, spoilage_pct))

    # Quality score: inverse of spoilage, penalised by deviations + gas signals
    quality_base = remaining_shelf / initial_shelf * 100
    gas_penalty  = (
        max(0, co2_ppm - 100) / 25
        + min(5, nh3_ppm * 2)
        + min(5, h2s_ppm * 5)
        + min(5, ethylene_ppm * 0.5)
    )
    penalty  = min(30, temp_dev * 2 + gas_penalty)
    quality  = round(max(0, quality_base - penalty), 1)

    meta         = FOOD_DATA[food_type]
    opt_temp_str = f"{meta['opt_temp'][0]}–{meta['opt_temp'][1]}°C"

    # Recommendation
    if final_label == "Safe":
        recommendation = "Maintain current temperature and humidity. Continue delivery as scheduled."
    elif final_label == "Warning":
        recommendation = (
            f"Reduce temperature to {opt_temp_str}. Monitor gas levels (NH₃, H₂S, CO₂). "
            "Consider expediting delivery to prevent further degradation."
        )
    else:  # Critical
        recommendation = (
            "Critical spoilage risk detected. Immediate corrective action required. "
            f"Target temperature: {opt_temp_str}. "
            "Evaluate whether cargo is fit for delivery."
        )

    remaining_days = round(remaining_shelf / 24, 1)
    initial_days   = round(initial_shelf   / 24, 1)

    return {
        "status":          final_label,
        "quality_score":   quality,
        "confidence":      round(confidence, 1),
        "remaining_shelf": remaining_days,
        "initial_shelf":   initial_days,
        "spoilage_pct":    spoilage_pct,
        "temp_deviation":  round(temp_dev, 1),
        "recommendation":  recommendation,
        "rf": {
            "label":      rf_label,
            "confidence": round(rf_confidence, 1),
        },
        "xgb": {
            "label":      xgb_label,
            "confidence": round(xgb_confidence, 1),
        },
        "food_meta": {
            "category":   meta["category"],
            "shelf_days": initial_days,
            "opt_temp":   opt_temp_str,
        },
        "gas_readings": {
            "co2_ppm":      co2_ppm,
            "ethylene_ppm": ethylene_ppm,
            "nh3_ppm":      nh3_ppm,
            "h2s_ppm":      h2s_ppm,
            "voc_ppm":      voc_ppm,
            "o2_pct":       o2_pct,
        },
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    foods = list(FOOD_DATA.keys())
    return render_template("index.html", foods=foods, food_data=FOOD_DATA)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data         = request.get_json()
        food_type    = data["food_type"]
        temperature  = float(data["temperature"])
        humidity     = float(data["humidity"])
        co2_ppm      = float(data.get("co2_ppm",      data.get("gas_ppm", 400)))
        ethylene_ppm = float(data.get("ethylene_ppm", 0.0))
        nh3_ppm      = float(data.get("nh3_ppm",      0.0))
        h2s_ppm      = float(data.get("h2s_ppm",      0.0))
        voc_ppm      = float(data.get("voc_ppm",      0.0))
        o2_pct       = float(data.get("o2_pct",       21.0))  # default ambient O₂
        transit_hrs  = float(data["transit_hrs"])
        season       = data.get("season", "Summer")  # informational, not used by model

        if food_type not in FOOD_DATA:
            return jsonify({"error": f"Unknown food type: {food_type}"}), 400

        result = make_prediction(
            food_type, temperature, humidity,
            co2_ppm, ethylene_ppm, nh3_ppm, h2s_ppm, voc_ppm, o2_pct, transit_hrs
        )
        result["season"] = season
        return jsonify(result)

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/foods")
def get_foods():
    return jsonify(FOOD_DATA)


if __name__ == "__main__":
    print("\n[OK] FreshSense running -> http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)