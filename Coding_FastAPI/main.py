from fastapi import FastAPI, UploadFile, File, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import joblib
import numpy as np
import pandas as pd
import re
from scipy.stats import kurtosis, skew
from pathlib import Path

from auth_service import SESSION_SECRET
from auth_guard import NotAuthenticatedException, require_auth
from auth_interceptor import AuthInterceptor
from auth_routes import router as auth_router

app = FastAPI(title="Bearing AI Predictor")

# ─── Middleware (ลำดับสำคัญ: AuthInterceptor ต้องอยู่หลัง SessionMiddleware) ──
app.add_middleware(AuthInterceptor)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# ─── Auth Routes (/login-page, /login, /home, /logout) ───────────────────────
app.include_router(auth_router)

# ─── Templates ───────────────────────────────────────────────────────────────
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ─── Exception Handler: redirect ไป login เมื่อยังไม่ได้ login ──────────────
@app.exception_handler(NotAuthenticatedException)
async def not_authenticated_handler(request: Request, exc: NotAuthenticatedException):
    return RedirectResponse(url="/login-page", status_code=302)


# --- Load models once at startup ---
MODEL_PATH = Path(__file__).parent / "bearing_model.pkl"
data = joblib.load(MODEL_PATH)
model = data["model"]
features_list = data["features"]

LOOSENESS_A_PATH = Path(__file__).parent / "LoosenessTypeA_model202605061825.pkl"
looseness_a_data = joblib.load(LOOSENESS_A_PATH)
looseness_a_model = looseness_a_data["model"]
looseness_a_features = looseness_a_data["features"]

LOOSENESS_C_PATH = Path(__file__).parent / "LoosenessTypeC_model202605061825.pkl"
looseness_c_data = joblib.load(LOOSENESS_C_PATH)
looseness_c_model = looseness_c_data["model"]
looseness_c_features = looseness_c_data["features"]


SEVERITY_MAP = {
    1: {"label": "Stage A: ปกติ (Normal)",          "color": "#22c55e", "icon": "✅"},
    2: {"label": "Stage B: เริ่มผิดปกติ (Warning)", "color": "#f59e0b", "icon": "⚠️"},
    3: {"label": "Stage C: อันตราย (Alarm)",         "color": "#ef4444", "icon": "🚨"},
    4: {"label": "Stage D: วิกฤต (Critical)",        "color": "#7f1d1d", "icon": "💀"},
}


def extract_features_from_content(content: str):
    try:
        data_start = content.find("---------")
        if data_start == -1:
            return None
        raw_text = content[data_start:]
        numbers = re.findall(r"-?\d*\.\d+|-?\d+", str(raw_text))
        if len(numbers) < 2:
            return None
        amplitudes = np.array([float(numbers[i]) for i in range(1, len(numbers), 2)])
        amplitudes = amplitudes[np.isfinite(amplitudes)]
        if len(amplitudes) < 2:
            return None
        rms = float(np.sqrt(np.mean(amplitudes**2)))
        peak = float(np.max(np.abs(amplitudes)))
        peak_to_peak = float(np.max(amplitudes) - np.min(amplitudes))
        kurt = float(kurtosis(amplitudes, fisher=False))
        skw = float(skew(amplitudes))
        crest = float(peak / rms) if rms > 1e-10 else 0.0
        features = {
            "RMS": rms, "Peak": peak, "Peak_to_Peak": peak_to_peak,
            "Kurtosis": kurt, "Skewness": skw, "Crest_Factor": crest,
        }
        if not all(np.isfinite(v) for v in features.values()):
            return None
        return features
    except Exception:
        return None


@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...),
    current_user: str = Depends(require_auth),
):
    results = []
    failed = []

    for file in files:
        content = (await file.read()).decode("latin-1")
        feats = extract_features_from_content(content)
        if feats is None:
            failed.append(file.filename)
            continue

        # FM1 — Bearing Defect
        df1 = pd.DataFrame([feats])[features_list]
        pred1 = int(model.predict(df1)[0])
        probs1 = model.predict_proba(df1)[0].tolist()
        classes1 = [int(c) for c in model.classes_]

        # FM2 — Looseness Type A
        df2 = pd.DataFrame([feats])[looseness_a_features]
        pred2 = int(looseness_a_model.predict(df2)[0])
        probs2 = looseness_a_model.predict_proba(df2)[0].tolist()
        classes2 = [int(c) for c in looseness_a_model.classes_]

        # FM3 — Looseness Type C
        df3 = pd.DataFrame([feats])[looseness_c_features]
        pred3 = int(looseness_c_model.predict(df3)[0])
        probs3 = looseness_c_model.predict_proba(df3)[0].tolist()
        classes3 = [int(c) for c in looseness_c_model.classes_]

        severity = SEVERITY_MAP.get(pred1, {"label": str(pred1), "color": "#6b7280", "icon": "❓"})
        results.append({
            "filename": file.filename,
            "features": feats,
            "prediction": pred1,
            "severity_label": severity["label"],
            "severity_color": severity["color"],
            "severity_icon": severity["icon"],
            "classes": classes1,
            "probabilities": probs1,
            "fm2_prediction": pred2,
            "fm2_classes": classes2,
            "fm2_probabilities": probs2,
            "fm3_prediction": pred3,
            "fm3_classes": classes3,
            "fm3_probabilities": probs3,
        })

    return {"results": results, "failed": failed}


@app.get("/feature-importance")
def feature_importance(current_user: str = Depends(require_auth)):
    importance = model.feature_importances_.tolist()
    return {"features": features_list, "importance": importance}


# ─── Grouped Analysis (analysis.html) ────────────────────────────────────────

def parse_filename(filename: str):
    """Return (machine_name, measurement_point) from filename like P-566S-1H_16-Oct-25_..."""
    name = re.sub(r'\.rtf$', '', filename, flags=re.IGNORECASE)
    first_segment = name.split('_')[0]          # e.g. 'P-566S-1H'
    parts = first_segment.split('-')            # ['P', '566S', '1H']
    if len(parts) < 2:
        return first_segment, 'UNKNOWN'
    measurement_point = parts[-1]               # '1H'
    machine_name = '-'.join(parts[:-1])         # 'P-566S'
    return machine_name, measurement_point


@app.post("/analyze-group")
async def analyze_group(
    files: list[UploadFile] = File(...),
    current_user: str = Depends(require_auth),
):
    parsed = []
    failed = []

    for file in files:
        content = (await file.read()).decode("latin-1")
        machine, point = parse_filename(file.filename)
        feats = extract_features_from_content(content)
        if feats is None:
            failed.append(file.filename)
            continue
        parsed.append({"machine": machine, "point": point,
                        "features": feats, "filename": file.filename})

    if not parsed:
        return {"machines": [], "failed": failed}

    # FM1 – Bearing Defect: ML model (Stage A-D)
    for p in parsed:
        df = pd.DataFrame([p["features"]])[features_list]
        p["fm1"] = int(model.predict(df)[0])

    # FM2 – Looseness Type A: ML model
    for p in parsed:
        df = pd.DataFrame([p["features"]])[looseness_a_features]
        p["fm2"] = int(looseness_a_model.predict(df)[0])

    # FM3 – Looseness Type C: ML model
    for p in parsed:
        df = pd.DataFrame([p["features"]])[looseness_c_features]
        p["fm3"] = int(looseness_c_model.predict(df)[0])

    # Group by machine name
    machines: dict = {}
    for p in parsed:
        m = p["machine"]
        if m not in machines:
            machines[m] = {}
        machines[m][p["point"]] = {
            "filename": p["filename"],
            "features": p["features"],
            "fm1": p["fm1"],
            "fm2": p["fm2"],
            "fm3": p["fm3"],
        }

    result = [
        {"machine": m, "points": dict(sorted(pts.items()))}
        for m, pts in sorted(machines.items())
    ]
    return {"machines": result, "failed": failed}


@app.get("/analysis")
def analysis_page(
    request: Request,
    current_user: str = Depends(require_auth),
):
    return templates.TemplateResponse(
        "analysis.html", {"request": request, "current_user": current_user}
    )


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/")
def index(current_user: str = Depends(require_auth)):
    return RedirectResponse(url="/analysis", status_code=302)
