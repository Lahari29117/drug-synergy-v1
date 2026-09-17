from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "synergy_pipeline.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file was not found at: {MODEL_PATH}. "
        "Run train_model.py before starting the API."
    )

pipeline = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Drug Synergy V1 API",
    description="A learning prototype that predicts a synthetic synergy score.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=APP_FOLDER / "static"), name="static")
templates = Jinja2Templates(directory=str(APP_FOLDER / "templates"))


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    drug_a: str = Field(min_length=1)
    drug_b: str = Field(min_length=1)
    cell_line: str = Field(min_length=1)

    drug_a_molecular_weight: float = Field(gt=0)
    drug_b_molecular_weight: float = Field(gt=0)
    drug_a_logp: float
    drug_b_logp: float
    dose_a_um: float = Field(gt=0)
    dose_b_um: float = Field(gt=0)
    cell_line_growth_rate: float = Field(gt=0)


class PredictionResponse(BaseModel):
    predicted_synergy_score: float
    model_type: str
    prototype_notice: str


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_file": MODEL_PATH.name,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    experiment = pd.DataFrame([request.model_dump()])
    predicted_score = float(pipeline.predict(experiment)[0])

    return {
        "predicted_synergy_score": round(predicted_score, 3),
        "model_type": "RandomForestRegressor",
        "prototype_notice": (
            "This is a V1 learning prototype trained on synthetic data. "
            "It is not a clinical or scientific recommendation."
        ),
    }