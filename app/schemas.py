"""Pydantic schemas for the surface-roughness prediction API."""

from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class MachiningParams(BaseModel):
    """
    Raw machining parameters for one cut. Field names match the standard
    machining-engineering symbols used throughout this project and in the
    training data, so factory-side integrators working from the same
    process sheets can map fields directly.
    """

    Vc: float = Field(..., gt=0, le=1000, description="Cutting speed (m/min)")
    Fz: float = Field(..., gt=0, le=10, description="Feed per tooth (mm/tooth)")
    ap: float = Field(..., gt=0, le=100, description="Axial depth of cut (mm)")

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"Vc": 10.0, "Fz": 0.1, "ap": 1.0}]}
    )


class RangeCheck(BaseModel):
    within_training_envelope: bool
    out_of_range_features: dict = Field(
        default_factory=dict,
        description="Present only for features outside the (min, max) the model was trained on.",
    )


class PredictionResponse(BaseModel):
    input: MachiningParams

    svr_prediction: float = Field(..., description="Predicted Ra (surface roughness, um) from the SVR model")
    gpr_prediction: float = Field(..., description="Predicted Ra from the Gaussian Process model")
    gpr_uncertainty_std: float = Field(..., description="GPR's predictive standard deviation at this point")
    weighted_ensemble_prediction: float = Field(..., description="alpha*GPR + (1-alpha)*SVR")
    ensemble_alpha: float
    stacking_ensemble_prediction: float = Field(..., description="RidgeCV meta-learner over [SVR, GPR]")

    recommended_model: str = Field(..., description="Which ensemble had the better out-of-fold RMSE at training time")
    recommended_prediction: float = Field(..., description="Prediction from recommended_model - use this one if you just want a single number")

    range_check: RangeCheck

    model_trained_at: str


class BatchPredictionRequest(BaseModel):
    items: list[MachiningParams] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    recommended_model: Optional[str] = None
    model_trained_at: Optional[str] = None
    n_train_samples: Optional[int] = None
