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
    job_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional identifier for the part/job this prediction is for. "
            "If supplied, a later physical measurement (POST /measurements) "
            "tagged with the same job_id can be compared against this "
            "prediction via GET /accuracy/report. Auto-generated if omitted."
        ),
    )

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
    job_id: str = Field(..., description="Echoes input.job_id, or an auto-generated one if none was supplied - use this to tag a later measurement.")

    svr_prediction: float = Field(..., description="Predicted Ra (surface roughness, um) from the SVR model")
    gpr_prediction: float = Field(..., description="Predicted Ra from the Gaussian Process model")
    gpr_uncertainty_std: float = Field(..., description="GPR's predictive standard deviation at this point")
    rf_prediction: float = Field(..., description="Predicted Ra from the Random Forest model")
    gbm_prediction: float = Field(..., description="Predicted Ra from the Gradient Boosting model")
    power_law_prediction: float = Field(..., description="Predicted Ra from the classical Ra=C*Vc^a*Fz^b*ap^c power-law model")
    weighted_ensemble_prediction: float = Field(..., description="alpha*GPR + (1-alpha)*SVR")
    ensemble_alpha: float
    stacking_ensemble_prediction: float = Field(..., description="RidgeCV meta-learner over [SVR, GPR, RandomForest, GradientBoosting]")

    recommended_model: str = Field(..., description="Whichever candidate (including individual models) had the best out-of-fold RMSE at training time")
    recommended_prediction: float = Field(..., description="Prediction from recommended_model - use this one if you just want a single number")

    range_check: RangeCheck

    model_trained_at: str


class MeasurementSubmission(BaseModel):
    """A physical roughness measurement, e.g. from a stylus tester."""

    Ra_measured: float = Field(..., gt=0, description="Measured Ra (surface roughness, um)")
    job_id: Optional[str] = Field(default=None, description="Same job_id used when calling /predict for this part, so the two can be compared later.")
    device: Optional[str] = Field(default=None, description="Instrument name/model, e.g. 'TIME3233'")
    raw_payload: Optional[str] = Field(default=None, description="Original raw reading/line from the device, kept for traceability/debugging")


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
    auth_enabled: bool = False
