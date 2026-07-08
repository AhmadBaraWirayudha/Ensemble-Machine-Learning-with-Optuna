"""
FreeCAD-independent core for the roughness-prediction integration: unit
conversions between what FreeCAD's Path (CAM) workbench naturally exposes
- spindle RPM, feed rate in mm/min, number of flutes, step-down - and the
Vc/Fz/ap this project's model actually takes, plus a small HTTP client for
the prediction API.

Deliberately has zero dependency on the `FreeCAD`/`FreeCADGui` modules, so
it can be imported and unit-tested in a normal Python environment (see
tests/test_freecad_integration.py) without FreeCAD installed. The actual
FreeCAD-facing macro (FreeCAD_RoughnessPredictor.FCMacro in this same
folder) imports this module and is the only part that touches FreeCAD's
own object model - see that file's docstring for why it's written
defensively (hasattr checks, fallback prompts) rather than assuming exact
property names, since those can vary a little between FreeCAD versions
and operation types in ways this module's author could not verify without
FreeCAD itself installed (not available in the environment this was
built in - see integrations/freecad/README.md).
"""

import math
import re

import requests


def vc_from_rpm(diameter_mm: float, rpm: float) -> float:
    """Cutting speed Vc (m/min) from tool diameter (mm) and spindle speed
    (RPM): Vc = pi * D * N / 1000."""
    if diameter_mm <= 0 or rpm <= 0:
        raise ValueError("diameter_mm and rpm must both be positive")
    return math.pi * diameter_mm * rpm / 1000.0


def rpm_from_vc(vc: float, diameter_mm: float) -> float:
    """Inverse of vc_from_rpm - what spindle speed gives a target Vc for a
    given tool diameter. Useful for suggesting a speed, not just reading one."""
    if vc <= 0 or diameter_mm <= 0:
        raise ValueError("vc and diameter_mm must both be positive")
    return vc * 1000.0 / (math.pi * diameter_mm)


def fz_from_feedrate(feedrate_mm_per_min: float, rpm: float, num_flutes: int) -> float:
    """Feed per tooth Fz (mm/tooth) from FreeCAD's feed rate (mm/min),
    spindle speed (RPM), and the tool's flute count:
    Fz = FeedRate / (RPM * num_flutes)."""
    if feedrate_mm_per_min <= 0 or rpm <= 0 or num_flutes <= 0:
        raise ValueError("feedrate_mm_per_min, rpm, and num_flutes must all be positive")
    return feedrate_mm_per_min / (rpm * num_flutes)


def feedrate_from_fz(fz: float, rpm: float, num_flutes: int) -> float:
    """Inverse of fz_from_feedrate - the FreeCAD-style feed rate (mm/min)
    for a target Fz, given speed and flute count."""
    if fz <= 0 or rpm <= 0 or num_flutes <= 0:
        raise ValueError("fz, rpm, and num_flutes must all be positive")
    return fz * rpm * num_flutes


class RoughnessPredictorClient:
    """Thin HTTP client for this project's prediction API, used by the
    FreeCAD macro (and directly testable without FreeCAD)."""

    def __init__(self, api_url="http://127.0.0.1:8000", api_key=None, timeout=5):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self):
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def health(self) -> dict:
        resp = requests.get(f"{self.api_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def predict(self, vc: float, fz: float, ap: float, job_id: str = None) -> dict:
        payload = {"Vc": vc, "Fz": fz, "ap": ap}
        if job_id:
            payload["job_id"] = job_id
        resp = requests.post(f"{self.api_url}/predict", json=payload, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def predict_batch(self, operations: list) -> dict:
        """`operations`: list of {"Vc":.., "Fz":.., "ap":.., "job_id": optional}."""
        resp = requests.post(
            f"{self.api_url}/predict/batch", json={"items": operations}, headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def submit_measurement(self, ra_measured: float, job_id: str = None, device: str = "manual") -> dict:
        resp = requests.post(
            f"{self.api_url}/measurements",
            json={"Ra_measured": ra_measured, "job_id": job_id, "device": device},
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()


def operation_to_machining_params(diameter_mm: float, rpm: float, feedrate_mm_per_min: float, num_flutes: int, step_down_mm: float, job_id: str = None) -> dict:
    """
    Convert one FreeCAD Path operation's raw properties into the
    {Vc, Fz, ap, job_id} shape /predict expects. This is the one function
    the macro calls per operation - having it here (rather than duplicated
    inline in the macro) means the conversion logic itself is covered by
    tests/test_freecad_integration.py even though the macro that gathers
    diameter_mm/rpm/feedrate_mm_per_min/num_flutes/step_down_mm from an
    actual FreeCAD document cannot be.
    """
    return {
        "Vc": vc_from_rpm(diameter_mm, rpm),
        "Fz": fz_from_feedrate(feedrate_mm_per_min, rpm, num_flutes),
        "ap": step_down_mm,
        "job_id": job_id,
    }


# --------------------------------------------------------------------
# Defensive extraction from a FreeCAD Path operation object.
#
# None of this actually imports FreeCAD - it's duck-typed attribute
# traversal over whatever object it's handed, which is what makes it
# testable here against plain mock objects (see
# tests/test_freecad_integration.py) standing in for real FreeCAD Path
# objects that couldn't be verified without FreeCAD installed - see this
# package's README.md.
# --------------------------------------------------------------------

def get_attr_chain(obj, *paths):
    """Try each dotted attribute path in turn (e.g. "ToolController.HorizFeed"),
    return the first that resolves to a non-None value, else None. Never raises."""
    for path in paths:
        target = obj
        try:
            for part in path.split("."):
                target = getattr(target, part)
            if target is not None:
                return target
        except AttributeError:
            continue
    return None


def to_float(value, default=None):
    """FreeCAD quantities (e.g. '600 mm/min', '10.5mm') stringify with a
    unit, with or without a separating space; plain floats/ints pass
    through unchanged. Extracts the leading numeric portion either way."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(value))
        return float(match.group(1)) if match else default


def find_path_jobs(doc):
    """Every object in a FreeCAD document that looks like a Path Job (has
    an Operations group). Identified by duck-typing (has an .Operations
    attribute, or a .Proxy whose class name contains "Job") rather than an
    exact TypeId string, since that's exactly the kind of detail that
    varies across FreeCAD versions in ways that couldn't be verified
    without FreeCAD installed."""
    jobs = []
    for obj in doc.Objects:
        if hasattr(obj, "Operations") or (hasattr(obj, "Proxy") and "Job" in type(obj.Proxy).__name__):
            jobs.append(obj)
    return jobs


def get_operations(job):
    group = get_attr_chain(job, "Operations.Group", "Operations")
    if group is None:
        return []
    try:
        return list(group)
    except TypeError:
        return [group]


def extract_operation_properties(op):
    """Best-effort extraction of what's needed from one Path operation
    object: tool diameter, spindle speed, feed rate, flute count, step
    down. Returns a dict with possibly-None values where nothing plausible
    was found - the calling UI is expected to show these in editable
    fields and let the user fill in gaps, not trust this blindly."""

    tool_controller = get_attr_chain(op, "ToolController")

    diameter = get_attr_chain(tool_controller, "Tool.Diameter", "Tool.ShapeParams.Diameter")
    rpm = get_attr_chain(tool_controller, "SpindleSpeed", "Tool.SpindleSpeed")
    feed_rate = get_attr_chain(tool_controller, "HorizFeed", "VertFeed") or get_attr_chain(op, "Feed")
    num_flutes = get_attr_chain(tool_controller, "Tool.CuttingEdgeCount", "Tool.FluteCount", "Tool.Flutes")
    step_down = get_attr_chain(op, "StepDown", "DepthOfCut", "MaxDepthOfCut")

    flutes_value = to_float(num_flutes, 0)

    return {
        "diameter_mm": to_float(diameter),
        "rpm": to_float(rpm),
        "feedrate_mm_per_min": to_float(feed_rate),
        "num_flutes": int(flutes_value) if flutes_value else None,
        "step_down_mm": to_float(step_down),
        "label": getattr(op, "Label", str(op)),
    }
