import pytest
import responses

from integrations.freecad.roughness_predictor_core import (
    vc_from_rpm,
    rpm_from_vc,
    fz_from_feedrate,
    feedrate_from_fz,
    operation_to_machining_params,
    RoughnessPredictorClient,
    get_attr_chain,
    to_float,
    find_path_jobs,
    get_operations,
    extract_operation_properties,
)


# --------------------------------------------------------------------
# Unit conversions - pure math, checked against the standard machining
# formulas (Vc = pi*D*N/1000; Fz = feed_rate/(N*flutes)) and round-tripped
# through their inverses.
# --------------------------------------------------------------------

def test_vc_from_rpm_matches_hand_calculation():
    # 10mm diameter tool at 1000 RPM: Vc = pi * 10 * 1000 / 1000 = 10*pi
    vc = vc_from_rpm(diameter_mm=10.0, rpm=1000.0)
    assert vc == pytest.approx(10 * 3.14159265, rel=1e-6)


def test_vc_from_rpm_scales_linearly_with_diameter():
    vc_small = vc_from_rpm(diameter_mm=5.0, rpm=1000.0)
    vc_large = vc_from_rpm(diameter_mm=10.0, rpm=1000.0)
    assert vc_large == pytest.approx(2 * vc_small)


def test_rpm_from_vc_is_the_inverse_of_vc_from_rpm():
    original_rpm = 1200.0
    diameter = 8.0
    vc = vc_from_rpm(diameter, original_rpm)
    recovered_rpm = rpm_from_vc(vc, diameter)
    assert recovered_rpm == pytest.approx(original_rpm)


@pytest.mark.parametrize("diameter,rpm", [(0, 1000), (-5, 1000), (10, 0), (10, -100)])
def test_vc_from_rpm_rejects_non_positive_inputs(diameter, rpm):
    with pytest.raises(ValueError):
        vc_from_rpm(diameter, rpm)


def test_fz_from_feedrate_matches_hand_calculation():
    # feed rate 600 mm/min, 1000 RPM, 4 flutes -> Fz = 600/(1000*4) = 0.15 mm/tooth
    fz = fz_from_feedrate(feedrate_mm_per_min=600.0, rpm=1000.0, num_flutes=4)
    assert fz == pytest.approx(0.15)


def test_feedrate_from_fz_is_the_inverse_of_fz_from_feedrate():
    original_feedrate = 450.0
    fz = fz_from_feedrate(original_feedrate, rpm=900.0, num_flutes=3)
    recovered_feedrate = feedrate_from_fz(fz, rpm=900.0, num_flutes=3)
    assert recovered_feedrate == pytest.approx(original_feedrate)


@pytest.mark.parametrize("feedrate,rpm,flutes", [(0, 1000, 4), (600, 0, 4), (600, 1000, 0), (-600, 1000, 4)])
def test_fz_from_feedrate_rejects_non_positive_inputs(feedrate, rpm, flutes):
    with pytest.raises(ValueError):
        fz_from_feedrate(feedrate, rpm, flutes)


def test_operation_to_machining_params_shape():
    params = operation_to_machining_params(
        diameter_mm=10.0, rpm=1000.0, feedrate_mm_per_min=600.0, num_flutes=4, step_down_mm=1.5, job_id="OP-1",
    )
    assert set(params.keys()) == {"Vc", "Fz", "ap", "job_id"}
    assert params["ap"] == 1.5
    assert params["job_id"] == "OP-1"
    assert params["Vc"] == pytest.approx(vc_from_rpm(10.0, 1000.0))
    assert params["Fz"] == pytest.approx(fz_from_feedrate(600.0, 1000.0, 4))


def test_operation_to_machining_params_within_this_projects_training_envelope():
    # Sanity check: parameters in a plausible real machining range should
    # land inside the (Vc: 7.5-17.5, Fz: 0.05-0.15, ap: 0.75-1.5) envelope
    # this project's model was actually trained on (see UPGRADE_NOTES.md) -
    # if this ever drifts wildly out of range it's a sign the conversion
    # direction or units got flipped somewhere.
    params = operation_to_machining_params(
        diameter_mm=10.0, rpm=350.0, feedrate_mm_per_min=140.0, num_flutes=4, step_down_mm=1.0,
    )
    assert 5 < params["Vc"] < 20
    assert 0.03 < params["Fz"] < 0.2


# --------------------------------------------------------------------
# RoughnessPredictorClient - HTTP calls, mocked (exercised for real
# against the live API in tests/test_api.py)
# --------------------------------------------------------------------

@responses.activate
def test_client_predict_sends_expected_payload_and_parses_response():
    responses.add(
        responses.POST, "http://fake-api:8000/predict",
        json={"recommended_prediction": 0.75, "recommended_model": "GradientBoosting", "job_id": "OP-1"},
        status=200,
    )

    client = RoughnessPredictorClient(api_url="http://fake-api:8000")
    result = client.predict(vc=10.0, fz=0.1, ap=1.0, job_id="OP-1")

    assert result["recommended_prediction"] == 0.75
    sent_body = responses.calls[0].request.body
    assert b'"job_id": "OP-1"' in sent_body or b'"job_id":"OP-1"' in sent_body


@responses.activate
def test_client_predict_batch():
    responses.add(
        responses.POST, "http://fake-api:8000/predict/batch",
        json={"count": 2, "predictions": [{"recommended_prediction": 0.5}, {"recommended_prediction": 0.6}]},
        status=200,
    )

    client = RoughnessPredictorClient(api_url="http://fake-api:8000")
    result = client.predict_batch([
        {"Vc": 10.0, "Fz": 0.1, "ap": 1.0},
        {"Vc": 12.0, "Fz": 0.1, "ap": 1.0},
    ])

    assert result["count"] == 2


@responses.activate
def test_client_submit_measurement():
    responses.add(responses.POST, "http://fake-api:8000/measurements", json={"status": "recorded"}, status=200)

    client = RoughnessPredictorClient(api_url="http://fake-api:8000", api_key="secret123")
    client.submit_measurement(ra_measured=0.6, job_id="OP-1", device="manual")

    assert responses.calls[0].request.headers["X-API-Key"] == "secret123"


@responses.activate
def test_client_health():
    responses.add(responses.GET, "http://fake-api:8000/health", json={"status": "ok"}, status=200)
    client = RoughnessPredictorClient(api_url="http://fake-api:8000")
    assert client.health() == {"status": "ok"}


# --------------------------------------------------------------------
# Defensive extraction from a FreeCAD Path document - this logic doesn't
# actually import FreeCAD (duck-typed attribute traversal), so it's
# tested here against plain mock objects standing in for the real
# FreeCAD.Path.Job / Path operation / ToolController object model. These
# mocks are a best-effort approximation of that model (see this module's
# docstring in roughness_predictor_core.py for the caveat on exact
# property names across FreeCAD versions) - what's actually verified here
# is that the traversal logic behaves correctly *given* objects shaped
# like that, gracefully handling missing attributes either way.
# --------------------------------------------------------------------

class MockTool:
    def __init__(self, diameter=None, cutting_edge_count=None):
        if diameter is not None:
            self.Diameter = diameter
        if cutting_edge_count is not None:
            self.CuttingEdgeCount = cutting_edge_count


class MockToolController:
    def __init__(self, spindle_speed=None, horiz_feed=None, tool=None):
        if spindle_speed is not None:
            self.SpindleSpeed = spindle_speed
        if horiz_feed is not None:
            self.HorizFeed = horiz_feed
        if tool is not None:
            self.Tool = tool


class MockOperation:
    def __init__(self, label, tool_controller=None, step_down=None):
        self.Label = label
        if tool_controller is not None:
            self.ToolController = tool_controller
        if step_down is not None:
            self.StepDown = step_down


class MockOperationsGroup:
    def __init__(self, operations):
        self.Group = operations


class MockJob:
    def __init__(self, label, operations):
        self.Label = label
        self.Operations = MockOperationsGroup(operations)


class MockDoc:
    def __init__(self, objects):
        self.Objects = objects


def test_get_attr_chain_tries_paths_in_order_and_skips_missing():
    obj = MockToolController(spindle_speed=1000.0)
    assert get_attr_chain(obj, "NotThere", "SpindleSpeed") == 1000.0


def test_get_attr_chain_returns_none_when_nothing_resolves():
    obj = MockToolController()
    assert get_attr_chain(obj, "SpindleSpeed", "Tool.Diameter") is None


def test_get_attr_chain_handles_nested_paths():
    obj = MockToolController(tool=MockTool(diameter=10.0))
    assert get_attr_chain(obj, "Tool.Diameter") == 10.0


def test_to_float_strips_freecad_style_units():
    assert to_float("600 mm/min") == 600.0
    assert to_float("10.5mm") == 10.5


def test_to_float_passes_through_plain_numbers():
    assert to_float(1000) == 1000.0
    assert to_float(0.15) == 0.15


def test_to_float_returns_default_for_unparseable():
    assert to_float(None) is None
    assert to_float("garbage") is None
    assert to_float("garbage", default=0.0) == 0.0


def test_extract_operation_properties_fully_populated():
    op = MockOperation(
        "Profile1",
        tool_controller=MockToolController(spindle_speed=1000.0, horiz_feed=600.0, tool=MockTool(diameter=10.0, cutting_edge_count=4)),
        step_down=1.5,
    )
    props = extract_operation_properties(op)

    assert props["diameter_mm"] == 10.0
    assert props["rpm"] == 1000.0
    assert props["feedrate_mm_per_min"] == 600.0
    assert props["num_flutes"] == 4
    assert props["step_down_mm"] == 1.5
    assert props["label"] == "Profile1"


def test_extract_operation_properties_partially_populated_does_not_crash():
    # only a label and step-down - no tool controller at all
    op = MockOperation("Pocket1", step_down=0.5)
    props = extract_operation_properties(op)

    assert props["diameter_mm"] is None
    assert props["rpm"] is None
    assert props["feedrate_mm_per_min"] is None
    assert props["num_flutes"] is None
    assert props["step_down_mm"] == 0.5
    assert props["label"] == "Pocket1"


def test_extract_operation_properties_bare_object_does_not_crash():
    class Bare:
        Label = "Bare1"

    props = extract_operation_properties(Bare())
    assert props["diameter_mm"] is None
    assert props["label"] == "Bare1"


def test_find_path_jobs_identifies_job_like_objects_only():
    job = MockJob("Job", [MockOperation("Op1")])

    class UnrelatedObject:
        pass

    doc = MockDoc([job, UnrelatedObject(), UnrelatedObject()])
    found = find_path_jobs(doc)

    assert found == [job]


def test_get_operations_handles_nested_group_structure():
    op1 = MockOperation("Op1")
    op2 = MockOperation("Op2")
    job = MockJob("Job", [op1, op2])

    ops = get_operations(job)
    assert ops == [op1, op2]


def test_get_operations_returns_empty_list_when_no_operations_group():
    class JobWithoutOperations:
        Label = "EmptyJob"

    assert get_operations(JobWithoutOperations()) == []
