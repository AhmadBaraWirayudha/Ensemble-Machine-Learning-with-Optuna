import re

import pandas as pd
import pytest
import responses

from integrations.time3233.reader import (
    find_ra_column,
    parse_timesurf_export,
    submit_measurement,
    _scan_folder_once,
    DEFAULT_SERIAL_PATTERN,
)


# --------------------------------------------------------------------
# find_ra_column - header-matching heuristic
# --------------------------------------------------------------------

@pytest.mark.parametrize("columns,expected", [
    (["Date", "Operator", "Ra (um)", "Rz (um)"], "Ra (um)"),
    (["Measurement", "Ra", "Parameter"], "Ra"),
    (["ra_value", "timestamp"], "ra_value"),
    (["Roughness Ra", "Notes"], "Roughness Ra"),
])
def test_find_ra_column_matches_expected(columns, expected):
    assert find_ra_column(columns) == expected


@pytest.mark.parametrize("columns", [
    ["Date", "Operator", "Range", "Parameter"],  # contain "ra" as a substring, not a token
    ["Timestamp", "Value", "Notes"],
    [],
])
def test_find_ra_column_avoids_false_positives(columns):
    assert find_ra_column(columns) is None


# --------------------------------------------------------------------
# parse_timesurf_export - CSV/Excel parsing
# --------------------------------------------------------------------

def test_parse_timesurf_export_csv(tmp_path):
    df = pd.DataFrame({"Date": ["2026-01-01"] * 3, "Ra (um)": [0.55, 0.61, 0.58], "Rz (um)": [3.1, 3.4, 3.2]})
    path = tmp_path / "export.csv"
    df.to_csv(path, index=False)

    values = parse_timesurf_export(path)
    assert values == [0.55, 0.61, 0.58]


def test_parse_timesurf_export_xlsx(tmp_path):
    df = pd.DataFrame({"Operator": ["A"] * 2, "Ra": [0.7, 0.8]})
    path = tmp_path / "export.xlsx"
    df.to_excel(path, index=False)

    values = parse_timesurf_export(path)
    assert values == [0.7, 0.8]


def test_parse_timesurf_export_explicit_column_override(tmp_path):
    df = pd.DataFrame({"WeirdHeaderName": [0.9, 1.0]})
    path = tmp_path / "export.csv"
    df.to_csv(path, index=False)

    values = parse_timesurf_export(path, ra_column="WeirdHeaderName")
    assert values == [0.9, 1.0]


def test_parse_timesurf_export_raises_clear_error_when_no_ra_column(tmp_path):
    df = pd.DataFrame({"Foo": [1, 2], "Bar": [3, 4]})
    path = tmp_path / "export.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Couldn't find an Ra column"):
        parse_timesurf_export(path)


def test_parse_timesurf_export_drops_non_numeric_rows(tmp_path):
    df = pd.DataFrame({"Ra": ["0.5", "n/a", "0.7"]})
    path = tmp_path / "export.csv"
    df.to_csv(path, index=False)

    values = parse_timesurf_export(path)
    assert values == [0.5, 0.7]


# --------------------------------------------------------------------
# Serial line regex (the actual serial connection can't be tested without
# real hardware - see this module's docstring for why - but the parsing
# regex applied to a line of text is plain Python and fully testable)
# --------------------------------------------------------------------

@pytest.mark.parametrize("line,expected", [
    ("Ra=1.234", 1.234),
    ("Ra: 0.567 um", 0.567),
    ("Ra  0.891", 0.891),
    ("Rz=2.0 Ra=0.432", 0.432),
])
def test_default_serial_pattern_extracts_ra(line, expected):
    match = re.search(DEFAULT_SERIAL_PATTERN, line)
    assert match is not None
    assert float(match.group(1)) == expected


def test_default_serial_pattern_no_match_on_unrelated_line():
    assert re.search(DEFAULT_SERIAL_PATTERN, "Rz=2.0") is None


# --------------------------------------------------------------------
# submit_measurement - HTTP call, mocked (a live-API version of this is
# exercised for real in tests/test_api.py's measurement tests)
# --------------------------------------------------------------------

@responses.activate
def test_submit_measurement_posts_expected_payload():
    responses.add(
        responses.POST, "http://fake-api:8000/measurements",
        json={"status": "recorded", "job_id": "J1"}, status=200,
    )

    result = submit_measurement("http://fake-api:8000", 0.65, job_id="J1", device="TIME3233", raw_payload="Ra=0.65um")

    assert result == {"status": "recorded", "job_id": "J1"}
    sent = responses.calls[0].request
    assert sent.headers.get("X-API-Key") is None


@responses.activate
def test_submit_measurement_includes_api_key_header_when_given():
    responses.add(responses.POST, "http://fake-api:8000/measurements", json={"status": "recorded"}, status=200)

    submit_measurement("http://fake-api:8000", 0.65, api_key="secret123")

    assert responses.calls[0].request.headers["X-API-Key"] == "secret123"


# --------------------------------------------------------------------
# _scan_folder_once - the testable core of watch_folder, without the
# infinite polling loop
# --------------------------------------------------------------------

@responses.activate
def test_scan_folder_once_submits_new_files_and_tracks_seen(tmp_path):
    responses.add(responses.POST, "http://fake-api:8000/measurements", json={"status": "recorded"}, status=200)

    df = pd.DataFrame({"Ra": [0.5, 0.6]})
    df.to_csv(tmp_path / "session1.csv", index=False)

    seen = set()
    results = _scan_folder_once(tmp_path, seen, "http://fake-api:8000", job_id="J1", prompt_for_job_id=False)

    assert len(results) == 2
    assert "session1.csv" in seen
    assert len(responses.calls) == 2

    # a second pass with no new files should submit nothing more
    results_again = _scan_folder_once(tmp_path, seen, "http://fake-api:8000", job_id="J1", prompt_for_job_id=False)
    assert results_again == []
    assert len(responses.calls) == 2


@responses.activate
def test_scan_folder_once_ignores_non_export_files(tmp_path):
    (tmp_path / "notes.txt").write_text("not an export")

    seen = set()
    results = _scan_folder_once(tmp_path, seen, "http://fake-api:8000", prompt_for_job_id=False)

    assert results == []
    assert len(responses.calls) == 0
