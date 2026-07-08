#!/usr/bin/env python3
"""
Example of a third-party system (an MES, a SCADA gateway, a quality-control
script, anything that isn't this Python codebase) querying the prediction
API over plain HTTP. This is the point of the API microservice upgrade:
the model is now reachable from any language/platform that can make an
HTTP request, not just from someone clicking a button in a Tkinter window.

Usage:
    python scripts/example_client.py
    python scripts/example_client.py --url http://192.168.1.50:8000
    python scripts/example_client.py --api-key some-long-random-string  # if CNC_API_KEY is set on the service
"""

import argparse
import sys

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of the running API")
    parser.add_argument("--api-key", default=None, help="X-API-Key value, if the service has CNC_API_KEY set")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    print(f"Checking {base}/health ...")
    try:
        health = requests.get(f"{base}/health", timeout=5).json()
    except requests.exceptions.ConnectionError:
        print(f"Could not reach {base}. Is the API running?")
        print("Start it with: uvicorn app.main:app --reload")
        return 1

    print(f"  status={health['status']}  recommended_model={health.get('recommended_model')}")
    if health.get("auth_enabled") and not args.api_key:
        print("  This service requires an API key - pass --api-key some-key-value.")
        return 1
    if health["status"] != "ok":
        print("Service is up but has no trained model loaded. Run scripts/train_model.py first.")
        return 1

    print(f"\nSingle prediction, {base}/predict:")
    job = {"Vc": 12.5, "Fz": 0.1, "ap": 1.0}
    resp = requests.post(f"{base}/predict", json=job, headers=headers, timeout=5)
    resp.raise_for_status()
    result = resp.json()
    print(f"  input: {job}")
    print(f"  recommended prediction (Ra): {result['recommended_prediction']:.4f} um  "
          f"[{result['recommended_model']}]")
    print(f"  within training envelope: {result['range_check']['within_training_envelope']}")

    print(f"\nBatch prediction, {base}/predict/batch (e.g. scoring a day's job queue at once):")
    jobs = [
        {"Vc": 7.5, "Fz": 0.05, "ap": 0.75},
        {"Vc": 12.5, "Fz": 0.1, "ap": 1.0},
        {"Vc": 17.5, "Fz": 0.15, "ap": 1.5},
    ]
    resp = requests.post(f"{base}/predict/batch", json={"items": jobs}, headers=headers, timeout=5)
    resp.raise_for_status()
    batch_result = resp.json()
    for job, pred in zip(jobs, batch_result["predictions"]):
        print(f"  {job} -> Ra={pred['recommended_prediction']:.4f} um")

    print(f"\nDrift report, {base}/drift/report:")
    resp = requests.get(f"{base}/drift/report", headers=headers, timeout=5)
    resp.raise_for_status()
    drift = resp.json()
    if drift.get("status") == "insufficient_data":
        print(f"  {drift['message']}")
    else:
        print(f"  overall_verdict: {drift['overall_verdict']}")

    print(f"\nFeature importance, {base}/model/feature-importance:")
    resp = requests.get(f"{base}/model/feature-importance", headers=headers, timeout=5)
    if resp.status_code == 404:
        print(f"  {resp.json()['detail']}")
    else:
        resp.raise_for_status()
        by_variable = resp.json()["by_variable"]
        for model_name in next(iter(by_variable.values())).keys():
            ranked = sorted(by_variable.items(), key=lambda kv: kv[1][model_name], reverse=True)
            print(f"  {model_name} ranks: " + " > ".join(var for var, _ in ranked))

    print(f"\nAuto-retrain history, {base}/retrain/history:")
    resp = requests.get(f"{base}/retrain/history", headers=headers, timeout=5)
    resp.raise_for_status()
    history = resp.json()
    if history["count"] == 0:
        print("  No auto-retrain runs yet (run `python -m monitoring.retrain_trigger` to start one).")
    else:
        for event in history["events"]:
            print(f"  {event['timestamp']}: {event['action']} ({event.get('reason', '-')})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
