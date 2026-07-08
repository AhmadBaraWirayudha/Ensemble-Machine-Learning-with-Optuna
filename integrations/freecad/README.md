# FreeCAD Integration

Predicts surface roughness directly from a FreeCAD Path (CAM) job's
parameters - cutting speed and feed per tooth are derived from the tool
diameter, spindle RPM, feed rate, and flute count you've already set up
for the operation, rather than needing to re-enter them.

## What's verified, and what isn't

This was built without a FreeCAD installation available to test against.
FreeCAD's Path workbench and Python API are well-documented and openly
developed, so the general approach here (Job -> Operations -> a
ToolController with a Tool, feed rates, spindle speed) is solid - but
exact property names have shifted a little across FreeCAD versions and
between operation types (Profile vs Pocket vs Adaptive have their own
property sets), in ways that couldn't be confirmed without FreeCAD itself.

So the integration is split in two, matching how confident each half is:

- **`roughness_predictor_core.py`** - the unit conversions (Vc from
  diameter+RPM, Fz from feed rate+RPM+flutes) and the attribute-extraction
  logic. Zero FreeCAD dependency, so it's fully unit-tested against plain
  Python (see `tests/test_freecad_integration.py` in the main repo) -
  including against mock objects standing in for FreeCAD's Path object
  model, which is as far as that can be verified without FreeCAD installed.
- **`FreeCAD_RoughnessPredictor.FCMacro`** - the actual FreeCAD-facing
  macro. Reads properties defensively (tries several plausible attribute
  paths per value; a property not being found just leaves that field
  blank, not a crash) and **always shows the derived values in editable
  fields before predicting** - review and correct these the first time you
  run it against a given FreeCAD version, rather than trusting extraction
  blindly. If no Path job is found at all in the active document, you can
  still type values in by hand and use the panel as a calculator + API client.

## Install

Copy this whole `integrations/freecad/` folder into your FreeCAD macro
directory (Edit -> Preferences -> General -> Macro, "Macro path" - the
folder needs both files, since the macro imports the core module from the
same directory it's run from).

```bash
pip install requests   # the macro's only non-stdlib dependency
```

## Use

Macro -> Macros -> run `FreeCAD_RoughnessPredictor.FCMacro`, with a Path
job open in the active document. A panel opens:

1. Pick a Path operation from the dropdown (auto-populated from the
   active document's Path job(s), if any are found).
2. Check/correct the derived Tool diameter, RPM, Feed rate, Flute count,
   and Step down fields.
3. Click **Predict Ra** - shows the derived Vc/Fz/ap, the recommended
   model's prediction, and whether the point falls inside this project's
   training envelope. A `job_id` is generated from the job/operation
   names (editable).
4. After machining and measuring the part, enter the actual Ra in
   **After machining** and click **Submit measurement** - tagged with the
   same `job_id`, so it shows up matched against the prediction in
   `GET /accuracy/report`.

If your service has `CNC_API_KEY` set, enter it in the API key field
first (see the main README's Authentication section).

## Testing

`tests/test_freecad_integration.py` covers the unit conversions (checked
against hand calculations and round-tripped through their inverses), the
HTTP client (mocked), and the extraction logic (against mock Path
objects). It cannot and does not test the Qt panel itself or real FreeCAD
property names - see the caveat above.
