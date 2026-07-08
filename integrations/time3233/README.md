# TIME3233 Surface Roughness Tester Integration

Feeds real physical measurements into this project (`POST /measurements`),
so `GET /accuracy/report` can compare what the model predicted against
what a part actually measured - a stronger signal than
`monitoring/drift_monitor.py`, which only watches whether *inputs* have
shifted, not whether predictions are still correct.

## What's verified, and what isn't

The TIME3233 (Beijing TIME High Technology) is a portable stylus-type
roughness tester: 50mm traversing length, 800um vertical range, ~55
parameters including Ra/Rz/Rq, and it transfers readings to a PC over
RS232 - normally via the vendor's own "TIMESurf" software, which can also
export a session to Excel. Those facts are confirmed from public product
listings.

**What's not publicly documented** (and wasn't found anywhere while
building this): the exact byte-level serial protocol TIMESurf itself
speaks to the device - baud rate, framing, command/response structure.
Beijing TIME High Technology doesn't appear to publish it. Guessing at
that and shipping it as "the protocol" would be worse than not shipping
anything - it would look like it works right up until it silently
doesn't. So this integration gives you two paths instead of one guess:

## Path 1 (recommended): watch TIMESurf's own export

TIMESurf can export a measurement session to Excel. Point this at the
folder it exports to:

```bash
pip install -r requirements.txt   # openpyxl, pandas already included
python integrations/time3233/reader.py watch --folder "C:\path\to\timesurf\exports" --job-id PART-42
```

It polls the folder, and for each new `.xlsx`/`.csv` file, looks for a
column whose header is recognizably "Ra" (`Ra`, `Ra (um)`, `ra_value`,
`Roughness Ra`, ... - see `find_ra_column()`) and submits each value found
to the API. If it can't find one, it lists the columns it did find so you
can pass `--ra-column` explicitly. This path only depends on TIMESurf's
export format, not on reverse-engineering an undocumented serial protocol
- the vendor's own software does the actual device communication.

Omit `--job-id` to be prompted for one per file instead (useful if you're
running one export per part rather than one long-running watch session).

## Path 2 (best-effort): direct serial

If you'd rather skip TIMESurf and read the device directly:

```bash
pip install pyserial

# First, see what the device actually sends - don't skip this step
python integrations/time3233/reader.py raw-capture --port /dev/ttyUSB0

# Then, once you've seen the real format and confirmed (or adjusted) the
# pattern below matches it:
python integrations/time3233/reader.py serial --port /dev/ttyUSB0 --job-id PART-42
```

Defaults to 9600 baud / 8N1 (common for this instrument class - verify
against your device/manual) and a regex (`Ra\s*[:=]?\s*([\d.]+)`) that
matches common patterns like `Ra=1.234` or `Ra: 0.567 um`. Override with
`--baud` / `--pattern` once `raw-capture` shows you the real format.
**Don't trust this path on a new device without running raw-capture
first** - it may just work, since a lot of instruments like this do print
a plain ASCII line per reading, but "may" is doing real work in that
sentence.

## Either way

Both paths end up calling the same `POST /measurements` endpoint. Tag
predictions and measurements for the same part with the same `job_id` (see
the main README's Authentication/API sections) to have them show up
together in `GET /accuracy/report`:

```bash
curl -X POST http://127.0.0.1:8000/predict -d '{"Vc": 10, "Fz": 0.1, "ap": 1.0, "job_id": "PART-42"}'
# ... machine the part, measure it ...
curl -X POST http://127.0.0.1:8000/measurements -d '{"Ra_measured": 0.65, "job_id": "PART-42", "device": "TIME3233"}'
curl http://127.0.0.1:8000/accuracy/report
```

## Testing

`tests/test_time3233_integration.py` covers everything that doesn't
require the physical device: the Ra-column detection heuristic, Excel/CSV
parsing, the regex pattern against sample lines of text, and the HTTP
call to `/measurements` (mocked). It cannot and does not test the actual
serial connection or the real TIMESurf export layout - those need
verification against real hardware, which wasn't available while building
this.
