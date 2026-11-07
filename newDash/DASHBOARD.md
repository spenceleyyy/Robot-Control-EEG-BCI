# Session dashboard

A 1920x1080 window built for screen recording: live EEG, the mu rhythm
changing, the decode, the arm and the participant, all in one frame.

## Install

Copy these files into the `mi-testbed` folder, next to `sim_arm.py`:

    bci_dashboard.py   the main loop
    neuro.py           display maths (ERD, spectra, topomap, quality)
    viz.py             the panels
    canvas.py          drawing, with pygame and matplotlib backends
    camera.py          threaded webcam
    session_log.py     recording
    preview_dashboard.py   renders a PNG with fake data, no hardware needed
    dashboard.bat      launcher

Then:

    dashboard.bat

It installs `pygame` and `opencv-python` into the existing `.venv` the
first time.

## Running it

    dashboard.bat                      live EEG from LSL
    dashboard.bat --source replay      replay a recorded .fif
    dashboard.bat --source keyboard    arrow keys, for rehearsing the shot
    dashboard.bat --no-camera          no webcam
    dashboard.bat --camera 1           a different webcam

The first 20 seconds are a baseline: sit still, eyes open, looking at the
screen. Every ERD number afterwards is a percentage change from that rest
period, so it is worth getting right. `--baseline 30` makes it longer.

Keys: `SPACE` restart trial, `B` re-take the baseline, `F` fullscreen,
`C` next camera, `S` screenshot, `ESC` quit.

## What gets saved

Into `data/live_<date>_<time>.*`:

- `.csv` one row per decoder window - probability, evidence, command, task
  state, dwell, and per-channel mu power and ERD%. Opens in Excel.
- `.npz` the same as arrays, plus **the raw EEG** and the baseline power.
  This is the one to re-analyse from: with the raw data the session can be
  re-decoded later with a better model, or re-plotted at a different
  timescale to match the edit.
- `.json` the decoder used, its offline accuracy, montage, band, timings.

`--no-raw` omits the EEG if you want small files; raw is about 2.5MB per
minute at 500Hz and 14 channels, which is worth paying.

## Checking the layout without hardware

    python preview_dashboard.py --state left
    python preview_dashboard.py --state right
    python preview_dashboard.py --state rest

Renders the real panels with synthetic EEG to `dashboard_preview.png`. Same
panel code the live window runs.

## How it attaches to the existing code

It does not re-implement the decode. `Tap` wraps the existing
`EvidenceDecoder.feed()` and `on_decision`, so the display and the log see
exactly the samples the classifier saw, and the task is the unmodified
`sim_arm.Task`. Nothing in `decoder_source.py` or `sim_arm.py` changes.

Two conventions worth knowing when reading the screen:

- Amber is C3, the **left** hemisphere, which drives the **right** hand.
  Blue is C4, the right hemisphere, driving the left hand. The colours are
  the same in every panel.
- Positive evidence means the **left** hand. The decoder panel's ends are
  labelled by hand *and* hemisphere so the two do not get crossed.
