# Motor imagery testbed

One thing to click. It connects to your actiCHamp, records a resting baseline,
a real-movement control block and a motor imagery block, saves everything, and
tells you whether you have usable ERD.

About 22 minutes with the cap on.

---

## One-time setup

### 1. Get the amplifier onto the network as an LSL stream

This project is set up for the **direct connector** route, which skips
BrainVision Recorder entirely.

1. Install the actiCHamp USB driver and confirm Windows sees the amp.
2. Get `LSL-actiCHamp` from https://github.com/brain-products/LSL-actiCHamp
   (you may also need the Visual Studio 2012 *and* 2015-2019 C++
   redistributables, both).
3. Run it, enter your device serial number, hit **Link**.

Two consequences of skipping Recorder, both already handled in `config.py`:

- **The connector sends raw, reference-free EEG.** Recorder was what applied
  your FCz reference. So `REREFERENCE = "average"` is not optional here.
- **The connector does not carry channel labels.** `CHANNEL_NAMES_OVERRIDE`
  is set to `ACTICAP_32`, the standard 32-channel actiCAP ordering. If your
  cap is a different size, edit that list. Extra AUX or trigger channels keep
  their own names.

If you ever do get Recorder working, the alternative is Recorder + the
`LSL-BrainVisionRDA` connector, which carries labels and referencing for you.
Set `CHANNEL_NAMES_OVERRIDE = None` in that case.

### 2. Which electrodes to gel

Only 14 sites are needed. The amp streams all 32 channels regardless; the
analysis drops the ones you left dry.

| Physical # | Site | Why |
|---|---|---|
| 8, 24, 25 | C3, Cz, C4 | the measurement |
| 7, 6, 12, 11 | FC1, FC5, CP1, CP5 | Laplacian ring around C3 |
| 29, 28, 23, 22 | FC2, FC6, CP2, CP6 | Laplacian ring around C4 |
| 1, 32 | Fp1, Fp2 | forehead, no hair. Blinks + artifact control |
| 17 | Oz | alpha blocking, proves the chain works |

Plus your reference (FCz) and ground (Fpz).

Edit `USE_CHANNELS` in `config.py` to change this. Set it to `None` to use
everything. Dropping Oz costs you the chain check; dropping the forehead pair
costs you the artifact control that catches a muscle-driven false positive.

Leave the unused electrodes sitting dry in their holders. They will look like
garbage in the monitor, which is expected and harmless - they are excluded
everywhere.

### 3. Impedance, without Recorder

The direct connector cannot measure impedance - Brain Products says to use
Recorder for that. So this project ships `signal_check.py`, a live per-channel
quality monitor that infers contact problems from the signal itself: dead
channels, electrodes acting as mains antennas, and unstable drifting contacts.

It scores each channel against its peers rather than against fixed limits,
which matters for reference-free data where noise is often common to the
whole array. Array-wide mains noise gets its own warning, because that is a
ground or reference problem rather than one bad electrode.

It is a proxy, not ohms. But it catches what actually ruins recordings.
The launcher opens it automatically so you can gel while watching.

**With active electrodes**, note that the hum column is a weaker signal than
it would be for passive electrodes - active electrodes suppress the antenna
effect that makes a badly gelled passive electrode pick up mains noise. Watch
the drift column instead. Active electrodes also tolerate considerably higher
impedance than passive ones, so you do not need to chase the numbers as hard,
though contact quality still matters.

### 4. Python

Any Python 3.9+. The launcher builds its own virtual environment on first run.

---

## Running it

**Windows:** double-click `start.bat`
**macOS / Linux:** `chmod +x start.sh` once, then run `./start.sh`

It will:

1. Check the LSL stream and print your channel list
2. Open the live signal quality monitor, so you can gel against it
3. Run the session full screen
4. Save to `data/`
5. Run the analysis and print a verdict

`ESC` aborts at any point and still saves what was recorded.

To re-analyse an existing recording without recording again:

```
.venv/bin/python analyze_erd.py data/mi_20260920_143000_raw.fif
```

---

## What the session actually does

| Block | Duration | Purpose |
|---|---|---|
| Eyes open / eyes closed | 2 min | Finds your individual alpha/mu peak. Alpha must rise when you close your eyes — if it doesn't, the recording chain is broken and nothing else is interpretable. |
| Real movement, 10 trials/class | ~3 min | Positive control. You physically squeeze. Real movement gives much larger ERD than imagery, so a flat result here means the setup is at fault, not you. |
| Motor imagery, 40 trials/class | ~13 min | The actual measurement. Break every 20 trials. |

Trial structure, anchored on cue onset:

```
-2.0s   fixation cross
 0.0s   arrow appears        <- epoch anchor, marker pushed here
 1.25s  arrow disappears, cross stays
 4.0s   cross disappears
        1.5-2.5s blank, jittered  ("blink now")
```

---

## What you get out

In `data/`, per session:

- `*_raw.fif` — MNE-readable, filtered nothing, annotations included
- `*_raw.npz` — raw backup with LSL timestamps, in case anything goes wrong
- `*_1_resting_psd.png` — eyes open vs closed spectrum, mu band marked
- `*_2_erd_movement.png` — time-frequency maps, real movement
- `*_3_erd_imagery.png` — time-frequency maps, imagery. **Blue = ERD.**
- `*_4_topomap.png` — where on the scalp the mu drop happened
- `*_report.txt` — everything the console printed

The report gives you:

- whether alpha blocking worked (recording chain sanity)
- your individual mu band, not a generic 8–13 Hz guess
- mu power change per class at C3/Cz/C4
- lateralisation: left hand should desynchronise C4, right hand C3
- CSP+LDA cross-validated accuracy
- **an artifact control**: the same classifier run on frontal channels only.
  If frontal-only beats sensorimotor, you are decoding eye movement or jaw
  muscle. This is the most common way BCI projects fool themselves.

---

## Reading the result

The verdict at the bottom of the report covers the normal cases. The key
things to keep in mind:

- **A weak first session means nothing.** ERD magnitude grows with practice.
- **Kinesthetic, not visual.** If you were picturing your hand from outside
  rather than feeling the squeeze from inside, you measured the wrong thing.
  Redo it.
- **Check the movement block before blaming yourself.** Flat there = setup.
- **Jaw tension ruins everything.** It produces broadband EMG that sits right
  on top of the beta band.

---

## Settings

Everything tunable is in `config.py`: trial counts, timings, channel groups,
frequency bands, rejection threshold. Add `"feet"` to `CLASSES` if you want a
third class later.

If auto-detection of units ever guesses wrong, set `UNITS_SCALE` to `1e-6`
(microvolts in) or `1.0` (volts in) explicitly.

---

## Controlling the robot arm

**Windows:** double-click `robot.bat`  **macOS / Linux:** `./robot.sh`

First run installs MuJoCo. Then pick a mode:

| Mode | Needs | What it does |
|---|---|---|
| **1 Replay** | nothing | Streams your recorded session through the decoder at real speed. Your actual brain data from that day, moving the arm. |
| **2 Live** | cap on, connector linked | Real-time control. |
| **3 Keyboard** | nothing | Arrow keys. Use this first to check the arm works at all. |

### How control is divided

EEG gives roughly one reliable bit every second or two - enough to *choose*,
not enough to steer six joints. So the work is split:

- **You** move the arm left and right above the row of blocks, and pick one by
  holding the arm over it for 1.5 seconds. The block brightens as the hold fills.
- **The arm** does the rest by itself: descends, grasps, lifts, carries the
  block to the yellow drop zone, releases, and returns.

This is shared autonomy, and it is how practical non-invasive BCIs work.

Two details worth knowing:

- **After the arm returns, there is a 2-second grace period.** It comes to rest
  over the middle block. To take that block, just wait. To take another, start
  moving within those 2 seconds.
- **Near a block, the arm helps you stop on it.** Stopping precisely is the
  hardest thing to do with this kind of signal.

In any mode, **up-arrow** grabs the nearest block immediately.

### The replay is honest

Replay uses `decoder_holdout.pkl`, trained on all but your last 20 imagery
trials, and replays only those 20. So it decodes brain data the decoder has
never seen - the same situation as live use. The cursor sphere turns orange
during a left cue and blue during a right cue, and the console shows how often
the decoded direction matched the cue.

Live mode uses `decoder.pkl`, trained on every trial, since more data helps
when there is nothing to hold back for.

### Tuning

At the top of `sim_arm.py`: `GAIN_Y` (speed), `DWELL_S` (hold time to grab),
`ASSIST_K` (how hard the arm helps you stop), `DEADZONE`.
