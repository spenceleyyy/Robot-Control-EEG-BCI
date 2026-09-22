"""
All tunable settings live here. Edit this file, not the others.
"""

# ---------------------------------------------------------------- LSL / stream
# Substring to look for in the LSL stream name. Leave as "" to accept the
# first stream whose type is "EEG".
# BrainVision RDA -> LSL usually names the stream "BrainVision RDA" or similar.
# The actiCHamp LSL connector usually names it "actiCHamp-<serial>" or "EEG".
STREAM_NAME_HINT = ""
STREAM_TYPE = "EEG"
STREAM_RESOLVE_TIMEOUT = 30.0  # keeps retrying, so you can start the connector meanwhile

# If the amp sends microvolts (almost always true for BrainVision), the data is
# scaled by 1e-6 before it goes into MNE, which expects volts.
# "auto" inspects the amplitude and decides. Override with 1e-6 or 1.0 if the
# auto-detect ever guesses wrong.
UNITS_SCALE = "auto"

# ------------------------------------------------------------------- paradigm
CLASSES = ["left", "right"]  # add "feet" if you want a 3rd class later

# Resting baseline block, used to find your individual alpha/mu peak and to
# confirm the whole recording chain works (alpha must rise when eyes close).
BASELINE_EYES_OPEN_S = 60.0
BASELINE_EYES_CLOSED_S = 60.0

# Motor EXECUTION block. You actually squeeze your hand. This is the positive
# control: if there is no ERD here, the problem is the setup, not your imagery.
ME_TRIALS_PER_CLASS = 10

# Motor IMAGERY block. This is the real measurement.
MI_TRIALS_PER_CLASS = 40
MI_TRIALS_PER_RUN = 20  # total trials between self-paced breaks

# Trial timing, seconds. Anchored on cue onset (t = 0).
T_FIXATION = 2.0  # fixation cross alone, before the cue
T_CUE_VISIBLE = 1.25  # how long the arrow stays on screen
T_TASK = 4.0  # imagine/move from cue onset until this time
T_ITI_MIN = 1.5  # blank inter-trial interval, jittered
T_ITI_MAX = 2.5

# ------------------------------------------------------------------- display
FULLSCREEN = True
WINDOW_SIZE = (1280, 800)  # used only when FULLSCREEN is False
BG_COLOR = (0, 0, 0)
FG_COLOR = (230, 230, 230)
CUE_COLOR = (120, 200, 255)
FIX_COLOR = (230, 230, 230)
FONT_SIZE = 34
FONT_SIZE_BIG = 64

# Some LSL sources send channels with no useful labels ("ch1", "ch2", ...).
# The direct actiCHamp LSL connector often does this, while the Recorder/RDA
# route carries your workspace labels through automatically.
#
# Set this to a list of names, in physical channel order, to relabel them.
# ACTICAP_32 below is the standard 32-channel actiCAP snap ordering
# (reference FCz, ground Fpz). Use it as:  CHANNEL_NAMES_OVERRIDE = ACTICAP_32
ACTICAP_32 = [
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3",
    "T7", "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1",
    "Oz", "O2", "P4", "P8", "TP10", "CP6", "CP2", "Cz",
    "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2",
]
# Active, because the direct actiCHamp LSL connector does not carry labels.
# Extra channels beyond these 32 (AUX, triggers) keep their original names.
CHANNEL_NAMES_OVERRIDE = ACTICAP_32

# Mains frequency where you are: 60 in North America, 50 most other places.
LINE_FREQ_HZ = 60.0

# --------------------------------------------------------- minimal montage
# Only gel these. Everything else can stay dry in its holder - the amp streams
# all 32 channels regardless, and the analysis drops the ones not listed here.
#
# Why each one is on the list:
#   C3, Cz, C4                     the measurement itself
#   FC1 FC5 CP1 CP5                Laplacian ring around C3
#   FC2 FC6 CP2 CP6                Laplacian ring around C4
#                                  (Cz borrows FC1 FC2 CP1 CP2)
#   Fp1, Fp2                       forehead, no hair, 30 seconds to apply.
#                                  Catch blinks, and drive the artifact
#                                  control that proves you are decoding
#                                  motor cortex and not eye movement.
#   Oz                             alpha blocking on eye closure, which is
#                                  how you know the whole chain works.
# Plus your hardware reference (FCz) and ground (Fpz), which are not channels.
#
# 14 sites instead of 32. Drop Oz and the forehead pair if you really must,
# but you lose the chain check and the artifact control - both of which exist
# to stop you believing a result that is not real.
MINIMAL_MI = [
    "C3", "Cz", "C4",
    "FC1", "FC2", "FC5", "FC6",
    "CP1", "CP2", "CP5", "CP6",
    "Fp1", "Fp2", "Oz",
]
# Set to None to use every channel in the recording.
USE_CHANNELS = MINIMAL_MI

# ------------------------------------------------------------------ montage
# Your hardware reference electrode does not appear in the recording, because
# it would be recorded against itself. If it sat on a position you care about
# (most importantly Cz), name it here and it gets reconstructed: added back as
# a zero channel, then given real values by the average reference below.
# Leave as None if your reference was FCz or anywhere you do not need.
RECOVER_REFERENCE_CHANNEL = None  # e.g. "Cz" or "FCz"

# Offline re-reference. Options:
#   "average"          common average reference
#   ["TP9", "TP10"]    linked mastoids, or any list of channel names
#   None               leave the data as the amp sent it
#
# The direct actiCHamp connector sends reference-free data, so something has
# to be applied here. "average" is the right default.
#
# One caveat worth understanding with MINIMAL_MI: an average over 14 mostly
# central electrodes is not an average of the head, and it carries some of the
# mu rhythm itself, which shrinks the apparent size of the ERD. This does not
# corrupt the results, because the ERD numbers and maps come from the surface
# Laplacian, which cancels any common reference by construction, and the
# decoding uses CSP, which cares about differences between channels rather
# than their absolute level.
REREFERENCE = "average"

# A surface Laplacian sharpens ERD a lot by subtracting the mean of the
# surrounding electrodes, which strips away the widespread activity that a
# single reference shares with every channel. Applied only to the ERD figures
# and table; decoding uses CSP, which does its own spatial filtering.
USE_LAPLACIAN_FOR_ERD = True


# Channels the report focuses on. Case-insensitive; missing ones are skipped.
FOCUS_CHANNELS = ["C3", "Cz", "C4"]
# Sensorimotor group for decoding. Listed generously - whatever your cap does
# not have is silently skipped, so this works for a 32ch as well as a 128ch.
CENTRAL_PICKS = [
    "FC3", "FC1", "FCz", "FC2", "FC4", "FC5", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4", "CP5", "CP6",
]
# T7/T8 are deliberately left out: they sit over temporalis muscle and let jaw
# tension masquerade as a decodable signal.
# Used for the artifact sanity check: if a classifier does well using ONLY
# these, you are probably decoding eye movement or jaw muscle, not motor cortex.
FRONTAL_PICKS = ["Fp1", "Fp2", "AF7", "AF8", "F7", "F8", "F3", "F4", "Fz"]

TFR_FREQS = (4, 36)  # Hz, range for the time-frequency maps
ERD_BASELINE = (-1.5, -0.5)  # seconds relative to cue onset
ERD_WINDOW = (0.5, 3.5)  # seconds relative to cue onset, where ERD is measured
CSP_BAND = (8.0, 30.0)  # Hz, bandpass for the CSP+LDA decoding check
REJECT_PEAK_TO_PEAK = 150e-6  # volts; epochs above this are dropped

DATA_DIR = "data"
