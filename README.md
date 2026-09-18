# StemScope

Intelligent Music Separation and Practice Platform: a portfolio/research project
with separation and benchmarking, exploratory failure analysis, an experimental
model selector, signal-quality diagnostics, practice mode, model comparison and a
**unified dashboard and Phase 11 local API/engineering setup**.
Upload a WAV or MP3 to inspect its audio
features, or choose local Demucs or Open-Unmix and play vocals, drums, bass, and other.
Audio analysis describes the input recording. Benchmarking measures separation quality
only when matching original reference stems are supplied; neither feature improves the stems.

See the [project status and five-minute demo](docs/project-status.md) for current
capabilities, measured limitations and the next research milestone.

## Supabase sign-in

Email/password signup and sign-in are available for the UI and API, with per-user
audio workspaces. Follow [Supabase setup](docs/supabase-auth.md) to connect a project.
Live account verification requires your project URL and publishable key.

## Environment and installation

Use Python **3.11** (supported range 3.10–3.12). The older Demucs release is paired
with matching PyTorch/torchaudio 2.2.2 and NumPy <2. Python 3.13+ is unsupported.
CPU is the default, including on Apple Silicon; CUDA may be selected with a
compatible PyTorch installation. Allow several GB of free RAM and disk space.

```sh
cd /path/to/stemscope
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python app/main.py
```

Alternatively, run `stemscope` after installation. Open the local URL printed by
Gradio (normally http://127.0.0.1:7860). The server binds only to localhost and does
not create a public share link. First inference with each model downloads its own weights;
subsequent jobs reuse the loaded model. Internet access is needed for installation
and the first model download. Model checkpoints use PyTorch's cache; set
`TORCH_HOME` to change it.

SoundFile/libsndfile handles WAV/MP3 decoding and float WAV writing without a
mandatory FFmpeg subprocess. Standard platform wheels include libsndfile; on
platforms without wheels install a recent libsndfile with MP3 support. Gradio may
need FFmpeg for additional media conversions outside this WAV output workflow.

```sh
STEMSCOPE_OUTPUT_DIR=/absolute/path/to/generated STEMSCOPE_DEVICE=cpu stemscope
```

`STEMSCOPE_DEVICE=cuda` enables an available NVIDIA GPU. Output defaults to
`./outputs` relative to the launch directory. Limits are 200 MB, 20 minutes, and
mono/stereo audio; edit `Settings` for other local limits. Long inputs can still
consume significant RAM and CPU time; inference splits the track into chunks,
but decoding and stem tensors are held in memory.

## Architecture

```text
app/main.py                   thin launch entry point
src/stemscope/
  config.py                   storage, device, resource limits
  application.py              composition of the two available separation services
  benchmarking/metrics.py     documented SI-SDR and sample-rate alignment
  benchmarking/resources.py   sampled RSS and process CPU monitoring
  benchmarking/runner.py      repeated runs, failures, CSV and provenance
  benchmarking/cli.py         batch dataset command
  benchmarking/demo.py        explicitly synthetic evaluation smoke data
  errors.py                   user-facing application errors
  audio/validation.py         extension, file, and size checks
  audio/loader.py             bounded decoding and sample validation
  audio/features.py           typed metadata, spectral/energy features, tempo
  audio/plots.py              standalone waveform, spectrogram, feature PNGs
  analysis.py                 analysis validation, timing and chart lifecycle
  separators/base.py          four-stem Separator contract
  separators/demucs.py        lazy model, preprocessing, inference, WAV export
  separators/openunmix.py     lazy umxhq model, contextual chunks, WAV export
  service.py                  job lifecycle, timing, output validation
  ui.py                       Gradio components and presentation only
  utils/timing.py             wall-clock timing
 tests/                       inexpensive tests with mocked inference
 data/                        ignored datasets placeholder
 outputs/                     ignored job folders and Gradio cache
```

The service is the small addition to the proposed layout: callers inject a
`Separator`, so the UI does not depend on Demucs execution details. Only the app
composition selects the adapter. Demucs-specific resampling and normalization
stay inside the adapter. The model is loaded lazily once per process, with a lock
protecting reuse. Gradio serializes requests and bounds its waiting queue.

Each successful separation has a unique `job-*` directory containing four WAVs
and `result.json` with original filename, model identity, processing time, and stem filenames.
Benchmarks retain `benchmark-*/results.csv` and `manifest.json`; synthetic checks
retain a separate `synthetic-check-*` folder containing generated test inputs.
Each analysis has a unique `analysis-*` directory containing three PNG charts.
The Matplotlib cache defaults to `outputs/.matplotlib` (override with `MPLCONFIGDIR`).
Failed jobs are deleted; successful jobs remain so playback links keep working.
**Stop the app, then delete old job folders when no longer needed.** Gradio's
upload/cache files reside under `outputs/.gradio` and expire after 24 hours while
the app runs; these are separate from retained stems. Uploaded originals are not
copied into job directories. Do not point the output folder at unrelated private
files: Gradio is allowed to serve that folder to local clients.

The displayed total time includes validation, decoding, first-use model loading,
inference, WAV export, and output validation, but excludes time waiting in the UI
queue. It is a measurement for that run, not a benchmark. Errors are summarized
in the UI; detailed exceptions appear in terminal logs.

## Phase 2: analyze audio

The **Audio analysis** tab is selected by default. Upload a song and click
**Analyze audio**. You do not need to separate the song first, and analysis does
not load or download model weights. The **Stem separation** tab retains the original flow.
If updating an existing installation, stop the app, run
`python -m pip install -e '.[dev]'` inside its activated environment, and restart it.
The dependency lockfile includes librosa 0.11, SciPy, and Matplotlib.

The summary shows duration, original sample rate, channel count, estimated BPM,
and means of four full-track feature curves. The charts show the original
per-channel waveform, spectrogram, RMS amplitude, spectral centroid, spectral
bandwidth, and zero-crossing rate. Gradio serializes analysis and separation
under one concurrency group to avoid running both heavy operations simultaneously.

Measurement definitions and limits:

- Waveform bins retain minima and maxima from original samples in each channel,
  preserving transients and stereo polarity with at most 2,000 bins.
- Spectral/energy features use a low-pass resampled copy at at most 22,050 Hz.
  This leaves source audio and separator inputs untouched. Higher frequencies
  above 11,025 Hz are outside this analysis; original sample rate remains visible.
- Frames use 2,048 samples, a 512-sample hop, a Hann window for spectra, and
  zero padding at the recording edges. Edge-frame values can be lower.
- RMS is the square root of mean channel mean-square amplitude per frame.
  It is an energy-related amplitude measure, not perceptual loudness or a quality score.
- Spectral features use the square root of mean channel spectral power, so
  opposite-phase stereo does not cancel. Centroid is magnitude-weighted frequency;
  bandwidth is the magnitude-weighted second-moment spread around the centroid.
  ZCR is the average channel fraction of sign changes per sample.
- Summary values are arithmetic means of the full-track frame curves, including
  zero-valued silent frames. These measurements depend on the documented analysis rate.
- STFT work uses blocks of at most 256 frames. The displayed spectrogram retains
  per-frequency maxima in at most 1,200 time bins and uses dB relative to the
  recording's largest spectral magnitude, with an 80 dB range. It is not calibrated
  SPL or absolute dBFS. Silent recordings display the bottom of the color scale.
- Tempo uses averaged channel onset envelopes from the first 120 seconds at most.
  Clips shorter than four seconds, silence, or fewer than three tracked beats
  return **Unavailable**. This is a heuristic rhythm estimate; half/double tempo,
  changing rhythms, and non-percussive music can give misleading values.
- Initial analysis may take longer while librosa/Numba compiles functions and
  Matplotlib creates its font cache. Subsequent requests reuse those caches.
  Decoded/resampled audio still resides in RAM; this is not a streaming decoder.

Manual Phase 2 acceptance: analyze a familiar WAV and MP3, check metadata and the
three charts, then analyze silence or a short clip and confirm tempo is unavailable.
Corrupt files should clear the prior analysis and show a friendly error. Retained
analysis charts can be deleted alongside old separation jobs after stopping the app.

Feature definitions follow the [librosa spectral feature documentation](https://librosa.org/doc/0.11.0/feature.html)
and [beat tracking documentation](https://librosa.org/doc/0.11.0/generated/librosa.beat.beat_track.html).

## Phase 3: choose a separator

Open **Stem separation**, choose **Demucs (htdemucs)** or **Open-Unmix (umxhq)**,
then click **Separate stems**. Demucs remains the default. The **Model used for
these stems** field identifies the completed output even if you subsequently
change the dropdown. Rerunning replaces the players; previous files remain in
separate job folders with their model metadata. There is no automatic selector
or side-by-side comparison dashboard yet.

Both adapters implement `Separator.separate(samples, sample_rate, output_dir)`
and return the same four named WAV paths. `SeparationService` performs the same
validation, timing, cleanup and output checks for either adapter. The application
composition provides a mapping of model names to services; Gradio routes to the
chosen service without model inference logic.

Open-Unmix 1.3.0 is now an explicit dependency (it was already installed transitively
by Demucs). Its `umxhq` checkpoint uses four pretrained target networks at 44,100 Hz
with stereo input and one Wiener refinement iteration. The adapter resamples once,
duplicates mono to stereo, and preserves output gains using float WAVs. It lazily
loads and reuses its model, independently of Demucs; download or inference failures
produce a friendly error and never silently switch models.

Open-Unmix processes 15-second cores with up to one second of surrounding context
on each side, cropping each estimate to its core and streaming it into the WAV.
Very short inputs are padded for the model FFT, then cropped to the true length.
This bounds model working tensors but still holds decoded/resampled input in RAM.
Context cropping preserves length and avoids gaps/duplication; the recurrent model
has limited context at each boundary, so results can differ from full-track inference
and audible boundary artifacts remain possible. Chunking is not a quality improvement.

Each model needs its own first-use weight download into `TORCH_HOME`/PyTorch's cache.
Both stay loaded once used; memory use can increase after switching models.
Open-Unmix is an alternative, not a promise of cleaner stems. No quality measurements
or invented benchmark comparisons are reported in Phase 3.

Manual check: use the same short music excerpt with each model, confirm the model
label and all four playable stems, and listen for different leakage/artifacts.
You can inspect each job's `result.json` to identify saved outputs later.

Integration follows the [Open-Unmix inference API](https://sigsep.github.io/open-unmix-pytorch/predict.html)
and the locally installed 1.3.0 implementation. Phase 4 adds benchmarking below; the model-choice control itself makes no accuracy claim.

## Phase 4: benchmark and evaluate

Restart the app and open **Benchmark**. There are two useful paths:

1. **Run synthetic pipeline check** needs no uploads. It creates tones/noise with
   known components and runs both real models. The labels are placeholders, not
   real vocals/instruments. This checks evaluation plumbing; these results must
   never be presented as music-separation accuracy.
2. **Benchmark both models** uses the mix uploaded at the top of the app. With no
   reference files it reports performance only. For quality scores, supply all
   four separately recorded original WAV stems and describe their source/split.
   Generated Demucs/Open-Unmix stems are not ground truth.

Download **results CSV** and **protocol and input hashes**. The CSV contains one
row per model, track, repetition and stem: SI-SDR, SI-SDR improvement over the
original mixture, status/errors, separation-call wall time, real-time factor
(seconds divided by track duration), process CPU seconds/percent, sampled RSS,
and output paths. Time and resource values repeat across the four stem rows;
**do not add these rows together**. Positive SI-SDR improvement means the output
beat the unseparated-mixture baseline under this metric. It is not percent accuracy.

The manifest records timestamp, platform/Python/package versions, seed, model
parameters and devices, SHA-256 hashes of every input, user-supplied provenance,
and the evaluation/resource protocol. These records support rerunning the same
experiment; determinism across different hardware/library versions is not guaranteed.

### Reference requirements and metric protocol

- Use a WAV mixture and all four original WAV references with exactly the same
  sample rate, channels and frame count. They must also begin at the same time.
  The program checks dimensions, but cannot certify authenticity or detect every
  wrongly aligned recording. Do not use separately trimmed or MP3-encoded references.
- Scores are full-track **zero-mean SI-SDR**, not BSS Eval SDR, perceptual quality,
  or the windowed SDR values reported on some music benchmarks. Each channel's DC
  mean is removed; stereo is scored jointly with one gain factor, preserving its
  relative channel balance. SI-SDR ignores global gain differences.
- The estimate is resampled to the reference rate using a polyphase filter.
  A mono reference is duplicated if the separator outputs stereo. At most one
  sample of resampling-rounding mismatch is trimmed; larger duration differences
  are errors. No delay search, arbitrary cropping, or channel cancellation is used.
- For reference `s` and estimate `x`, project `x` onto `s` to obtain target `t`;
  the score is `10 log10(sum(t²) / sum((x-t)²))`. This follows
  [Le Roux et al., SDR — half-baked or well done?](https://arxiv.org/abs/1811.02508).
- Silence (mean-square energy below 1e-20) has no numeric score. Mathematical
  positive/negative infinity is represented by a blank CSV value and an explicit
  status (`perfect_positive_infinity` / `zero_projection_negative_infinity`).
  Numerically negligible projection/residual energies use float64 epsilon.
  `silent_reference`, `silent_estimate`, `no_reference`, and `not_scored` are also
  explicit. Never interpret missing scores as zero or average them as numbers.
- Quality evaluation errors and separation failures are retained as rows. Successful
  models still finish; preflight rejects a malformed reference set before inference.
  CSV is saved after each model run, so completed results survive later interruptions.

### Performance protocol

Models run sequentially and stay cached. There is **no warmup or reset**; an initial
run may include weight downloads/loading, and later repetitions may be faster.
The UI may already have loaded a model; the benchmark does not assert cold-start
status. For comparisons, start a fresh process with weights already cached, use
multiple repetitions, and distinguish the first repetition from later ones.
Timing covers the separation service (decode, model execution, WAV export and output
checks); reference scoring, hashes and report export are excluded. Seed 0 is set
before each model call. The model ordering is recorded and fixed.

`psutil` samples **the entire process's RSS every 50 ms**, including cached models,
Python libraries and UI work; transient peaks can be missed. The baseline RSS is
also exported. This is not isolated model memory or allocator memory. CPU percent
is CPU seconds / wall seconds ×100 and can exceed 100 on multicore machines.
GPU utilization is left unavailable, not reported as zero. Current CPU measurements
are useful observations, not controlled hardware benchmark claims.

### Batch CLI and real evaluation data

Arrange aligned recordings like this (only folders containing `mixture.wav` are read):

```text
data/evaluation/
  track-a/
    mixture.wav
    vocals.wav
    drums.wav
    bass.wav
    other.wav
  track-b/
    ...
```

References can all be omitted for performance-only evaluation. A partial set fails.
Use the installed environment:

```sh
python -m stemscope.benchmarking.cli data/evaluation \
  --output outputs --repeats 3 \
  --provenance "Dataset name, version, held-out test split, and source"
```

The installed `stemscope-benchmark` entry point runs the same command. Exit code
0 means all requested runs completed, 1 means some saved rows failed, and 2 means
setup/validation failed. Paths are configurable and generated data remains ignored.

For actual evidence, use your own aligned multitracks or an appropriate held-out
music dataset such as [MUSDB18 / MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html).
Respect its access terms and keep datasets/checkpoints out of Git. A model trained
on a dataset's training split must not be evaluated on that split as unseen data.
State dataset/source, split, number of tracks, exclusions and metric protocol in
any reported result. Use several diverse tracks, summarize each stem across tracks,
report failures/missing scores and variation, and pair scores with listening checks.
No dataset is downloaded at app launch. The prepared public pilot described below
provides a small real-reference evaluation; it does not establish general musical accuracy.

## Checks

```sh
ruff check .
ruff format --check .
pytest -q
python -c "from stemscope.ui import build_app; from stemscope.separators.demucs import DemucsSeparator"
```

Unit tests use a fake separator and mocked Demucs inference, with real audio I/O.
Analysis tests use known tones, silence, opposite-phase stereo, and a click track
to check the meaning of extracted values, plus short inputs and rendering failures.
They do not download weights or measure separation quality. Dependency versions
are constrained in `pyproject.toml`; `uv.lock`, when present, supports exact
resolution using `uv sync --extra dev --locked`.

Manual acceptance:

1. Launch the app, upload a short WAV, and choose **Stem separation**; confirm filename, Processing status,
   completion time, and four playable stems. Check the resulting job directory.
2. Repeat with an MP3, a mono track, and the same filename twice; verify unique
   directories and reasonable playback length.
3. Upload corrupt audio renamed `.wav`; confirm an actionable message, cleared
   prior players, and a useful terminal log. Try submitting without a file.
4. On first use, allow the model download. Listen to a real song to confirm
   plausible vocal/instrument separation. Synthetic smoke audio cannot establish
   music separation quality.

## Current limitations and roadmap

Phase 3 offers manual selection between htdemucs and umxhq. There is no stem mixer, speed/pitch control,
looping, reference-free quality estimator, comparison dashboard, or intelligent selector yet.
Float WAV output preserves relative gains, but browser playback may clip peaks
above full scale. Local model artifacts and dependency downloads are required;
CPU runtime depends strongly on hardware and song length. The service is for a
trusted local user, not a hardened public upload endpoint.

Later phases: failure
analysis; a selector trained only after enough benchmark data exists; interpretable
quality estimation; practice controls; model comparison; technical dashboard;
and finally API/deployment engineering. Phase 5 and beyond are not implemented.

Implementation references: [Demucs 4.0.1 inference](https://github.com/facebookresearch/demucs/blob/v4.0.1/demucs/separate.py),
[Gradio file access](https://www.gradio.app/guides/file-access).

## Phase 1 verification performed

On this Apple Silicon workspace with isolated Python 3.11.16:

- Ruff lint and formatting checks passed.
- All 22 pytest cases passed, including actual WAV/MP3 decoding.
- Application entry point imported successfully and local HTTP startup returned 200.
- Real htdemucs inference produced four validated stems from a one-second synthetic WAV.
- A Gradio client uploaded that WAV, received Complete status, and downloaded all
  four audio outputs. The empty-upload UI request returned the expected friendly error.

The temporary test server was stopped afterward. These checks establish execution
and file handling, not musical quality or performance on full songs. Generated
smoke-test audio and downloaded model weights are outside the project deliverable.

## Phase 2 verification performed

- All 43 pytest cases passed; lint, formatting, application import and lockfile
  consistency checks passed.
- Known-tone frequency/energy, regular-click tempo, silent/short inputs,
  anti-phase stereo and FFT block-boundary checks passed.
- Gradio client WAV and MP3 uploads returned metadata and three downloadable
  charts; charts were visually inspected. Empty and corrupt uploads returned
  friendly errors and cleared previous chart outputs.
- A real Demucs request through the updated UI still produced four downloadable
  stems. The temporary verification server on port 7861 was stopped afterward.

These checks do not measure musical separation quality. Restart any existing
app process to load the Phase 2 code and see the new tab.

## Phase 3 verification performed

- All 57 tests passed; Ruff lint/format checks, entry-point import, and dependency
  lock consistency passed.
- Mocked adapter tests verify lazy loading, model reuse, mono/stereo handling,
  resampling, short-input padding, exact chunk concatenation, missing dependencies,
  invalid/missing outputs, failure cleanup, model routing, and saved model identity.
- Real Gradio requests for both Demucs and Open-Unmix returned four downloadable
  stems and the correct model label. The Open-Unmix input was a 16.25-second,
  8 kHz mono synthetic clip, exercising resampling and a contextual chunk boundary.
- Audio analysis still returned its metadata and three charts through Gradio.

No musical-quality or comparative benchmark claim follows from these smoke tests.
The verification server used port 7861 and was stopped after testing. Restart your
usual local app to see the Phase 3 model selector.

## Initial Phase 4 implementation verification (historical)

- All 75 pytest cases passed. Ruff lint/format checks, application/CLI imports,
  and dependency-lock consistency checks passed.
- Analytic SI-SDR checks include a known 20 dB signal, gain and DC invariance,
  silence, perfect reconstruction, orthogonal estimates, stereo balance and
  alignment rejection. A copy-mixture separator gives zero baseline improvement.
- Runner tests cover repeated identical inputs, input hashes, missing/partial
  references, renamed MP3 rejection, failed-model retention and CSV export.
- Real Gradio synthetic-check requests ran both pretrained models and returned
  eight scored rows plus downloadable CSV/manifest. Performance-only requests
  left quality fields empty; empty uploads returned a friendly error.
- The batch CLI also completed both real models on the generated synthetic set.
  This verifies the workflow, not accuracy on real music. No genuine musical
  reference corpus was evaluated in that initial milestone; the completed study below adds it.
- The temporary verification server on port 7861 was stopped after testing.

## Prepared real-music pilot

A real-reference pilot is now available locally. Restart the app, open **Benchmark**,
and click **Run prepared real-music pilot**. No file uploads are needed. The table
now includes the track name. This action evaluates both models once per excerpt;
CSV and protocol downloads remain available.

The pilot uses the only two CC BY-NC-SA-listed tracks in the official MUSDB18 test
preview archive under an alphabetical, pre-inference selection rule (requested
maximum five): **The Easton Ellises (Baumi) - SDRNR** and **The Easton Ellises - Falcon 69**.
They are short activity-selected excerpts by the same artist, not a diverse sample.
They were downloaded from the public [SiSEC18-MUS 7s Excerpts record](https://zenodo.org/records/1256064)
(version 1.0.0). The official [track metadata](https://github.com/sigsep/website/blob/master/content/datasets/assets/tracklist.csv)
lists CC BY-NC-SA 3.0 for both. Retain attribution and license terms; audio remains
in ignored `data/` and generated stems in ignored `outputs/`.

`STEMSCOPE_PILOT_DIR` overrides the default `data/musdb7-pilot` folder. The loader
checks saved SHA-256 hashes before evaluation and rejects changed/missing files.
The prepared files are original reference WAVs, not separator outputs or synthetic audio.

To prepare this pilot on another machine, from the project root:

```sh
python scripts/prepare_public_pilot.py --output data/musdb7-pilot
```

This explicit network command retrieves only selected ZIP members with HTTP ranges,
verifies ZIP CRCs, saves SHA-256 hashes, and copies the source/license metadata. It
refuses to overwrite an existing output directory. The `--limit` argument may
restrict the selection further; it never includes training or restricted-license tracks.

The completed three-repetition evaluation is saved under
`outputs/real-music-pilot/benchmark-adbnjhit/`: `report.md`, `results.csv`, and
`manifest.json`. It contains 48 result rows with no failed or unavailable scores.
Repetitions are collapsed before comparing quality, so there are still only two
independent excerpts. Demucs has the higher SI-SDR in seven of eight track/stem
pairs; Open-Unmix has the higher vocals score on Falcon 69. These claims apply only
to this pilot and metric protocol, not overall model accuracy. No listening
assessment or full Phase 5 failure analysis has been performed.

Recreate a report for a rerun of this same two-track pilot:

```sh
python scripts/summarize_public_pilot.py outputs/real-music-pilot/benchmark-RUN_ID --output outputs/real-music-pilot/report.md
```

The report script is intentionally scoped to the named two-track pilot. Its
selection/license narrative must not be reused for an unrelated dataset.

Public-pilot validation: all 79 tests passed, including dataset selection and input
integrity checks. Lint/format checks passed. The new Gradio action completed both
real models on both prepared tracks, returning 16 correctly labeled scored rows
and downloadable CSV/manifest files. The temporary test server was stopped.

## Completed Phase 4 research study

The two-track pilot above is retained as a quick demonstration. The broader study
uses **all 50 official MUSDB18 test previews**, with original reference stems,
both models, and two measured repetitions after one excluded warm-up per model.
These are seven-second activity-selected excerpts, not full songs. The original
source restrictions are retained; this research download is not a CC-only corpus
and does not grant permission to redistribute its audio.

In the app, open **Benchmark → Load completed Phase 4 results**. This reads the
saved report, chart and CSV without running the models again. The study must be
under `STEMSCOPE_OUTPUT_DIR/phase4-study` (default `outputs/phase4-study`).
Restart an already running app to see the new button.

Reproduce from the project directory using the activated environment:

```sh
python scripts/prepare_research_testset.py --output data/musdb7-test
python -m stemscope.benchmarking.study data/musdb7-test --output outputs/phase4-study --repeats 2 --threads 4
python -m stemscope.benchmarking.summary outputs/phase4-study/study-RUN_ID
```

The downloader saves the selection before scoring, verifies ZIP CRCs and file
hashes, and resumes verified partial downloads. Each model runs sequentially in
a fresh CPU process with four compute threads and one inter-op thread. One
excluded warm-up removes model loading from measured calls. Timing includes
input decoding, inference, WAV export and validation; scoring is excluded.
The machine is not exclusively reserved and CPU frequency/affinity are not fixed.
GPU usage is unavailable in this CPU study. Process RSS is sampled, not an exact
allocation or guaranteed maximum.

`benchmarking/study.py` manages the isolated workers. `benchmarking/summary.py`
validates coverage and matching input hashes, collapses repeated quality scores
within each track, and exports per-stem distributions, paired comparisons,
call-level runtime summaries and every failed/unavailable-score row. Repetitions
do not increase the number of independent tracks, and the four repeated per-stem
time fields are never added together. `benchmarking/results.py` loads saved
artifacts for the UI. The JSON protocol includes model settings, package versions,
hardware information, warm-up details and reference hashes.

This completes the excerpt-based Phase 4 evaluation. Full-song evaluation, GPU
profiling and listening studies remain extensions. Phase 5 condition-based failure
analysis and Phase 6 selector training have not been implemented.


Completed local run: [`study-ko5nfljz/report.md`](outputs/phase4-study/study-ko5nfljz/report.md).
All 200 measured calls and 800 stem observations succeeded with available scores.
Demucs scored higher in 190/200 paired track/stem comparisons. Median call times
were 5.94 seconds for Demucs and 0.55 seconds for Open-Unmix on this machine.
An Open-Unmix call took 20.43 seconds; it remains included and the report shows
maximum timings. These results apply to the specified excerpts and SI-SDR protocol.

Final validation: **95 tests passed**, Ruff lint and formatting checks passed,
and a real local Gradio request loaded all eight 50-track summaries plus the
chart, report and raw CSV downloads. The temporary verification server was stopped.

## Phase 5 — exploratory failure analysis

Phase 5 now analyzes the completed reference benchmark without rerunning models.
From the project directory:

```sh
python -m stemscope.failure_analysis outputs/phase4-study/study-ko5nfljz --output outputs/phase5-analysis
```

Restart the app. Open **Failure analysis → Load Phase 5 analysis**, select an
example, then click **Load listening example**. Four players show the mix,
original target reference, Demucs estimate and Open-Unmix estimate. Listen for
leakage, missing target sound and watery/muffled artifacts. Displayed scores are
medians across repetitions; playback uses the earliest successful repetition.
Audio is checked against its saved score and hashed during analysis; playback
rejects changed files and copies the selected evidence into controlled output
storage. This workflow performs no model inference or training.

The analysis exports to `outputs/phase5-analysis` (use `--output` to change it;
the app reads `STEMSCOPE_OUTPUT_DIR/phase5-analysis`):

- `features.csv`: 50 rows of mixture features plus original-reference energy/activity measures.
- `conditions.csv`: model/stem distributions by source activity, vocal/drum energy share and genre, including group sizes and negative-improvement counts.
- `correlations.csv`: all 56 exploratory feature/model/stem Spearman correlations.
- `cases.csv`: bottom-five scores per model/stem plus additional negative-improvement cases.
- `analysis.json`: source hashes, inclusive quartile thresholds, group membership and playback hashes.
- `report.md`: protocol, condition tables and limitations.

The completed analysis has 88 condition summaries and 42 review cases. There are
14 negative-improvement track/model/stem results among 400 unique outcomes.
For the 13 excerpts in the lowest vocal-energy-share quartile, median vocal
SI-SDR was 6.15 dB for Demucs and 1.64 dB for Open-Unmix, versus 8.08 and 5.96 dB
across all 50 excerpts. These overlapping groups describe associations, not causes.

Energy shares use mean-square reference stem energies divided by their sum, not
mixture energy; source cross terms are excluded. Activity counts how many of the
four source classes exceed -30 dB of their own peak energy in 100 ms blocks.
It is not an instrument count or validated musical-density label. Inclusive
quartile ties can create overlapping high/low groups. Groups with fewer than five
scored tracks are flagged in the CSV. Correlations have no significance claims.

Reverb, backing vocals, distortion and acoustic instrumentation still require
listening or reliable annotations. No human listening conclusions are claimed.
These reference-assisted descriptors must not be used as inference-time features
for a selector. Because this test set has now been explored, future Phase 6 work
needs separate training/validation data and an untouched evaluation set.

Validation: 100 tests pass, including stereo energy preservation, silence,
quartile ties/missing values, changed-estimate rejection and playback integrity.
The live Gradio verification returned all 88 condition rows, 42 selectable cases,
and four playable audio files for a selected case. Lint/format checks passed;
the temporary verification server was stopped.

## Phase 6 — experimental model selector

The selector uses **mixture-only features** and learns the difference in mean
four-stem SI-SDR between Demucs and Open-Unmix. A fixed small random forest is
compared with always choosing either model. Each preference falls back to a
training-selected constant if validation does not show at least 0.1 utility gain.
Fallbacks are explicitly labeled; they are not presented as intelligent predictions.

The original 50-track test corpus has already been explored in Phase 5 and is
excluded from selector development. The public archive has **94 available train
mixtures**, not all 100 full training songs. Two spelling differences between ZIP
members and metadata are recorded as aliases. This remains restricted local
research data; do not redistribute audio or generated stems.

Reproduce from the project directory, in order:

```sh
python scripts/prepare_research_testset.py --split train --workers 4 --output data/musdb7-train
python -m stemscope.selection.plan data/musdb7-train --output outputs/phase6-selector
python -m stemscope.benchmarking.study data/musdb7-train --output outputs/selector-benchmark --repeats 1 --threads 4
python -m stemscope.selection.train outputs/selector-benchmark/study-RUN_ID --output outputs/phase6-selector
```

The plan freezes name-derived artist groups, the train/validation/test assignment,
forest settings, benchmark settings and preference penalties before scoring. Each
artist group stays in one partition. Approximately 60/20/20% of groups are assigned
to train/validation/test; track counts may differ. Artist identity is inferred,
not independently verified. The separator checkpoints may have seen official
training tracks: this holdout evaluates the **selector**, not unseen-song separator
performance. Hyperparameters are fixed, with no test-driven tuning or refitting.

Each model gets one excluded warm-up and one measured CPU call per excerpt. This
provides quality labels and observed runtime; unlike Phase 4, the selector labeling
run does not estimate within-track timing variation from repetitions. Failed or
unavailable labels are retained in the benchmark and listed as exclusions before
selector fitting. All four stems must have available scores for a training example.

Restart the app and select an **Auto** option under **Stem separation**:

- **Auto: Quality:** maximize predicted mean four-stem SI-SDR.
- **Auto: Balanced:** subtract 0.5 dB per second of benchmark CPU time.
- **Auto: Speed:** subtract 2 dB per second of benchmark CPU time.

These are engineering preferences, not calibrated perceptual scores. Choices
use training-only median per-model runtime costs. Evaluation uses observed held-out
runtime and includes outliers. Costs describe seven-second clips on this machine
with four CPU threads; they are not live predictions of full-song processing time.
Automatic-mode elapsed time includes feature extraction, which is excluded from
the benchmark utility. The first seven seconds supply features when a learned
policy is active; later changes in instrumentation are not modeled. A fallback
skips unnecessary feature extraction.

`selection/features.py` defines the same 11 features for training and prediction.
`selection/plan.py` freezes the experiment. `selection/train.py` fits training-only
imputation and the forest, gates policies on validation, then evaluates the frozen
policy on held-out groups. `selection/inference.py` evaluates plain JSON trees;
it does not load pickle/joblib objects. Exported predictions are checked against
scikit-learn. Reference energies, artist/genre labels and filenames are not inputs.

Artifacts are in `outputs/phase6-selector`: `plan.json`, `frozen-policy.json`,
`selector.json`, `features.csv`, `evaluation.csv`, `predictions.csv`,
`exclusions.json`, and `report.md`. The app reads
`STEMSCOPE_OUTPUT_DIR/phase6-selector/selector.json`. A missing artifact returns a
clear error and leaves manual model choices available. The benchmark inputs,
model settings and versions remain in the study manifest; the selector records
its input/protocol hashes and package versions. Regret and oracle agreement are
selection metrics, not percentages of audio accuracy. Phase 7 is not implemented.


Completed Phase 6 run: 94 previews, 188 measured calls, 752 stem observations,
no excluded tracks; 55 selector-training tracks, 16 validation tracks and 23
held-out tracks. The frozen plan is saved with the run. Quality uses the Demucs
fallback; Speed uses the Open-Unmix fallback. Balanced enables the learned
candidate after a validation utility gain of 0.142, but **did not beat the fixed
Open-Unmix baseline on the selector holdout** (mean utility 6.367 versus 6.401).
This is a completed experiment with a negative holdout result, not a demonstrated
accuracy improvement. The validation-selected policy was not changed after seeing
test results. Manual choices remain the default.

Open **Model selector → Load selector evaluation** for the saved comparisons and
report. Auto modes are restricted to CPU; other devices retain manual choices.
All 110 tests pass, including partition integrity, JSON-forest equivalence,
held-out-label isolation and device calibration checks. Lint and format checks pass.
Live Gradio verification loaded all 18 held-out comparisons and report/CSV downloads.
Both Auto: Speed (fallback) and Auto: Balanced (learned policy) selected a registered
separator and returned four playable stems. Temporary verification servers were stopped.
The published selector is byte-identical to the policy frozen before holdout evaluation.

## Phase 7 — interpretable signal-quality diagnostics

A reference-free **diagnostic prototype** is now available. After separating a
song, click **Inspect these stems** in **Stem separation**, keeping the same
uploaded mix. Four rows show evidence and review flags; the JSON download records
measurements, scope, thresholds and SHA-256 hashes of all five input files.
This runs separately from separation and does not change benchmark timings.

The checks inspect up to the first 30 seconds, at no more than 22,050 Hz:

- Very quiet/absent target: RMS below -60 dBFS; silence may be legitimate.
- Playback headroom: at least 0.1% of original-rate samples reach/exceed full scale;
  float-WAV overs do not prove clipping distortion.
- Shared waveform: absolute zero-lag, channel-aware correlation of at least 0.9
  with another stem; this suggests review, not confirmed instrument leakage.
- Mix consistency: RMS of mix minus summed stems divided by mix RMS above 0.1;
  mixture-consistent leakage can evade reconstruction checks.

High-band power fraction above 4 kHz and spectral flatness are descriptive values,
not muffling/watery-artifact classifiers. Low high-frequency content is normal for
bass. No HIGH/MEDIUM/LOW quality rating or probability is displayed. **No configured
flags does not mean high-quality separation.** Source annotations, shared rhythm
and legitimate silence can confound these checks. Full duration is checked before
prefix analysis; only one resampling-rounding sample is trimmed, with no delay
search. Mono may be duplicated for a compatible stereo comparison.

The fixed thresholds were audited against 400 first-repetition stem outputs from
the already-explored Phase 4 corpus, with input hashes checked and reference
SI-SDR recomputed. Four outputs were flagged. Of 14 negative-improvement outputs,
**four were detected and ten missed**. These rules therefore are not a reliable
general quality estimator. This audit target is worse-than-mix SI-SDR, not human
artifact labels, and this corpus is not independent perceptual-validation data.
Thresholds were not tuned after seeing the audit. Listening annotations and new
validation data are needed before calibrated quality labels are justified.

Reproduce the audit from the project directory:

```sh
python -m stemscope.quality.audit outputs/phase4-study/study-ko5nfljz --output outputs/phase7-audit
```

`quality/diagnostics.py` implements bounded decoding, rate/channel alignment and
signal measurements; `quality/service.py` saves evidence; `quality/audit.py`
compares the fixed diagnostics with reference results. The audit folder contains
`report.md`, `protocol.json`, `observations.csv`, `summary.csv`, `correlations.csv`
and `audit.json`. Per-upload diagnostics are stored in unique `outputs/quality-*`
folders. All 32 feature/model/stem correlations are exported without significance
claims. The report retains missed problems and original evidence hashes.

Validation: 119 tests pass. New tests cover silence, stereo polarity, headroom,
non-finite/misaligned inputs, sample-rate conversion, bounded long-audio analysis
and a deliberately mixture-consistent leakage example that defeats reconstruction.
Live Gradio verification returned four diagnostic rows, the inspected duration,
and downloadable JSON containing five input fingerprints. Lint/format checks
passed, and the temporary verification server was stopped.

## Phase 8 — practice mode

Restart the app and separate a song as usual. Open **Practice**, choose the stems
and section you want, then click **Render practice mix**. Play the resulting
synchronized audio file; changing settings takes effect when you render again.
This reuses existing stems and does not rerun separation.

- **Mute:** removes the selected stems.
- **Solo:** includes only selected stems; an empty Solo list includes all unmuted
  stems. Mute wins if the same stem is also soloed.
- **Volume:** 0–150% independently for each stem.
- **Section:** original-song start/end times, before speed changes. End=0 selects
  up to the next 120 seconds or the track end. An end beyond the track is clamped.
  Explicit sections must be 0.1–120 seconds long.
- **Speed:** 0.5–1.5×, preserving pitch. A 4-second section at 0.8× lasts 5 seconds.
- **Pitch:** ±12 semitones, independently of the selected speed.
- **Loop:** repeats the rendered section in the player; it can be toggled without
  rerendering. Browser playback is not guaranteed sample-gapless.

For bass practice, mute **bass**, set Speed to **0.8**, choose your section and
render. For focused listening, solo a stem instead. Original stems stay unchanged.
The result and settings can both be downloaded.

`practice.py` handles options, strict stem alignment, bounded section decoding,
mixing, pitch/time processing and export. `ui.py` contains the controls and routes
requests through the same serialized audio-processing queue. All active stems are
mixed before transformation, so playback does not rely on synchronizing four
independent browser players. The first/last 5 ms are faded to soften edge clicks.
If necessary, one shared gain limits sample peaks to 0.98 and preserves stem and
stereo balance. This is sample-peak headroom, not a true-peak limiter.

Each successful render saves `outputs/practice-*/practice.wav` (16-bit PCM) and
`practice.json`: effective gains, mute/solo choices, exact source frame range,
speed/pitch, output length, shared headroom gain, edge-fade duration, library
versions, render time and SHA-256 hashes of the decoded source segments. Segment
hashes are little-endian float32, frames × channels in C order. Unique directories
prevent collisions, and failed render jobs are cleaned up.

Speed and pitch transformations can introduce audible artifacts. Rendering is
limited to 120 source seconds to bound memory and processing cost; choose another
section to practice later parts of a long song. Practice mode does not improve
separator quality, stretch stems live while playing, or promise seamless looping.


Practice validation: **134 tests passed**, including mute/solo precedence, gain
mixing, source-section alignment, independent speed/pitch controls, sample-peak
headroom, source preservation, unique exports and invalid/blank options.
A live Gradio request rendered a real four-second excerpt with vocals muted at
0.8× and -2 semitones into a five-second stereo WAV. Audio/settings downloads,
loop toggling and all-muted error handling passed. The temporary server was stopped.
Lint and formatting checks passed.

## Phase 9 — model comparison

Upload a song, open **Compare models**, and click **Compare both models**.
The app runs Demucs and Open-Unmix on one captured copy of the upload. Select
vocals, drums, bass or other to switch the two players and chart without rerunning
either model. Play one model at a time; levels are preserved, not loudness-matched,
so louder output can bias listening.

The tab shows each model's playable stem, processing time, signal diagnostic
status, RMS and strongest cross-stem waveform similarity. It also shows the mix
and both selected stems as spectrograms with one frequency range and a fixed
-100 to 0 dB STFT-amplitude scale. Panels are not independently normalized; the
color reference is amplitude one, not a calibrated perceptual loudness scale.
Charts and diagnostics inspect up to the first 30 seconds; players contain the
full separated track. Spectral differences do not determine which model is better.

The comparison uses sequential interactive service calls with shared model caches.
First-use time may include downloads/loading, and successful timing values are
repeated across the model's four stem rows. Do not sum those rows. Use Phase 4 for
controlled benchmark timing. No reference SDR/SI-SDR is invented for uploads;
the Phase 7 diagnostic limitations still apply.

The comparison.py module owns input capture, common-adapter execution, partial-result
retention and shared-scale plotting. Each run saves a comparison.json manifest,
captured input and on-demand spectrograms in outputs/comparison-*.
The manifest records model identities, adapter settings, devices, separation
times, per-stem diagnostics, errors and input/output hashes. Audio remains in the
separation job directories. Changing the original upload cannot change the captured
comparison, and changed captured audio/outputs are rejected when switching stems.
If one separator fails, the other model's audio and its evidence remain available;
a diagnostic failure is recorded separately from a separation failure.

Comparison storage is local and ignored by Git. Keep its referenced separation
jobs while using a saved comparison. The two players are intended for sequential
listening, not synchronized switching at an identical playback position.


Comparison validation: **140 tests passed**. Tests cover identical captured input,
no repeated inference on stem changes, partial-model failures, separate diagnostic
errors, evidence-integrity rejection and absolute spectrogram level differences.
A live Gradio request ran both pretrained models and returned two playable stems,
eight evidence rows, the shared-scale figure and a manifest. Switching to bass
created no new separation jobs. The generated chart was visually checked,
lint/format checks passed, and the temporary verification server was stopped.

## Phase 10 — unified analysis dashboard

Upload a song, open **Dashboard**, choose a separator and click **Build dashboard**.
The dashboard captures one copy of the upload and brings together:

- Original filename, duration, sample rate, channels and extracted audio features.
- The requested/selected model and the selector's explanation or fallback reason.
- Input-analysis time, individual model times and total dashboard processing time.
- Four playable stems from the selected model.
- Per-stem diagnostic evidence, with the existing Phase 7 limitations.
- Input waveform, spectrogram and feature curves in an expandable panel.
- Optional two-model results and a shared-scale vocals comparison.
- Historical reference-benchmark context, clearly labeled as different recordings.
- A downloadable JSON manifest containing the evidence and source fingerprint.

By default only the chosen model runs. Check **Include both models** to run both;
the selected result is reused from that comparison rather than separated twice.
Use **Compare models** for other side-by-side stem views. Model/selection failures
leave available input analysis and successful model evidence visible. A missing
historical study is marked unavailable and does not block current-file processing.
No historical median is substituted for the upload's quality score.

The dashboard.py module composes the existing services and formats their recorded
evidence. Each run saves a captured input and dashboard.json under a unique
outputs/dashboard-* directory. The manifest links its analysis charts, model jobs
and optional comparison manifest, records input/output hashes through those jobs,
and snapshots the historical table with its source CSV hash. Retain those linked
job folders while using the dashboard. It does not use results from unrelated UI
tabs or automatically reuse a previous upload.

Total time is measured up to final manifest export. Model times are interactive
calls and may include loading; input analysis, selection, diagnostics and chart
work are separate overhead. These are not controlled benchmark timings.
Input feature charts cover the full uploaded audio, with the Phase 2 tempo limit;
diagnostics and the optional comparison chart inspect at most the first 30 seconds.
The uploaded song has no reference-based SI-SDR unless separately benchmarked with
original reference stems. Diagnostic flags are review aids, not quality ratings.

Dashboard validation: **146 tests passed**. Coverage includes selected-model-only
execution, reuse of both-model results, partial failures, missing selector artifacts
and separation of historical scores from upload metrics. A live Gradio request
ran both pretrained models and returned 12 summary rows, two model results, four
playable selected stems, four diagnostic rows, input/comparison charts, eight
explicitly historical benchmark rows and a JSON manifest. Lint and format checks
passed, and the temporary verification server was stopped. These checks establish
software behavior, not separation accuracy on new music.

## Phase 11 — engineering interfaces

The local Gradio workflow is unchanged. A separate FastAPI application now exposes
separation and stem downloads through the same model adapters and validation.
Run these commands from the project directory:

```sh
uv sync --frozen --extra dev
uv run --frozen stemscope-api
```

Open http://127.0.0.1:8000/docs for interactive upload/API documentation.
`GET /health` checks the server (it does not load model weights), `GET /models`
lists model names, and `POST /separations` accepts multipart fields `audio` and
`model`. A successful response contains four download URLs and processing time.
For example, in another terminal:

```sh
curl -X POST http://127.0.0.1:8000/separations \
  -F 'audio=@/absolute/path/song.wav' \
  -F 'model=Open-Unmix (umxhq)'
```

Use the returned stem URLs to download WAV files. The API allows one active
separation per process; overlapping requests receive 409 and can be retried.
Invalid audio/model requests return 400, oversized audio returns 413 and missing
stems return 404. Unexpected failures return a short 500 message with detail in
server logs. Captured uploads are removed; completed stems remain available across
server restarts. This is a synchronous, local API without authentication. Run one
worker and keep it bound to localhost. Research, dashboard and practice workflows
remain available through Gradio; they are not yet separate REST endpoints.

### Configuration and logs

Copy `settings.example.json` to a local configuration file, then set
`STEMSCOPE_CONFIG` to its path. Defaults are overridden by that JSON file, then by
`STEMSCOPE_OUTPUT_DIR`, `STEMSCOPE_DEVICE`, `STEMSCOPE_MAX_BYTES`,
`STEMSCOPE_MAX_SECONDS` and `STEMSCOPE_PILOT_DIR`. Relative paths resolve from the
working directory. `TORCH_HOME` independently controls downloaded checkpoint
storage. For example:

```sh
STEMSCOPE_CONFIG=settings.example.json uv run --frozen stemscope-api
```

Both entry points emit JSON application logs to stderr. Uvicorn access logs use
its standard format. Loaded models are reused in each process, and PyTorch reuses
cached checkpoints on disk. Outputs are deliberately generated afresh: there is no
result cache that could silently reuse a different model/configuration. UI and API
processes maintain separate in-memory models; running both can increase RAM usage.

### Docker (CPU API)

The Dockerfile installs the committed dependency lock with Python 3.11, runs as a
non-root user and excludes datasets, outputs and checkpoints from its build context.
With Docker running, build and launch the API:

```sh
docker build --platform linux/amd64 -t stemscope .
docker run --rm --platform linux/amd64 -p 127.0.0.1:8000:8000 \
  -v stemscope-storage:/storage stemscope
```

The named volume retains generated stems and checkpoints. First inference needs
network access to download model weights. The locked PyTorch stack can make the
image large; amd64 emulation on Apple Silicon can be slow. GPU deployment is not
configured. The Linux amd64 image was built and verified locally with both
separators; this is a local CPU smoke test, not public or GPU deployment validation.

### Reproducible checks and design

```sh
uv sync --frozen --extra dev
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
uv run --frozen pytest -q
```

The committed lock pins Python dependencies; model downloads and benchmark input
hashes are recorded separately by the existing experiment manifests. See
[architecture and execution contracts](docs/architecture.md) for service boundaries,
concurrency, caching, storage lifecycle and testing strategy. This phase adds HTTP
contract/error/concurrency tests and configuration/logging tests to the existing
suite. Test success does not establish separation accuracy on a user's song.

Local Phase 11 validation: **156 tests passed**, lint/format checks passed and the
frozen dependency sync succeeded. Real Demucs and Open-Unmix API requests each
returned four downloadable, decodable WAV stems using cached weights. These
requests used FastAPI TestClient. Docker verification is recorded below.

To repeat the running-server audio smoke check with a short WAV file:

```sh
uv run --frozen python scripts/verify_api.py /absolute/path/short-song.wav
```

The script runs both models, downloads every stem, checks finite samples and
matching duration, and writes `outputs/api-verification.json`. Override `--url`
for a different local port. This checks API/audio integrity, not music quality.

Docker verification (2026-09-13): the frozen image build passed. The container ran
as UID 10001, served its health endpoint, separated a seven-second pilot clip with
both models and returned eight finite, decodable WAVs of matching duration. All
stem downloads survived a container restart; corrupt audio returned 400 and a
missing stem returned 404. Evidence is saved locally in
`outputs/docker-verification.json` and `outputs/docker-checkpoint-hashes.json`.
The temporary containers were removed after testing; the image and
`stemscope-verification` volume remain available locally.

Demucs downloaded its checkpoint successfully. Open-Unmix's upstream Zenodo host
returned 504 twice, so its inference check used the existing local checkpoints
copied into the container with `docker cp`; SHA-256 hashes matched the originals.
Thus Open-Unmix inference is verified, while its fresh network download did not
succeed during this run. A host bind mount was denied by macOS; the documented
Docker-managed volume worked. The image is approximately 3.4 GB, and the first
build took over 20 minutes to fetch the Linux/PyTorch dependencies on this connection.
