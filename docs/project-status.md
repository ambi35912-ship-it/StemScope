# Project status and demonstration

StemScope is a working local research/practice application with two pretrained
separators, audio analysis, reproducible evaluation, an experimental selector,
practice controls, comparison, a dashboard and a separation API. The engineering
milestones have been implemented and locally tested. The research goals of a
consistently better learned selector and reliable artifact-quality ratings remain open.

## What is ready

| Area | Delivered | Evidence or limitation |
| --- | --- | --- |
| Separation | Demucs and Open-Unmix, four playable stems | Both exercised locally and inside Docker |
| Analysis | Waveform, spectrum, energy and other features | Describes the recording; does not improve separation |
| Benchmarking | Reference SI-SDR, timing, memory and CSV reports | 50 seven-second official test previews; not full songs |
| Failure analysis | Conditions, metric failures and listening cases | Exploratory associations, not established causes |
| Model selection | Mixture-feature Random Forest and fixed fallbacks | Balanced policy did not beat the fixed baseline on holdout |
| Quality inspection | Interpretable signal diagnostics | Missed 10 of 14 negative-improvement outputs in the audit |
| Practice | Mute, solo, gain, speed, pitch and section looping | Render changes before playback; browser looping may have gaps |
| Comparison/dashboard | Audio, common-scale charts and recorded evidence | Historical benchmark scores are not upload accuracy |
| Engineering | API, JSON config/logging, cached models, lockfile, tests, Docker | 156 software tests passed; local CPU container smoke tests passed |

The separators use pretrained weights; this project has not trained a new
source-separation network. The original learned component is the small model
selector. Its 23-track holdout Balanced utility was 6.367 versus 6.401 for always
choosing Open-Unmix. These are utility values, not accuracy percentages.

The quality inspection is a diagnostic prototype. It does not justify calibrated
HIGH/MEDIUM/LOW ratings or establish that an unflagged stem sounds clean.

## Five-minute demo

Start from the project directory:

```sh
.venv/bin/stemscope
```

1. Upload a short music excerpt you have permission to use. Start with roughly
   10–20 seconds containing vocals, drums and bass.
2. Open **Dashboard**, select a model manually and click **Build dashboard**.
   Explain the duration/features, play each stem and point out audible leakage
   where present. Do not describe historical SI-SDR as a score for this upload.
3. Use **Stem separation** to populate its players, then use **Practice**. Mute
   bass, select a short section and render at 80% speed. Play the practice mix.
   Practice uses the separation-tab stems; dashboard results do not populate it.
4. Open **Compare models**, run both and compare the same stem. Discuss the
   runtime/quality tradeoff and shared-scale spectrogram. Spectral appearance alone
   does not establish which output sounds better.
5. Show the saved benchmark/selector reports. Explain the held-out result honestly:
   the selector is implemented, but this experiment did not demonstrate an advantage
   over the fixed Balanced baseline.

For the backend demo, start `.venv/bin/stemscope-api`, open
http://127.0.0.1:8000/docs and submit a WAV through `POST /separations` using a model
name from `GET /models`. Download the returned stem URLs. The API runs separately
from Gradio and loads its own model copies.

## Evidence available in this workspace

Generated data and reports remain local and are excluded from source control:

- `outputs/phase4-study/study-ko5nfljz/report.md`: controlled reference benchmark.
- `outputs/phase5-analysis`: exploratory failures and listening cases.
- `outputs/phase6-selector/report.md`: validation gates and holdout baselines.
- `outputs/phase7-audit/report.md`: diagnostic coverage and missed problems.
- `outputs/docker-verification.json`: both models, downloads and restart checks.
- `outputs/docker-checkpoint-hashes.json`: copied checkpoint integrity evidence.

Docker built and ran successfully as a non-root user. Open-Unmix's fresh download
failed twice with upstream 504 responses; its container inference passed using
verified cached checkpoints. The temporary test containers were removed; the image
and verification volume remain. This is not public-hosting or GPU validation.

## Next research milestone: improve measured music quality

Before changing models or training anything, collect a new, appropriately licensed
set of music with matching original isolated stems. Split by song/artist and reserve
a final test set before examining its outcomes. The existing selector holdout has
already been inspected and must not be reused as a fresh test for further tuning.

Freeze the baseline models/settings and evaluation protocol. Include full songs or
representative longer sections, measure per-stem reference scores and runtime, and
collect blinded listening judgments for leakage and watery/muffled artifacts.
Compare each proposed change against the same baseline and report regressions as
well as wins. Improve the diagnostic estimator only after obtaining suitable human
labels and separate validation data. Establish an improvement on fresh held-out
recordings before claiming better accuracy.

Training a separator is a separate future project requiring licensed multitracks,
a training pipeline, compute and independent evaluation. No training, cloud spend
or publication has been started as part of this handoff.
