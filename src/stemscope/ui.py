import os
import time
from collections.abc import Mapping
from pathlib import Path

from stemscope.analysis import AnalysisService
from stemscope.application import choose_model, create_services
from stemscope.benchmarking.datasets import load_public_pilot
from stemscope.benchmarking.demo import create_demo
from stemscope.benchmarking.results import completed_study
from stemscope.benchmarking.runner import BenchmarkRunner, BenchmarkTrack
from stemscope.comparison import ComparisonService
from stemscope.config import Settings
from stemscope.dashboard import DashboardService, dashboard_tables
from stemscope.errors import StemScopeError
from stemscope.failure_analysis import load_review, review_audio
from stemscope.practice import PracticeOptions, render_practice
from stemscope.quality.service import inspect_and_save
from stemscope.selection.inference import AUTO_CHOICES, load_evaluation
from stemscope.separators.base import STEMS
from stemscope.service import SeparationService


def build_app(
    services: Mapping[str, SeparationService], analysis_service: AnalysisService | None = None
):
    """Build the local UI without loading model weights."""
    import gradio as gr

    if not services:
        raise ValueError("At least one separator service is required")
    analysis_service = analysis_service or AnalysisService(next(iter(services.values())).settings)

    benchmark_runner = BenchmarkRunner(services, next(iter(services.values())).settings)
    comparison_service = ComparisonService(services, benchmark_runner.settings)
    dashboard_service = DashboardService(services, benchmark_runner.settings, analysis_service)

    def benchmark(upload, vocals, drums, bass, other, provenance):
        yield "Benchmarking both models… this may include first-use downloads.", [], None, None
        try:
            if not upload:
                raise StemScopeError("Choose a WAV or MP3 mix first.")
            refs = {
                name: Path(path) for name, path in zip(STEMS, (vocals, drums, bass, other)) if path
            }
            track = BenchmarkTrack(
                Path(upload).name,
                Path(upload),
                refs,
                provenance.strip() or "user-supplied; authenticity not verified",
            )
            result = benchmark_runner.run([track])
            yield benchmark_view(result)
        except StemScopeError as exc:
            yield str(exc), [], None, None

    def benchmark_view(result):
        rows = [
            [
                row["track"],
                row["model"],
                row["stem"],
                round(row["wall_seconds"], 2),
                round(row["sampled_peak_rss_bytes"] / 1024**2, 1),
                round(row["si_sdr_db"], 2) if row["si_sdr_db"] is not None else "—",
                round(row["si_sdr_improvement_db"], 2)
                if row["si_sdr_improvement_db"] is not None
                else "—",
                row["error"] or row["metric_status"],
            ]
            for row in result.rows
        ]
        status = (
            "Complete"
            if all(row["status"] == "ok" for row in result.rows)
            else "Finished with errors; inspect the results."
        )
        return status, rows, str(result.csv_path), str(result.manifest_path)

    def real_music_check():
        yield "Evaluating prepared real-music references with both models…", [], None, None
        try:
            tracks = load_public_pilot(benchmark_runner.settings.pilot_dir)
            result = benchmark_runner.run(tracks)
            view = benchmark_view(result)
            yield (view[0] + f" · {len(tracks)} real excerpts · small pilot only", *view[1:])
        except StemScopeError as exc:
            yield str(exc), [], None, None

    def synthetic_check():
        yield "Running a synthetic pipeline check… not a music-quality evaluation.", [], None, None
        try:
            track = create_demo(benchmark_runner.settings.output_dir)
            result = benchmark_runner.run([track])
            view = benchmark_view(result)
            yield (view[0] + " · SYNTHETIC TEST ONLY", *view[1:])
        except StemScopeError as exc:
            yield str(exc), [], None, None

    def separate(upload, model_name):
        filename = Path(upload).name if upload else ""
        yield filename, "Processing… first use may download model weights.", "", "", *([None] * 4)
        try:
            if not upload:
                raise StemScopeError("Choose a WAV or MP3 file first.")
            started = time.perf_counter()
            automatic = model_name in AUTO_CHOICES
            model_name, selection_note = choose_model(
                Path(upload), model_name, benchmark_runner.settings
            )
            if model_name not in services:
                raise StemScopeError("Choose one of the available separation models.")
            result = services[model_name].process(upload)
            yield (
                result.filename,
                "Complete" + (f" · {selection_note}" if selection_note else ""),
                f"{time.perf_counter() - started if automatic else result.seconds:.2f} seconds",
                result.model,
                *(str(result.stems[name]) for name in STEMS),
            )
        except StemScopeError as exc:
            yield filename, str(exc), "", "", *([None] * 4)

    def analyze(upload):
        yield "Analyzing audio… first use may take longer.", [], "", None, None, None
        try:
            if not upload:
                raise StemScopeError("Choose a WAV or MP3 file first.")
            result = analysis_service.process(upload)
            audio = result.audio
            rows = [
                ["Original filename", result.filename],
                ["Duration", f"{audio.duration:.3f} seconds"],
                ["Original sample rate", f"{audio.sample_rate:,} Hz"],
                ["Channels", str(audio.channels)],
                [
                    "Estimated tempo",
                    f"{audio.tempo_bpm:.1f} BPM" if audio.tempo_bpm else "Unavailable",
                ],
                ["Mean frame RMS", f"{audio.rms.mean():.5f} amplitude"],
                ["Mean spectral centroid", f"{audio.centroid.mean():.1f} Hz"],
                ["Mean spectral bandwidth", f"{audio.bandwidth.mean():.1f} Hz"],
                ["Mean zero-crossing rate", f"{audio.zcr.mean():.5f} crossings/sample"],
            ]
            notes = (
                f"{audio.tempo_note}\n\n"
                f"Features cover the full track at {audio.analysis_rate:,} Hz; "
                f"the spectrogram shows frequencies up to {audio.analysis_rate / 2:,.0f} Hz. "
                "Waveforms show the original channels. Silence contributes zero to feature averages. "
                "These are audio descriptions, not separation-quality scores."
            )
            yield (
                f"Complete · {result.seconds:.2f} seconds",
                rows,
                notes,
                *(str(result.plots[name]) for name in ("waveform", "spectrogram", "features")),
            )
        except StemScopeError as exc:
            yield str(exc), [], "", None, None, None

    with gr.Blocks(title="StemScope", delete_cache=(3600, 86400)) as app:
        gr.Markdown("# StemScope\nLocal music separation, analysis and practice")
        upload = gr.File(
            label="Upload WAV or MP3 (max 200 MB, 20 minutes)",
            file_types=[".wav", ".mp3"],
            type="filepath",
        )
        with gr.Tab("Dashboard"):
            gr.Markdown(
                "Build one report for the uploaded song: input analysis, selected model, timings, playable stems and diagnostic evidence. Historical benchmark scores are shown separately and do not score this upload."
            )
            dashboard_choice = gr.Dropdown(
                choices=[*services, *AUTO_CHOICES],
                value=next(iter(services)),
                label="Dashboard separator",
                interactive=True,
            )
            dashboard_both = gr.Checkbox(
                value=False, label="Include both models (runs both separators)"
            )
            dashboard_button = gr.Button("Build dashboard", variant="primary")
            dashboard_status = gr.Textbox(label="Dashboard status", interactive=False)
            dashboard_summary = gr.Dataframe(headers=["Measurement", "Value"], interactive=False)
            dashboard_note = gr.Textbox(
                label="Selection, timing and quality notes", lines=5, interactive=False
            )
            dashboard_models = gr.Dataframe(
                headers=[
                    "Model for this upload",
                    "Status",
                    "Separation seconds",
                    "Device",
                    "Error / note",
                ],
                interactive=False,
            )
            dashboard_players = [
                gr.Audio(label=f"Selected model — {stem}", type="filepath", interactive=False)
                for stem in STEMS
            ]
            dashboard_quality = gr.Dataframe(
                headers=["Stem", "Review status", "Evidence", "RMS dBFS", "Waveform similarity"],
                interactive=False,
            )
            with gr.Accordion("Input waveform and spectral analysis", open=False):
                dashboard_waveform = gr.Image(label="Input waveform", interactive=False)
                dashboard_spectrogram = gr.Image(label="Input spectrogram", interactive=False)
                dashboard_features = gr.Image(label="Input feature curves", interactive=False)
            dashboard_comparison = gr.Image(
                label="Optional vocals comparison — shared scale", interactive=False
            )
            gr.Markdown(
                "Historical reference benchmark — different recordings. These medians are not the current upload's accuracy. Use Compare models for other side-by-side stems."
            )
            dashboard_benchmark_note = gr.Textbox(
                label="Historical benchmark source", interactive=False
            )
            dashboard_benchmark = gr.Dataframe(
                headers=[
                    "Model",
                    "Stem",
                    "Tracks",
                    "Median SI-SDR dB",
                    "Q1 dB",
                    "Q3 dB",
                    "Median improvement dB",
                ],
                interactive=False,
            )
            dashboard_file = gr.File(label="Download full dashboard evidence")
            dashboard_outputs = [
                dashboard_status,
                dashboard_summary,
                dashboard_note,
                dashboard_waveform,
                dashboard_spectrogram,
                dashboard_features,
                dashboard_models,
                *dashboard_players,
                dashboard_quality,
                dashboard_comparison,
                dashboard_benchmark_note,
                dashboard_benchmark,
                dashboard_file,
            ]

            def build_dashboard(mix, choice, both):
                empty = [
                    "Building dashboard… model processing may take a while.",
                    [],
                    "",
                    None,
                    None,
                    None,
                    [],
                    *([None] * 4),
                    [],
                    None,
                    "",
                    [],
                    None,
                ]
                yield tuple(empty)
                try:
                    if not mix:
                        raise StemScopeError("Upload a song first.")
                    result = dashboard_service.build(mix, choice, both)
                    payload = result.payload
                    summary, models, diagnostics = dashboard_tables(payload)
                    note = (
                        payload["selection_note"]
                        + ". "
                        + payload["audio"]["tempo_note"]
                        + " "
                        + payload["timing_note"]
                        + " "
                        + payload["upload_reference_quality"]
                    )
                    note += (
                        " Diagnostic flags are review aids; no flags does not imply clean audio."
                    )
                    if payload["errors"]:
                        note += " Issues: " + "; ".join(payload["errors"])
                    yield (
                        payload["status"],
                        summary,
                        note,
                        *(
                            payload["charts"][key]
                            for key in ["waveform", "spectrogram", "features"]
                        ),
                        models,
                        *(payload["selected_stems"].get(stem) for stem in STEMS),
                        diagnostics,
                        payload["comparison_chart"],
                        payload["historical_benchmark"]["status"],
                        payload["historical_benchmark"]["rows"],
                        str(result.manifest),
                    )
                except StemScopeError as exc:
                    empty[0] = str(exc)
                    yield tuple(empty)

            dashboard_button.click(
                build_dashboard,
                [upload, dashboard_choice, dashboard_both],
                dashboard_outputs,
                api_name="build_dashboard",
                concurrency_limit=1,
                concurrency_id="audio-processing",
            )
        with gr.Tab("Audio analysis"):
            gr.Markdown("Inspect the uploaded song without running separation.")
            analyze_button = gr.Button("Analyze audio", variant="primary")
            analysis_status = gr.Textbox(label="Analysis status", value="Ready", interactive=False)
            summary = gr.Dataframe(
                headers=["Measurement", "Value"],
                row_count=9,
                datatype=["str", "str"],
                interactive=False,
                label="Audio summary",
            )
            notes = gr.Textbox(label="Analysis notes", interactive=False, lines=4)
            waveform = gr.Image(label="Original waveform", interactive=False)
            spectrogram = gr.Image(label="Spectrogram", interactive=False)
            features = gr.Image(label="Feature curves", interactive=False)
            with gr.Accordion("What do these measurements mean?", open=False):
                gr.Markdown(
                    "- **RMS:** signal amplitude over short windows; a proxy for energy, not perceived loudness.\n"
                    "- **Spectral centroid:** the spectrum's frequency center, often related to brightness.\n"
                    "- **Spectral bandwidth:** frequency spread around that center.\n"
                    "- **Zero-crossing rate:** how frequently the signal changes sign.\n"
                    "- **Spectrogram:** frequency content over time; brighter colors show stronger components.\n"
                    "- **Tempo:** a rhythm estimate that can be ambiguous or unavailable."
                )
            analyze_button.click(
                analyze,
                upload,
                [analysis_status, summary, notes, waveform, spectrogram, features],
                concurrency_limit=1,
                concurrency_id="audio-processing",
            )
        with gr.Tab("Stem separation"):
            model = gr.Dropdown(
                choices=[*services, *AUTO_CHOICES],
                value=next(iter(services)),
                label="Separation model",
                interactive=True,
            )
            gr.Markdown(
                "Choose a model, then separate. Auto modes use the Phase 6 experimental policy and show when validation requires a fixed-model fallback. Quality prioritizes reference score; Balanced and Speed apply increasing CPU-time penalties. Short-excerpt evidence does not guarantee full-song quality. Each model downloads its weights on first use."
            )
            button = gr.Button("Separate stems", variant="primary")
            filename = gr.Textbox(label="Original filename", interactive=False)
            status = gr.Textbox(label="Status", value="Ready", interactive=False)
            seconds = gr.Textbox(label="Total processing time", interactive=False)
            used_model = gr.Textbox(label="Model used for these stems", interactive=False)
            players = [
                gr.Audio(label=name.title(), type="filepath", interactive=False) for name in STEMS
            ]
            button.click(
                separate,
                [upload, model],
                [filename, status, seconds, used_model, *players],
                concurrency_limit=1,
                concurrency_id="audio-processing",
            )
            gr.Markdown(
                "Phase 7 diagnostics inspect up to the first 30 seconds. Keep the same uploaded mix that produced these stems. Flags suggest what to review; no flags does not mean clean audio. These checks do not confirm watery artifacts or give an accuracy percentage."
            )
            quality_button = gr.Button("Inspect these stems")
            quality_note = gr.Textbox(
                label="Diagnostic scope and reconstruction", interactive=False
            )
            quality_table = gr.Dataframe(
                headers=[
                    "Stem",
                    "Review status",
                    "Evidence",
                    "RMS dBFS",
                    "Sample peak",
                    "Most similar stem",
                    "Waveform similarity",
                    "High-band power fraction",
                    "Spectral flatness",
                ],
                interactive=False,
            )
            quality_file = gr.File(label="Download quality diagnostics")

            def inspect_quality(mix, vocals, drums, bass, other):
                try:
                    if not mix or not all([vocals, drums, bass, other]):
                        raise StemScopeError("Separate a song first, then inspect its stems.")
                    result, path = inspect_and_save(
                        Path(mix),
                        {
                            name: Path(value)
                            for name, value in zip(STEMS, [vocals, drums, bass, other])
                        },
                        benchmark_runner.settings.output_dir,
                    )
                    rows = [
                        [
                            r["stem"],
                            r["review_status"],
                            "; ".join(r["flags"]) or "None of the configured checks triggered",
                            *[
                                r[key]
                                for key in [
                                    "rms_dbfs",
                                    "sample_peak",
                                    "most_similar_stem",
                                    "max_waveform_similarity",
                                    "high_band_fraction",
                                    "spectral_flatness",
                                ]
                            ],
                        ]
                        for r in result["rows"]
                    ]
                    note = (
                        f"Inspected {result['analyzed_seconds']:.1f} of {result['full_duration_seconds']:.1f} seconds. {result['reconstruction_status']}. "
                        + result["limitations"]
                    )
                    return note, rows, str(path)
                except (StemScopeError, OSError, ValueError) as exc:
                    raise gr.Error(str(exc)) from exc

            quality_button.click(
                inspect_quality,
                [upload, *players],
                [quality_note, quality_table, quality_file],
                api_name="quality_diagnostics",
                concurrency_limit=1,
                concurrency_id="audio-processing",
            )
        with gr.Tab("Practice"):
            gr.Markdown(
                "Separate a song in Stem separation first. Choose the section and instruments you want, then render one synchronized practice mix. Change controls and render again to apply them; no model rerun is needed."
            )
            muted_stems = gr.CheckboxGroup(choices=list(STEMS), label="Mute stems", value=[])
            solo_stems = gr.CheckboxGroup(choices=list(STEMS), label="Solo stems", value=[])
            gr.Markdown(
                "Solo plays only the selected stems. Mute always wins if a stem is in both lists."
            )
            gains = [
                gr.Slider(0, 150, value=100, step=5, label=f"{name.title()} volume (%)")
                for name in STEMS
            ]
            with gr.Row():
                section_start = gr.Number(value=0, label="Section start (seconds)", minimum=0)
                section_end = gr.Number(
                    value=0, label="Section end (0 = next 120 seconds or track end)", minimum=0
                )
            gr.Markdown(
                "Render up to 120 seconds of source audio at a time. An end beyond the track is clamped to its end. Times refer to the original song, before speed changes."
            )
            with gr.Row():
                practice_speed = gr.Slider(
                    0.5, 1.5, value=1, step=0.05, label="Speed (pitch preserved)"
                )
                practice_pitch = gr.Slider(
                    -12, 12, value=0, step=1, label="Pitch shift (semitones)"
                )
            loop_enabled = gr.Checkbox(value=True, label="Loop the rendered section")
            render_button = gr.Button("Render practice mix", variant="primary")
            practice_status = gr.Textbox(label="Practice status", interactive=False)
            practice_player = gr.Audio(
                label="Synchronized practice mix",
                type="filepath",
                interactive=False,
                loop=True,
                show_download_button=True,
            )
            practice_manifest = gr.File(label="Download practice settings")
            gr.Markdown(
                "The player repeats the rendered section when Loop is on. Five-millisecond edge fades soften clicks; browser looping may have a gap. Speed/pitch processing can add artifacts. If the mix exceeds playback headroom, all stems are reduced together to preserve their balance."
            )

            def practice(
                vocals,
                drums,
                bass,
                other,
                muted,
                solo,
                vocal_gain,
                drum_gain,
                bass_gain,
                other_gain,
                start,
                end,
                speed,
                pitch,
            ):
                yield "Rendering practice mix…", None, None
                try:
                    files = [vocals, drums, bass, other]
                    if not all(files):
                        raise StemScopeError("Separate a song first, then return to Practice.")
                    options = PracticeOptions(
                        gains={
                            name: gain / 100
                            for name, gain in zip(
                                STEMS, [vocal_gain, drum_gain, bass_gain, other_gain]
                            )
                        },
                        muted=tuple(muted or []),
                        solo=tuple(solo or []),
                        start_seconds=start,
                        end_seconds=end,
                        speed=speed,
                        pitch_semitones=pitch,
                    )
                    result = render_practice(
                        {name: Path(path) for name, path in zip(STEMS, files)},
                        options,
                        benchmark_runner.settings.output_dir,
                    )
                    note = f"Ready: original section {result.source_start:.2f}–{result.source_end:.2f}s → {result.output_seconds:.2f}s at {speed:.2f}×, pitch {pitch:+g} semitones. Rendered in {result.render_seconds:.2f}s."
                    if result.normalization_gain < 1:
                        note += f" Applied shared headroom gain {result.normalization_gain:.3f}."
                    yield note, str(result.audio), str(result.manifest)
                except (StemScopeError, TypeError, ValueError) as exc:
                    yield f"Cannot render: {exc}", None, None

            render_button.click(
                practice,
                [
                    *players,
                    muted_stems,
                    solo_stems,
                    *gains,
                    section_start,
                    section_end,
                    practice_speed,
                    practice_pitch,
                ],
                [practice_status, practice_player, practice_manifest],
                api_name="practice_mix",
                concurrency_limit=1,
                concurrency_id="audio-processing",
            )
            loop_enabled.change(
                lambda enabled: gr.update(loop=enabled),
                [loop_enabled],
                [practice_player],
                api_name="practice_loop",
            )
        with gr.Tab("Compare models"):
            gr.Markdown(
                "Run both separators on the same captured upload, then switch stems to compare their sound and spectrograms. Play one model at a time. Audio keeps each model's original level; loudness differences can bias listening."
            )
            compare_button = gr.Button("Compare both models", variant="primary")
            comparison_state = gr.State(value=None)
            compare_status = gr.Textbox(label="Comparison status", interactive=False)
            compare_stem = gr.Dropdown(
                choices=list(STEMS), value="vocals", label="Stem to compare", interactive=True
            )
            comparison_players = [
                gr.Audio(label=name, type="filepath", interactive=False) for name in services
            ]
            comparison_chart = gr.Image(
                label="Spectrogram comparison — shared scale", interactive=False
            )
            comparison_table = gr.Dataframe(
                headers=[
                    "Model",
                    "Stem",
                    "Status",
                    "Separation seconds",
                    "Diagnostic status",
                    "Evidence",
                    "RMS dBFS",
                    "Waveform similarity",
                ],
                interactive=False,
            )
            comparison_file = gr.File(label="Download comparison evidence")
            gr.Markdown(
                "Timing is one interactive call per model and may include loading; it is not the controlled benchmark. Spectrograms and diagnostics inspect up to 30 seconds. No reference-based quality score is available for this upload. Review flags can miss artifacts, and spectral differences do not establish which model is better."
            )

            def compare_models(mix, stem):
                yield (
                    "Running both models… first use may load weights.",
                    None,
                    *([None] * len(services)),
                    None,
                    [],
                    None,
                )
                try:
                    if not mix:
                        raise StemScopeError("Upload a song first.")
                    result = comparison_service.run(mix)
                    rows = [
                        [
                            r[key]
                            for key in [
                                "model",
                                "stem",
                                "status",
                                "separation_seconds",
                                "review_status",
                                "evidence",
                                "rms_dbfs",
                                "waveform_similarity",
                            ]
                        ]
                        for r in result.rows
                    ]
                    try:
                        audio, chart = comparison_service.view(result.directory, stem)
                        note = result.status + f" · {Path(mix).name}"
                    except (StemScopeError, OSError, ValueError, KeyError) as exc:
                        audio, chart = [None] * len(services), None
                        note = result.status + f" · cannot load comparison view: {exc}"
                    yield (
                        note,
                        str(result.directory),
                        *audio,
                        str(chart) if chart else None,
                        rows,
                        str(result.manifest),
                    )
                except StemScopeError as exc:
                    yield str(exc), None, *([None] * len(services)), None, [], None

            def switch_comparison(directory, stem):
                try:
                    if not directory:
                        return *([None] * len(services)), None
                    audio, chart = comparison_service.view(Path(directory), stem)
                    return *audio, str(chart)
                except (StemScopeError, OSError, ValueError, KeyError) as exc:
                    raise gr.Error(str(exc)) from exc

            compare_button.click(
                compare_models,
                [upload, compare_stem],
                [
                    compare_status,
                    comparison_state,
                    *comparison_players,
                    comparison_chart,
                    comparison_table,
                    comparison_file,
                ],
                api_name="compare_models",
                concurrency_limit=1,
                concurrency_id="audio-processing",
            )
            compare_stem.change(
                switch_comparison,
                [comparison_state, compare_stem],
                [*comparison_players, comparison_chart],
                api_name="comparison_stem",
                concurrency_limit=1,
                concurrency_id="audio-processing",
            )
        with gr.Tab("Benchmark"):
            with gr.Accordion("Completed Phase 4 study", open=True):
                gr.Markdown(
                    "Load the saved full study without rerunning separation. Scores are in dB, not accuracy percentages."
                )
                saved_button = gr.Button("Load completed Phase 4 results")
                saved_status = gr.Textbox(label="Saved study", interactive=False)
                saved_table = gr.Dataframe(
                    headers=[
                        "Model",
                        "Stem",
                        "Tracks scored",
                        "Median SI-SDR (dB)",
                        "Q1 (dB)",
                        "Q3 (dB)",
                        "Median improvement (dB)",
                    ],
                    interactive=False,
                )
                saved_chart = gr.Image(label="Quality across test excerpts")
                saved_report = gr.File(label="Download study report")
                saved_csv = gr.File(label="Download all measured results")

                def load_saved_study():
                    try:
                        return completed_study(
                            benchmark_runner.settings.output_dir / "phase4-study"
                        )
                    except StemScopeError as exc:
                        raise gr.Error(str(exc)) from exc

                saved_button.click(
                    load_saved_study,
                    [],
                    [saved_status, saved_table, saved_chart, saved_report, saved_csv],
                    api_name="completed_study",
                )
            gr.Markdown(
                "Run both models on the uploaded mix. Without references, only performance is measured. "
                "For quality scores, supply the matching original WAV stems—not outputs from a separator. "
                "All five WAV files must have identical sample rate, channels, length, and start time."
            )
            reference_files = [
                gr.File(
                    label=f"Original {name} reference (optional)",
                    file_types=[".wav"],
                    type="filepath",
                )
                for name in STEMS
            ]
            provenance = gr.Textbox(
                label="Reference source / dataset and split",
                placeholder="For example: my original recording, or dataset name / test split",
            )
            benchmark_button = gr.Button("Benchmark both models", variant="primary")
            demo_button = gr.Button("Run synthetic pipeline check")
            real_button = gr.Button("Run prepared real-music pilot")
            gr.Markdown(
                "The prepared real-music pilot uses two official seven-second test excerpts with original references. "
                "Both are from the same artist; the scores apply to these clips, not music in general."
            )
            gr.Markdown(
                "The synthetic check uses generated tones/noise with known components; "
                "it does not establish accuracy on music. SI-SDR is in dB, not percent. "
                "Higher is better; positive improvement means better than the original mix. "
                "A dash means unavailable—see the metric status. Memory is sampled for the whole process, "
                "including cached models. Time is repeated per stem; do not add the four rows together."
            )
            benchmark_status = gr.Textbox(
                label="Benchmark status", value="Ready", interactive=False
            )
            benchmark_table = gr.Dataframe(
                headers=[
                    "Track",
                    "Model",
                    "Stem",
                    "Time (s)",
                    "Process peak (MiB)",
                    "SI-SDR (dB)",
                    "Improvement (dB)",
                    "Metric status / error",
                ],
                datatype=["str", "str", "str", "number", "number", "str", "str", "str"],
                row_count=8,
                interactive=False,
            )
            csv_download = gr.File(label="Download results CSV", interactive=False)
            protocol_download = gr.File(
                label="Download protocol and input hashes", interactive=False
            )
            benchmark_outputs = [benchmark_status, benchmark_table, csv_download, protocol_download]
            benchmark_button.click(
                benchmark,
                [upload, *reference_files, provenance],
                benchmark_outputs,
                concurrency_limit=1,
                concurrency_id="audio-processing",
            )
            real_button.click(
                real_music_check,
                [],
                benchmark_outputs,
                concurrency_limit=1,
                concurrency_id="audio-processing",
            )
            demo_button.click(
                synthetic_check,
                [],
                benchmark_outputs,
                concurrency_limit=1,
                concurrency_id="audio-processing",
            )
        with gr.Tab("Failure analysis"):
            gr.Markdown(
                "Explore saved Phase 5 conditions and listen to low-scoring examples. These are diagnostic flags, not confirmed artifacts. Higher SI-SDR is better; scores are not percentages."
            )
            review_dir = benchmark_runner.settings.output_dir / "phase5-analysis"
            load_button = gr.Button("Load Phase 5 analysis")
            condition_table = gr.Dataframe(
                headers=[
                    "Condition",
                    "Model",
                    "Stem",
                    "Tracks scored",
                    "Median SI-SDR (dB)",
                    "Worse than mix",
                ],
                interactive=False,
            )
            case_choice = gr.Dropdown(label="Choose an example to review", choices=[])
            review_report = gr.File(label="Analysis report")
            review_csv = gr.File(label="Condition results CSV")
            gr.Markdown(
                "Compare the original reference with each estimate. Listen for other instruments leaking in, missing target sound, and watery or muffled artifacts. Playback uses the first successful run; displayed scores are medians across repetitions."
            )
            listen_button = gr.Button("Load listening example")
            listening = [
                gr.Audio(label=label, type="filepath", interactive=False)
                for label in [
                    "Original mix",
                    "Original target reference",
                    "Demucs estimate",
                    "Open-Unmix estimate",
                ]
            ]

            def show_review():
                try:
                    table, choices, report, csv = load_review(review_dir)
                    return table, gr.update(choices=choices, value=None), report, csv
                except StemScopeError as exc:
                    raise gr.Error(str(exc)) from exc

            def listen_to_case(case):
                try:
                    return review_audio(review_dir, case)
                except (StemScopeError, OSError, KeyError, ValueError) as exc:
                    raise gr.Error(
                        "Cannot load the example. Check that the analysis and original audio files are present and unchanged."
                    ) from exc

            load_button.click(
                show_review,
                [],
                [condition_table, case_choice, review_report, review_csv],
                api_name="failure_analysis",
            )
            listen_button.click(
                listen_to_case, [case_choice], listening, api_name="failure_example"
            )
        with gr.Tab("Model selector"):
            gr.Markdown(
                "Inspect the Phase 6 experiment before using an Auto mode in Stem separation. The learned candidate is compared with fixed choices; validation determines whether it is enabled. Utility includes your quality/time preference. Oracle agreement is model-choice agreement, not audio accuracy."
            )
            selector_button = gr.Button("Load selector evaluation")
            selector_table = gr.Dataframe(
                headers=[
                    "Preference",
                    "Policy",
                    "Held-out tracks",
                    "Mean utility",
                    "Mean regret",
                    "Oracle agreement",
                ],
                interactive=False,
            )
            selector_report = gr.File(label="Selector report")
            selector_csv = gr.File(label="All evaluation results")

            def show_selector():
                try:
                    return load_evaluation(benchmark_runner.settings.output_dir / "phase6-selector")
                except StemScopeError as exc:
                    raise gr.Error(str(exc)) from exc

            selector_button.click(
                show_selector,
                [],
                [selector_table, selector_report, selector_csv],
                api_name="selector_evaluation",
            )
    return app.queue(max_size=8)


def main() -> None:
    from stemscope.logging_config import configure_logging

    configure_logging()
    settings = Settings.from_env()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(settings.output_dir / ".matplotlib"))
    os.environ.setdefault("GRADIO_TEMP_DIR", str(settings.output_dir / ".gradio"))
    from stemscope.auth import AuthSettings, SupabaseAuth

    auth_settings = AuthSettings.from_env()
    if auth_settings:
        import uvicorn

        from stemscope.auth_web import create_authenticated_ui

        uvicorn.run(
            create_authenticated_ui(
                settings, create_services(settings), SupabaseAuth(auth_settings)
            ),
            host="127.0.0.1",
            port=7860,
        )
        return
    app = build_app(create_services(settings))
    app.launch(
        server_name="127.0.0.1",
        share=False,
        allowed_paths=[str(settings.output_dir)],
        max_file_size=settings.max_bytes,
        show_error=False,
    )
