# PCG-BP analysis version 2

After three stages of selecting clean signals from the 78 participants in the PCG-BP dataset, 20 participants were retained. The third selection was based on the consistency of provisional S1/S2 detection.

### Table 1. Participants retained after the third selection

| Selected participants | | | | |
| :---: | :---: | :---: | :---: | :---: |
| 08 | 09 | 16 | 17 | 18 |
| 19 | 23 | 24 | 32 | 33 |
| 37 | 41 | 42 | 44 | 53 |
| 54 | 55 | 68 | 72 | 78 |

The original signals are taken from `../datasets/PCG-BP/Original_signal/Original_78_participants`. S1 and S2 detection uses the existing implementation from `../PCG-BP_analysis/S1S2 detection`, with an unchanged local copy in `code/vendor/stage2.py`. The detector is rerun on each **entire original recording**, after continuous filtering and envelope calculation. Previously calculated segment detections are not reused.

Signals are segmented at heart-sound boundaries, using S1 onset or S2 offset as the intended boundary types. **This run uses S1 onset consistently**: successive segments share an S1-onset boundary and never overlap. Original-ID segments are stored in `data/Participant_XX`. The participants are also reorganized to IDs 1-20, and those copies are stored in `data_reorder/Participant_XX`. The original selection and workflow above are preserved from the original README; an unchanged copy is included as `original_README.md`.

Time-domain features and normalized amplitude features are calculated for each segment using the definitions in Tang et al. (2018), [PCG Classification Using Multidomain Features and SVM Classifier](https://doi.org/10.1155/2018/4205027), Tables 1 and 2, Sections 2.3-2.5.3. The requested 20 timing and 12 normalized amplitude features are provided. Other feature families from the paper are outside this run.

## Run results: 6 October 2026

- 20 participants, six approximately 10-second feature segments each: **120 segments and 120 detection plots**.
- **975 complete valid cycles** contribute to both feature families.
- 117 segments have at least two valid cycles and all 32 features are finite.
- Original participant 19, segment 01 (reordered ID 06), and participant 41, segment 01 (reordered ID 12), have one valid cycle: means are available, SDs are NaN.
- Original participant 72, segment 01 (reordered ID 19), has no complete valid adjacent cycle: all 32 features are NaN. Its waveform and plot are retained for review.
- Automated verification passed: source hashes, exact reconstruction including edge fragments, nonoverlap, original/reordered equivalence, filter reproduction, independent feature recalculation, and decoding of every plot. See `qc/verification_summary.json`.

## Output layout

```text
data/Participant_08/...                  Original IDs, raw/filtered WAV and NPZ
  segment_01.npz                        Signal arrays plus provenance
  segment_01_raw.wav                    Exact original waveform slice
  segment_01_filtered.wav               Continuously filtered waveform slice
  excluded_edges/prefix.npz             Preserved leading samples
  excluded_edges/suffix.npz             Preserved trailing samples
data_reorder/Participant_01/...          Same feature segments, reordered IDs
S1S2 detection/Participant_08/           Existing requested detection folder
  full_recording_detection.npz          Full-record representations and composite
  full_recording_landmarks.csv          Global labels and onset/peak/offset samples
  full_recording_candidates.csv         Includes ambiguous candidates
  full_recording_model.json             Model and full-record QC
  cycle_audit.csv                       Accepted/rejected candidate cycle audit
  segment_01.png                        Detection visualization for this segment
  segment_01_landmarks.csv              Local and original recording coordinates
Feature extraction_reorder 20 participants_resegment/
  time domain features/
    segment_features.csv                120 rows, 20 features plus metadata
    Participant_01/segment_features.csv  Per-reordered-participant tables
    feature_equations.pdf               Rendered equations
    feature_equations.tex               Editable LaTeX source
    feature_equations_overleaf.txt      Complete source to paste into Overleaf
  normalized amplitude features/        Corresponding 12-feature outputs
  cycle_level_features.csv              975 cycles, boundaries and intermediate values
participant_id_mapping.csv              Explicit 1-20 mapping
segment_manifest.csv                    Exact cuts, duration, counts, paths and flags
processing_config.json                  Filter, detector and aggregation conventions
qc/                                     Source hashes, edge inventory and verification
code/                                   Reproducible pipeline, validator and vendor code
```

The existing directory spelling `S1S2 detection` is used for the requested S1/S2 detection visualizations. All plot and segment directories retain original participant IDs. Feature subdirectories use reordered IDs, and every feature row carries both IDs.

## Participant ID mapping

| Original ID | Reordered ID | Original ID | Reordered ID |
| --- | --- | --- | --- |
| 08 | 01 | 37 | 11 |
| 09 | 02 | 41 | 12 |
| 16 | 03 | 42 | 13 |
| 17 | 04 | 44 | 14 |
| 18 | 05 | 53 | 15 |
| 19 | 06 | 54 | 16 |
| 23 | 07 | 55 | 17 |
| 24 | 08 | 68 | 18 |
| 32 | 09 | 72 | 19 |
| 33 | 10 | 78 | 20 |

## Processing and segmentation

1. Read each original mono WAV at 2,000 Hz. No resampling is performed. Save its SHA-256 hash.
2. Apply the existing 30-500 Hz Butterworth bandpass (`N=4`, eight-order bandpass transfer function) with SciPy `sosfiltfilt` over the entire recording. Never filter each segment separately.
3. Recompute the existing mean/peak-normalized Hilbert and Shannon representations over the entire filtered signal. Smooth using the existing 51-sample centered moving average. Run the original candidate detector, temporal sequence classification and 20%-height boundary detection with unchanged parameters. Detector edge review is limited to the original recording ends.
4. Start the feature span at the first resolved S1 onset and end at the last resolved S1 onset. Select subsequent cuts nearest the previous cut plus 10 seconds. If at most 15 seconds remain, use the final onset as the last cut, avoiding a tiny residual feature segment. Actual durations are recorded, not assumed to equal 10 seconds.
5. Save zero-based, half-open intervals `[start_sample,end_sample)`. A shared cut belongs to the later segment, so adjacent segments have no duplicate samples. Store raw, filtered and segment-normalized arrays in NPZ; store raw and filtered WAVs as well.
6. Preserve samples before the first cut and after the final cut as `excluded_edges` NPZ files under the original participant folder. These fragments are not feature segments and are not copied into the reordered feature dataset. Their sample counts and reasons appear in `qc/edge_fragments.csv`. Concatenating prefix, all six segments and suffix exactly reconstructs the original signal.

NPZ metadata includes the original and reordered participant IDs, displayed participant ID, segment ID, sampling rate, absolute sample cuts, recording times, source path/hash and amplitude normalization parameters. In reordered NPZs, `participant_id` uses the new ID; the original ID remains explicit. Raw and filtered arrays are identical between the two datasets.

## Feature conventions

Only complete **adjacent** S1-S2-S1 triplets with valid, ordered boundaries are eligible. An intervening ambiguous candidate invalidates that triplet; it is not silently skipped to bridge a cycle. All ten timing quantities share the complete-cycle pool. All six amplitude ratios share the pool with finite nonzero state amplitudes. This conservative pool is an explicit implementation choice, rather than an assertion that the paper specifies this rejection rule.

For boundaries `a=S1 onset`, `b=S1 offset`, `c=S2 onset`, `d=S2 offset`, `e=next S1 onset`, the four states are `[a,b)`, `[b,c)`, `[c,d)`, `[d,e)`. Durations are differences divided by sampling frequency. The identity `RR = IntS1 + IntSys + IntS2 + IntDia` is verified. Here RR is the PCG cycle period between S1 onsets; there is no ECG R-peak measurement.

The timing quantities are `RR`, `IntS1`, `IntS2`, `IntSys`, `IntDia`, `Ratio_SysRR`, `Ratio_DiaRR`, `Ratio_SysDia`, `Ratio_S1RR` and `Ratio_S2RR`. Their segment means (`m_`) and sample standard deviations (`sd_`) produce 20 features. Intervals are in seconds, and ratios are dimensionless fractions.

For amplitude features, first normalize each entire saved filtered segment to zero mean and unit population SD. Compute the mean absolute normalized waveform amplitude separately in each state. The ratios are `Sys/S1`, `Dia/S2`, `S1/S2`, `S1/Dia`, `Sys/Dia` and `S2/Sys`. Compute ratios per cycle before aggregation; these are not ratios of pooled state means, peak amplitudes, envelope amplitudes or RMS values. Denominators at or below `1e-12` invalidate the cycle. Each ratio's mean and sample SD produce the 12 requested columns.

Aggregation SD uses `ddof=1`; signal normalization uses `ddof=0`. The paper names SD without specifying that denominator, so this convention is documented explicitly. No cycles gives NaN means and SDs; one cycle gives a mean but NaN SD. Missing values are never replaced with zero. A terminal next-S1 onset can equal a segment's exclusive endpoint to close its last cycle; the next S1 waveform still belongs only to the following segment.

The PDFs and Overleaf files provide equations and all feature-column names. PDFs were generated and visually checked independently using ReportLab and rendered mathematical expressions. The built-in LaTeX compiler could not initialize its platform directories; therefore compilation of the provided `.tex`/`.txt` source remains unverified here.

## Detection review and interpretation

The plots show filtered PCG, smoothed Hilbert and smoothed Shannon signals. Green markers indicate S1, red S2, amber crosses ambiguous candidates. Dashed lines indicate onset, dotted lines offset, and shading indicates resolved sound windows. Plot HR is the **whole-recording** detector estimate; displayed S1/S2/ambiguous counts are local to the segment. Full-record landmark samples are absolute; segment landmark files contain both local and recording sample coordinates. Candidate IDs remain stable across the recording.

Labels remain provisional. The detector is the project's timing-based baseline, not Tang et al.'s Springer HSMM. Its confidence values are heuristic consistency scores, not calibrated probabilities. Whole-record detection can differ from earlier overlapping-window results because normalization, autocorrelation and temporal-model selection now use the entire recording. Selection as a clean participant does not guarantee that every cycle is valid. Review the three flagged segments before downstream modeling, and keep all segments of each participant in the same train/test group.

The paper's preprocessing used 10 Hz high-pass filtering, spike removal, zero-mean/unit-SD normalization and Springer segmentation. This rerun intentionally retains the requested existing project detector and 30-500 Hz preprocessing while adopting the requested feature definitions. It is not an exact reproduction of the paper's classification pipeline. No BP values are used for detection, segmentation or feature extraction.

## Reproduction

Use the existing named Conda environment `C:\Users\yiky4\.conda\envs\scg_pcg_bp`; no environment is placed inside this project. From this directory in PowerShell:

```powershell
& 'C:\Users\yiky4\.conda\envs\scg_pcg_bp\python.exe' -X utf8 .\code\run_pipeline.py --seconds 10
& 'C:\Users\yiky4\.conda\envs\scg_pcg_bp\python.exe' -X utf8 .\code\validate_outputs.py
```

The pipeline deterministically writes its named outputs in this directory. Back up results before deliberately rerunning with changed settings. NumPy 2.4.6 and SciPy 1.17.1 were used. Plotting uses Matplotlib; the validator uses Pillow. Equation generation (`code/build_documents.py`) additionally requires ReportLab; this run accessed ReportLab from the bundled Codex runtime while using the existing Conda scientific packages.
