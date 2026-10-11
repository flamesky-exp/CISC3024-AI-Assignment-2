# IVON on frozen StarNet features - CISC3024 Assignment 2

This AI-authored project evaluates a Bayesian linear classifier on frozen
StarNet-S050 features. It follows Assignment 1's controlled-experiment and LaTeX
workflow. The original Assignment 1 directory is not modified.

## Deliverables

- `output/pdf/IVON_StarNet_CIFAR10_Report.pdf`: compiled English report.
- `report/main.tex`: editable LaTeX source, with figures and university logo.
- `results/`: measured predictions, per-run records, summaries and audit evidence.
- `scripts/experiment.py`: numerical checks, feature extraction, training and evaluation.
- `scripts/analyze.py`: statistics, conditional paired bootstrap and figures.
- `scripts/build_report.py`: generate the report from results and compile with Tectonic.

## Main finding

Across classifier seeds 42, 43 and 44, Adam achieves 85.48% mean accuracy and
0.4656 NLL; IVON BMA (32 draws) achieves 85.42% and 0.4730. Posterior averaging
slightly improves IVON's NLL compared with its mean weights but does not beat
Adam under this protocol. These are repeated heads on one shared backbone,
not three independently trained image classifiers. See the report for limitations.

## Environment and setup

The recorded run used Python 3.13.1, PyTorch 2.9.1+cu128 and an NVIDIA RTX 3070
Laptop GPU. The exact environment is recorded in `results/environment.json`
and `results/package_versions.json`.

Create a virtual environment and install PyTorch for the intended hardware.
The recorded CUDA setup uses:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For CPU execution install the matching CPU PyTorch wheels. CPU execution is
supported by the experiment code but was not benchmarked in this study.

IVON is vendored under `vendor/ivon`; it does not require a separate pip install.
Its downloaded files and SHA-256 hashes are recorded in `results/provenance.json`.
The archive includes the reused backbone checkpoint and exact split indices.
`scripts/bootstrap.py` was used only to import local Assignment 1 assets and
download IVON initially; users of the complete archive do not need to run it.

Download/extract the CIFAR-10 Python dataset to a chosen folder. Torchvision can
do this using `CIFAR10(data_root, train=True, download=True)` and the analogous
test call. The folder must contain `cifar-10-batches-py/`. The recorded local
data root was `D:\CISC3024 project 1\data\cifar10_stable`.

## Reproduction

Run from the project root, replacing `DATA_ROOT` with that dataset folder:

```powershell
python scripts/experiment.py --stage smoke --data-root DATA_ROOT
python scripts/experiment.py --stage full --data-root DATA_ROOT
python scripts/experiment.py --stage audit --data-root DATA_ROOT
python scripts/retime.py
python scripts/analyze.py
python scripts/build_report.py
```

The full run overwrites the corresponding run artifacts. Use a separate copy
of the project if you want to preserve the supplied results. Cached features
are reused if `data/features.pt` exists; regenerate that cache if changing the
backbone, data or feature preprocessing. Resume-from-interrupted-training is
not implemented. Audit reloads the saved posterior and reproduces saved BMA
probabilities using the recorded prediction seeds.

Tectonic must be on PATH or specified with the `TECTONIC_EXE` environment
variable. The report builder also detects the original local project's portable
compiler. Its initial compilation may download TeX packages. To generate the
LaTeX source without compiling, run `python scripts/build_report.py --source-only`.
Alternatively upload the complete `report/` folder to a LaTeX editor and compile
`main.tex` using XeLaTeX or pdfLaTeX. The report uses an internal bibliography.

## Protocol and interpretation

- Fixed original 45,000/5,000/10,000 train/validation/test split.
- Features standardized using training-only moments; no new image augmentation.
- Two candidate learning rates per method, selected with seed-42 validation NLL.
- Seeds 43 and 44 reuse the selected rate; every checkpoint is selected by validation NLL.
- 60 head-training epochs; validation at epochs 1 and multiples of 5.
- Same head initialization, independent matched minibatch ordering and L2 coefficient.
- 20 IVON draws for validation; 32 for the main test result.
- Test evaluation starts only after `results/selected.json` is written.
- The old backbone already used this validation split and test benchmark.
- Bootstrap intervals resample paired test images conditional on the saved models.
- The project does not claim full-network IVON reproduction or OOD robustness.

## Submission status

The report, code, results, LaTeX and reproduction materials are prepared.
The project-specific source-code webpage is **not yet published**. The assignment
requires a webpage link to this implementation. Upload the prepared source archive
to the chosen repository or other course-accepted web location and insert the
verified URL in the report before submission. An official IVON link alone does
not fulfill that requirement. The learning reflection is an AI-assisted draft
and should be checked against the student's own understanding.

## Sources and licenses

- IVON (ICML 2024): https://proceedings.mlr.press/v235/shen24b.html
- Official optimizer: https://github.com/team-approx-bayes/ivon
- StarNet (CVPR 2024): https://github.com/ma-xu/Rewrite-the-Stars
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- Temperature scaling: https://proceedings.mlr.press/v70/guo17a.html

See `THIRD_PARTY_NOTICES.md` and `LICENSES/`. Do not remove the third-party
license texts when sharing the project. University branding is used only in
the academic report and is not covered by the software license.
