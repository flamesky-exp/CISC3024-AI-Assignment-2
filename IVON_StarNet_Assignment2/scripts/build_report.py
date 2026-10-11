"""Build a complete LaTeX source from measured JSON artifacts, then compile."""
from pathlib import Path
import json, shutil, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
S=json.loads((R/'summary.json').read_text())
M=json.loads((R/'metrics.json').read_text())
B=json.loads((R/'bootstrap.json').read_text())
P=json.loads((R/'provenance.json').read_text())
F=json.loads((R/'features.json').read_text())
E=json.loads((R/'environment.json').read_text())
selected=json.loads((R/'selected.json').read_text())
labels={'adam':'Adam','adam_ts':'Adam + TS','ivon_mean':'IVON mean','ivon_bma':'IVON BMA (32)'}
def num(m,k,scale=1,d=4):
    z=S[m][k];return f"{scale*z['mean']:.{d}f} \\pm {scale*z['sd']:.{d}f}"
table='\n'.join(labels[m]+' & '+ ' & '.join('$'+num(m,k,scale,d)+'$' for k,scale,d in [('accuracy',100,2),('nll',1,4),('brier',1,4),('ece',100,2)])+r' \\' for m in labels)
seedtable='\n'.join(f"{r['seed']} & {labels[r['method']]} & {100*r['accuracy']:.2f} & {r['nll']:.5f} & {r['brier']:.5f} & {100*r['ece']:.3f}"+r' \\' for r in M)
boottable='\n'.join(f"BMA minus {labels[m]} & {B[m]['nll_difference']:+.6f} & [{B[m]['ci95'][0]:+.6f}, {B[m]['ci95'][1]:+.6f}]"+r' \\' for m in B)
timing='\n'.join(f"{labels[m]} & {S[m]['fit_seconds']['mean']:.3f} & {1000*S[m]['head_inference_seconds']['mean']:.3f}"+r' \\' for m in labels)
temps=', '.join(f"{r['temperature']:.4f}" for r in M if r['method']=='adam_ts')
paper=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm,headheight=15pt]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,array,float,xcolor}
\usepackage{microtype,fancyhdr,enumitem,placeins}
\usepackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue]{hyperref}
\graphicspath{{figures/}{assets/}}
\setlength{\parindent}{1.5em}
\setlength{\parskip}{0.3em}
\setlength{\emergencystretch}{3em}
\setcounter{tocdepth}{1}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small CISC3024 Pattern Recognition}
\fancyhead[R]{\small AI Assignment \#2}
\fancyfoot[C]{\thepage}
\newcommand{\E}{\mathbb{E}}
\begin{document}
\hypersetup{pageanchor=false}
\begin{titlepage}
\thispagestyle{empty}
\begin{center}
\vspace*{\fill}
\includegraphics[width=.80\textwidth]{UM_Logo.png}
\vspace{1cm}

{\fontsize{14}{18}\selectfont CISC 3024 Pattern Recognition -- AI Assignment \#2 Report\par}
\vspace{.7cm}
{\fontsize{22}{27}\selectfont\bfseries
Bayesian Image Classification with IVON\\
on Frozen StarNet Features\\
Accuracy and Calibration on CIFAR-10\par}
\vspace{1.1cm}
{\fontsize{12}{15}\selectfont\underline{Ren Xianglin U-C3-2745-1}\par}
\vspace{.9cm}
{\fontsize{12}{15}\selectfont October 2026\par}
\vspace*{\fill}
\end{center}
\end{titlepage}
\pagenumbering{roman}
\hypersetup{pageanchor=true}
\begin{abstract}
This report evaluates Improved Variational Online Newton (IVON), a Bayesian
variational learning method presented at ICML 2024, for CIFAR-10 image
classification. A previously trained StarNet-S050 is reused as a fixed feature
extractor. A deterministic Adam-trained classifier is compared with temperature
scaling, an IVON posterior-mean classifier, and Bayesian model averaging (BMA)
over 32 posterior draws. All new classifiers use the same feature representation,
data split, regularization coefficient, and paired initialization seeds. Learning
rates and checkpoints are selected using validation negative log-likelihood
(NLL), before test evaluation. Across three classifier seeds, Adam obtains
85.48\% accuracy, NLL 0.4656, and expected calibration error (ECE) 1.01\%; IVON
BMA obtains 85.42\%, 0.4730, and 2.59\%, respectively. Averaging improves NLL
slightly relative to IVON's posterior mean, but does not outperform Adam under
this protocol. Independent checkpoint reloading reproduces the IVON BMA
probabilities exactly in the recorded environment. The study documents algorithm
selection, implementation, numerical checks, experimental limitations, and
learning reflections. It is a last-layer adaptation with a reused backbone,
rather than a reproduction of the paper's full-network experiments.
\end{abstract}
\clearpage
\tableofcontents
\clearpage\pagenumbering{arabic}

\section{Introduction and research question}
Image classification requires assigning an image to a category, but many uses
also require a meaningful estimate of confidence. Two classifiers can predict
the same labels while allocating very different probabilities to those labels.
An incorrect prediction with high confidence is penalized strongly by NLL,
even when its effect on accuracy is only one misclassified image.

Assignment 1 studied a CIFAR-10 adaptation of StarNet, whose defining block uses
element-wise multiplication between two feature branches \cite{ma}. This report
retains that learned representation and investigates the probabilistic classifier
placed on top of it. IVON is selected because it provides an explicit approximate
weight posterior through variational learning \cite{shen}. The hypothesis is that
posterior averaging may improve probability quality while retaining competitive
classification accuracy. The experiment is designed to permit a negative answer.

The research question is: \emph{under a fixed StarNet feature representation,
does an IVON-trained Bayesian classifier improve test NLL and calibration over
a deterministic classifier?} A second question asks whether averaging posterior
predictions improves on using only IVON's mean weights. Keeping the feature
extractor fixed isolates the head-level comparison and makes repeated runs
practical. It also deliberately restricts the scope: uncertainty in the feature
extractor is not modeled.

\section{How AI selected the algorithm}
The AI agent first read the assignment PDF and inspected the previous project,
including its LaTeX source, model implementation, split indices, checkpoint, and
experiment records. It then searched for recent Bayesian image-classification
methods. Search queries included ``2024 2025 Bayesian image classification last
layer Laplace approximation paper'' and ``Variational Learning is Effective for
Large Deep Networks IVON 2024 github official.'' The search also identified
Laplace Redux (2021) and Variational Linearized Laplace Approximation (2024)
\cite{daxberger,ortega}.

Selection used three practical criteria: an explicit probability or Bayesian
foundation, a recent peer-reviewed publication, and an inspectable implementation
that could support an experiment on the available hardware. IVON satisfied these
criteria and enabled a direct posterior-mean versus posterior-average comparison.
The publication year is 2024; it is a recent candidate relative to this October
2026 assignment, not a claim that it is the newest available method. The search
was targeted, not a systematic review of all Bayesian classifiers.

The AI verified the publication through the ICML/PMLR entry and inspected the
authors' official optimizer repository \cite{ivoncode}. The original optimizer
was downloaded with its GPL-3.0 license and SHA-256 hashes recorded. It was used
without rewriting its update rules. The new experiment driver, validation,
analysis, and report-generation code were authored by AI. This distinction
separates integration work from inventing a new optimization algorithm.

\section{Bayesian formulation and algorithm}
\subsection{A probabilistic linear classifier}
Let $\phi(x)\in\mathbb{R}^{128}$ denote a fixed StarNet feature vector, standardized
using only training-set statistics. Appending a constant permits the weights and
biases of a ten-class linear head to be collected into $w\in\mathbb{R}^{1290}$.
For logits $z_k(x,w)$, the categorical likelihood is
\begin{equation}
 p(y=k\mid x,w)=\frac{\exp z_k(x,w)}{\sum_{j=1}^{10}\exp z_j(x,w)}.
\end{equation}
The training data $D$ update a weight prior according to Bayes' rule:
\begin{equation}
 p(w\mid D)=\frac{p(D\mid w)p(w)}{p(D)}.
\end{equation}
The softmax likelihood makes exact integration of this posterior inconvenient.
We approximate it with a diagonal Gaussian $q(w)=\mathcal{N}(m,\operatorname{diag}
(\sigma^2))$. The frozen backbone itself has no posterior in this experiment.

\subsection{Variational objective and IVON state}
With $N=45{,}000$ training images, consider
\begin{equation}\label{eq:objective}
 \mathcal{J}(q)=\E_q\left[\frac{1}{N}\sum_{i=1}^N
 -\log p(y_i\mid x_i,w)\right]+\frac{1}{N}\operatorname{KL}(q\Vert p).
\end{equation}
For the Gaussian prior $p(w)=\mathcal{N}(0,(N\delta)^{-1}I)$, its contribution
includes $\delta\E_q[\|w\|^2]/2$. We set $\delta=0.001$, corresponding to
prior precision $N\delta=45$. The deterministic comparator minimizes average
cross-entropy plus the same quadratic penalty using Adam with coupled weight
decay. The numerical penalty is matched; the two objectives still differ because
IVON optimizes a distribution and includes its entropy.

The implementation maintains a mean, gradient momentum, and diagonal curvature
state $h$. The sampling variance is
\begin{equation}\label{eq:sigma}
 \sigma_j^2=\frac{1}{N(h_j+\delta)}.
\end{equation}
For one sampled weight vector and its minibatch gradient $g$, the implementation's
curvature estimate can be written
\begin{equation}
 \widehat h=(w-m)\odot g\oslash\sigma^2.
\end{equation}
Here $\odot$ and $\oslash$ are element-wise operations. With curvature decay
$\beta_2$, the positive-correction update is
\begin{equation}
 h_{t+1}=\beta_2h_t+(1-\beta_2)\widehat h_t+
 \frac{(1-\beta_2)^2}{2}\frac{(h_t-\widehat h_t)^2}{h_t+\delta}.
\end{equation}
The mean step divides a bias-corrected momentum plus $\delta m$ by
$h_{t+1}+\delta$, with component-wise clipping at radius 1. The official
implementation also rescales its supplied learning rate by $h_0+\delta=0.101$.
Thus its nominal learning rate is not directly comparable to Adam's.

\subsection{Prediction and uncertainty}
Posterior-mean prediction evaluates $p(y\mid x,m)$. BMA instead estimates
\begin{equation}\label{eq:bma}
 \bar p_k(x)=\frac{1}{S}\sum_{s=1}^S p(y=k\mid x,w^{(s)}),
 \qquad w^{(s)}\sim q(w).
\end{equation}
Softmax probabilities are averaged, rather than logits or hard class labels.
The final prediction is $\arg\max_k\bar p_k(x)$. In general,
$\E_q[\operatorname{softmax}(z)]\ne\operatorname{softmax}(\E_q[z])$.
The comparison between BMA and posterior-mean prediction measures the practical
effect of this distinction for the same fitted posterior.

A distribution over weights can represent parameter uncertainty, but neither
posterior averaging nor a Bayesian interpretation guarantees calibration on an
arbitrary dataset. Predictive entropy also combines several sources of uncertainty;
we do not interpret it as a pure measure of epistemic uncertainty or claim
out-of-distribution detection without a corresponding experiment.

\section{How AI implemented and checked the experiment}
The AI copied the previous StarNet-S050 implementation, checkpoint, split indices,
and license into a separate assignment-2 project. The original project remained
unchanged. The backbone was put in evaluation mode so that batch normalization
statistics were fixed. Each image was normalized using the original CIFAR-10
constants, and the 128-dimensional pooled features were cached. Feature centering
and scaling were fitted only on the 45,000 training vectors.

Each new classifier has 1,290 trainable parameters. Paired seeds initialize the
two methods with the same zero biases and normally distributed weights with
standard deviation 0.01. A separate random-number generator controls minibatch
order. Validation posterior sampling saves and restores the training RNG state,
so evaluation does not change subsequent training samples. Model state and IVON
optimizer state are saved together to preserve both posterior means and variances.

The implementation workflow was:
\begin{enumerate}[leftmargin=*]
\item inspect the assignment, previous project, publication, and official code;
\item save a protocol specifying data use, candidate learning rates, seeds,
metrics, checkpoint selection, and posterior sample counts;
\item verify normalization, posterior restoration, finite updates, positive
precision, and metric reference cases;
\item extract features and run a two-epoch smoke test on 2,048 training vectors;
\item run the declared validation search and three-seed formal experiment;
\item freeze chosen learning rates before computing the new test metrics;
\item save probabilities, reload checkpoints independently, and regenerate
tables and figures from measured files.
\end{enumerate}

For a uniform ten-class prediction, the metric test recovers NLL $\log 10$,
Brier score 0.9, and ECE zero on ten balanced labels. Perfect one-hot predictions
recover zero NLL and Brier score. The posterior sampling check confirms exact
restoration of mean parameters. These tests examine mathematical properties
instead of merely checking that a script runs.

The smoke test completed for both methods. A later audit started a separate
Python process, reloaded the saved IVON checkpoints, and reproduced all three
saved BMA probability arrays with maximum absolute difference $0$. This verifies
the saved posterior and prediction path on the recorded software/hardware;
it does not guarantee bitwise equality on every other device.

\section{Experimental settings}
\subsection{Data and provenance}
CIFAR-10 contains 60,000 color images of size $32\times32$ across ten classes
\cite{cifar}. The official training set is split into 45,000 training and 5,000
validation images using the saved class-stratified seed-42 indices from
Assignment 1. The official 10,000-image test set is evaluated after new model
selection. Training and validation indices are checked for disjointness and
complete coverage of the 50,000-image training set.

The reused backbone was originally trained for 50 epochs with image augmentation,
and its checkpoint was selected by validation accuracy. The current head training
uses fixed, unaugmented features. The old validation set therefore participates
in both historical backbone selection and current head selection. The test set
was also already evaluated in Assignment 1. This is a follow-up study on a known
benchmark, not a claim of a newly untouched test set. No new head setting was
chosen by its test performance.

\subsection{Matched comparisons and selection}
\begin{table}[H]\centering\small
\caption{Predeclared head-training and evaluation settings.}
\begin{tabular}{p{.31\textwidth}p{.60\textwidth}}\toprule
Setting & Value \\\midrule
Feature extractor & Frozen StarNet-S050, 414,010 total parameters \\
New classifier & Linear $128\to10$; 1,290 parameters \\
Head seeds & 42, 43, 44; one shared backbone and data split \\
Epoch budget / batch size & 60 / 1,024; final partial batch retained \\
Adam candidates & Learning rates 0.001 and 0.01; defaults $\beta=(0.9,0.999)$ \\
IVON candidates & Nominal learning rates 0.05 and 0.2 \\
IVON state & $h_0=0.1$, $\beta_1=0.9$, $\beta_2=0.999$, clip radius 1 \\
Regularization & Coupled coefficient $\delta=0.001$ for weights and biases \\
IVON effective sample size & 45,000; one weight draw per training minibatch \\
Schedule & Cosine decay to 1\% of initial learning rate \\
Checkpoint selection & Lowest validation NLL at epochs 1, 5, 10, \ldots, 60 \\
Posterior draws & 20 for validation; 32 for main test evaluation \\
Temperature scaling & Scalar $T\in[0.05,20]$ fitted on validation NLL \\
\bottomrule\end{tabular}\end{table}

Both candidate learning rates are evaluated at seed 42. The selected rate is then
used for seeds 43 and 44, with each run selecting its checkpoint by validation
NLL. IVON selection uses 20-draw BMA. The winning rates are 0.001 for Adam and
0.05 for IVON. All six selected runs choose epoch 5; the entire 60-epoch budget
was nevertheless executed. The higher-rate candidates select epoch 1 and have
worse validation NLL. Sampling uses fixed evaluation seeds. Validation is
reused for temperature fitting, so its apparent calibration improvement should
not be interpreted as an independent estimate of generalization.

Temperature scaling replaces logits $z$ by $z/T$, with a positive scalar $T$
\cite{guo}. It preserves each image's predicted class, while modifying its
probability vector. The fitted temperatures for seeds 42, 43, and 44 are
@TEMPS@, respectively.

\subsection{Metrics, uncertainty estimates, and timing}
We report accuracy, mean NLL, multiclass Brier score, and 15-bin ECE:
\begin{align}
 \mathrm{NLL}&=-\frac{1}{n}\sum_i\log p_{i,y_i},\\
 \mathrm{Brier}&=\frac{1}{n}\sum_i\sum_{k=1}^{10}(p_{ik}-\mathbf{1}[y_i=k])^2,\\
 \mathrm{ECE}&=\sum_{b=1}^{15}\frac{|B_b|}{n}
 |\mathrm{accuracy}(B_b)-\mathrm{confidence}(B_b)|.
\end{align}
The Brier score is summed over classes, not divided by ten. Confidence is the
largest predicted probability. ECE bins have equal width over $[0,1]$, with empty
bins omitted and confidence 1 assigned to the final bin. Probabilities are
clipped below at $10^{-12}$ only when taking logarithms. ECE is descriptive and
bin-dependent, so NLL and Brier provide additional evidence.

Reported standard deviations are sample standard deviations across three head
seeds. For paired NLL differences, image-wise losses are first averaged across
the three seeds, then 10,000 image indices are resampled with replacement 2,000
times using bootstrap seed 20261011. Percentile intervals quantify test-image
variation conditional on these fitted models, not uncertainty across independently
trained backbones or new datasets.

The machine uses an NVIDIA GeForce RTX 3070 Laptop GPU and PyTorch @TORCH@.
Training uses FP32, deterministic PyTorch algorithms, no TF32, and fixed random
seeds. Feature extraction takes @FEATURESECONDS@ seconds. Head-fit times include
the full 60 epochs and periodic validation, but exclude feature extraction.
Head-only inference times are measured on all 10,000 cached test vectors, after
five warm-ups, as the median of 20 synchronized repetitions per seed. The table
reports the mean of those medians. This is a local implementation measurement,
not an end-to-end image-processing benchmark.

\section{Results}
\subsection{Accuracy and probability quality}
\begin{table}[H]\centering\small
\caption{Test results: mean $\pm$ sample SD across three head seeds.
Accuracy and ECE are percentages. Lower is better except for accuracy.}
\begin{tabular}{lcccc}\toprule
Method & Accuracy & NLL & Brier & ECE \\\midrule
@TABLE@
\bottomrule\end{tabular}\label{tab:main}\end{table}

Adam has the lowest mean NLL and ECE in Table~\ref{tab:main}. IVON BMA is
0.06 percentage points lower in accuracy and about 0.00736 higher in NLL than
Adam. Thus this experiment does not support an overall probability-quality
advantage for IVON. Temperature scaling leaves accuracy unchanged, slightly
improves mean Brier score, and slightly worsens test NLL and mean ECE. Optimization
of validation NLL does not guarantee an improvement in every test metric.

Within IVON, BMA lowers mean NLL from 0.4738 to 0.4730 and slightly improves
Brier and mean ECE relative to the posterior mean. Its accuracy is slightly
lower. The averaging effect is real in these saved predictions but small compared
with the gap to the deterministic comparator.

\subsection{Calibration and selective prediction}
\begin{figure}[H]\centering
\includegraphics[width=.90\textwidth]{reliability.pdf}
\caption{Seed-42 reliability diagrams. A point on the diagonal has matching
confidence and empirical accuracy within its bin. Marker size reflects bin
population. Sparse bins should not be overinterpreted.}
\end{figure}
The seed-42 reliability diagrams agree with the numerical ECE comparison:
IVON has larger weighted deviations between confidence and empirical accuracy.
These plots are not proof of calibration at every input or within every class.
The displayed seed is fixed in advance by convention, rather than chosen for
the most favorable picture.

\begin{figure}[H]\centering
\includegraphics[width=.90\textwidth]{risk_coverage.pdf}
\caption{Seed-42 selective error versus coverage. Images are retained in
descending maximum-probability order. This is a descriptive test-set curve,
not a deployed abstention threshold selected on test data.}
\end{figure}
At 80\% coverage, Adam's error among retained images is 6.300\%, compared with
6.3625\% for IVON BMA. At 50\% coverage, the values are 1.42\% and 1.46\%.
The methods therefore provide similar confidence ranking in this example;
IVON does not show a selective-prediction advantage. Temperature scaling can
change the ordering of maximum confidence across images even though it
preserves the predicted label of each image.

\subsection{Training trajectory and posterior sample count}
\begin{figure}[H]\centering
\includegraphics[width=.86\textwidth]{validation.pdf}
\caption{Seed-42 validation NLL at the selected learning rates. The best
recorded checkpoint is epoch 5 for both methods; later epochs are not chosen.}
\end{figure}
The validation trajectories show why the final epoch is not used automatically.
Training a head longer on a representation already fitted to the same training
images can worsen held-out probability estimates. Checkpoint selection prevents
the 60-epoch budget from forcing use of a poorer final model. The coarse five-epoch
inspection interval may miss a better checkpoint between recorded epochs.

\begin{figure}[H]\centering
\includegraphics[width=.92\textwidth]{sample_sensitivity.pdf}
\caption{Seed-42 posterior sample sensitivity, using prefixes of one fixed
sampling sequence. The main 32-draw setting was specified before this diagnostic.}
\end{figure}
The sample-count diagnostic evaluates 1, 4, 8, 16, 32, and 64 draws without
retraining or changing the selected checkpoint. Monte Carlo estimates can
fluctuate as samples are added, and ECE need not improve monotonically. This
single sampling sequence does not establish an optimal draw count or quantify
all Monte Carlo variability. No setting is changed in response to this test plot.

\subsection{Paired uncertainty and computational cost}
\begin{table}[H]\centering\small
\caption{Paired mean NLL differences and conditional 95\% bootstrap intervals.
Negative differences favor IVON BMA.}
\begin{tabular}{lrr}\toprule
Comparison & Difference & 95\% interval \\\midrule
@BOOT@
\bottomrule\end{tabular}\end{table}
The intervals are consistent with a small NLL improvement from averaging the
IVON posterior and a larger disadvantage relative to Adam. They should not be
read as evidence that IVON is generally inferior: they condition on one learned
feature extractor, one split, and this narrow training protocol.

\begin{table}[H]\centering\small
\caption{Measured compute cost. Fit includes 60 head-training epochs and
validation. Inference covers all 10,000 cached test feature vectors.}
\begin{tabular}{lrr}\toprule
Method & Head fit (s) & Head inference (ms) \\\midrule
@TIMING@
\bottomrule\end{tabular}\end{table}
The temperature row shares the Adam fit; its scalar fitting time is not included.
The two IVON rows share one trained posterior. BMA repeats the head calculation
and includes sampling and RNG-management overhead. Since the head is small,
these timings are especially sensitive to Python and GPU-launch overhead.
Caching the backbone makes the experiment inexpensive, but the original backbone
training and current feature extraction remain necessary costs.

\section{Discussion and limitations}
The main hypothesis is not supported under this setting. Bayesian inference
provides a principled way to incorporate parameter uncertainty, but predictive
quality depends on the approximate posterior, representation, prior, optimizer
settings, and selection protocol. The study cannot isolate which of these causes
the observed disadvantage. In particular, the limited two-rate search and fixed
prior do not establish the best achievable result for either method.

Several scope restrictions are important:
\begin{itemize}[leftmargin=*]
\item \textbf{Frozen representation.} Only the final 1,290 parameters are Bayesian;
feature uncertainty and end-to-end adaptation are excluded. The comparison is
not a reproduction of full-network IVON training in the original paper.
\item \textbf{Shared historical selection.} The backbone and current heads use
the same validation split at different stages. Repeated selection can adapt to
that validation set. A nested split or independently trained backbone would
provide a stronger future evaluation.
\item \textbf{Three conditional repeats.} Head seeds vary initialization, data
order, and IVON sampling. They do not vary the backbone, split, or dataset.
Small SD therefore does not imply universal stability.
\item \textbf{Approximation and budget.} A diagonal Gaussian cannot capture
weight correlations or multiple modes. Hyperparameters other than learning
rate are not searched. Early selected checkpoints may constrain posterior
adaptation, but testing that explanation would require a separate ablation.
\item \textbf{Metric scope.} ECE depends on binning, and accuracy is insensitive
to most probability changes. No robustness, corruption, fairness, or
out-of-distribution conclusion is supported by these experiments.
\item \textbf{Known benchmark.} The old test results were already available.
The current protocol avoids new test-based tuning, but does not turn this reused
test set into an unseen external validation dataset.
\end{itemize}

A useful next study would predeclare a separate validation split and evaluate
full-network IVON, several priors, and multiple independently trained backbones.
That would test a broader question at higher computational cost. Retuning the
current method after viewing these test results and reporting only an improved
variant would weaken the evidential value of this comparison.

\section{Learning reflection}
This assignment connects Bayes' rule to a working image-classification system.
I learned to distinguish a distribution over weights from the softmax probabilities
produced by a single set of weights. A softmax layer already outputs probabilities,
but that alone does not make its training Bayesian. In the present implementation,
the Bayesian component is the approximate posterior and the predictive averaging
over its samples.

I also learned that accuracy and probability quality answer different questions.
The four methods have similar accuracies, while their NLL and ECE differ more
clearly. Temperature scaling demonstrates the distinction particularly well:
the labels remain unchanged, but the probabilities and Brier score change.
A small improvement in one metric should not be described as an improvement
in every aspect of uncertainty estimation.

Another lesson is that experimental controls are part of the algorithm evaluation.
Matching the feature representation, initialization, regularization, and data order
makes the comparison easier to interpret. Separating validation selection from
test reporting reduces one source of optimistic conclusions, although reuse of
the historical backbone and validation set still imposes limitations.

Finally, AI can write and execute a substantial part of the workflow, but its
outputs need inspectable evidence. In this project the useful checks were saved
probabilities, explicit metric definitions, numerical reference tests, independent
checkpoint reloading, and source provenance. A negative experimental result is
valuable when it is reproducible and its scope is stated accurately. These
reflections are an AI-assisted draft for the student to review against their
own understanding before submission.

\section{Conclusion and source-code availability}
The project implemented an AI-assisted last-layer Bayesian classifier using
IVON and evaluated it on fixed StarNet features. Under the declared protocol,
Adam achieved better NLL and ECE than IVON BMA, while their accuracies differed
by only 0.06 percentage points. Averaging the IVON posterior provided a modest
NLL benefit relative to its mean weights. The evidence supports this limited
finding and does not establish a general ranking of Bayesian and deterministic
learning methods.

The accompanying project contains the AI-authored experiment and analysis scripts,
vendored IVON source with its license, the reused backbone, selected head
checkpoints, split indices, prediction arrays, raw metrics, and this LaTeX source.
The authors' reference implementation is available at
\url{https://github.com/team-approx-bayes/ivon}.
\textbf{The assignment-specific source-code webpage has not yet been published.}
The local source archive is prepared, but the assignment's requirement for a
webpage link remains outstanding until this project is uploaded and its URL is
inserted here. The official repository URL is not a substitute for that project URL.

\begin{thebibliography}{9}
\bibitem{shen} Y. Shen et al. ``Variational Learning is Effective for Large Deep
Networks.'' \emph{ICML}, PMLR 235, pp. 44665--44686, 2024.
\url{https://proceedings.mlr.press/v235/shen24b.html}.
\bibitem{ivoncode} Team Approximate Bayes. \emph{IVON: Official PyTorch optimizer
implementation}. Accessed October 11, 2026.
\url{https://github.com/team-approx-bayes/ivon}.
\bibitem{ma} X. Ma, X. Dai, Y. Bai, Y. Wang, and Y. Fu. ``Rewrite the Stars.''
\emph{CVPR}, 2024. \url{https://github.com/ma-xu/Rewrite-the-Stars}.
\bibitem{cifar} A. Krizhevsky. \emph{Learning Multiple Layers of Features from
Tiny Images}. Technical report, University of Toronto, 2009.
Dataset: \url{https://www.cs.toronto.edu/~kriz/cifar.html}.
\bibitem{guo} C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. ``On Calibration
of Modern Neural Networks.'' \emph{ICML}, PMLR 70, pp. 1321--1330, 2017.
\url{https://proceedings.mlr.press/v70/guo17a.html}.
\bibitem{daxberger} E. Daxberger, A. Kristiadi, A. Immer, R. Eschenhagen,
M. Bauer, and P. Hennig. ``Laplace Redux -- Effortless Bayesian Deep Learning.''
\emph{NeurIPS}, 2021. \url{https://github.com/runame/laplace-redux}.
\bibitem{ortega} L. A. Ortega, S. Rodriguez Santana, and D. Hernandez-Lobato.
``Variational Linearized Laplace Approximation for Bayesian Deep Learning.''
\emph{ICML}, PMLR 235, pp. 38815--38836, 2024.
\url{https://proceedings.mlr.press/v235/ortega24a.html}.
\end{thebibliography}

\clearpage\appendix
\section{Per-seed results and reproducibility}
\begin{table}[H]\centering\small
\caption{Individual test results; accuracy and ECE in percent.}
\begin{tabular}{rlrrrr}\toprule
Seed & Method & Accuracy & NLL & Brier & ECE \\\midrule
@SEEDTABLE@
\bottomrule\end{tabular}\end{table}
The result tables and plots are generated from JSON and NPZ artifacts, rather
than manually entered measurements. The implementation hashes the official IVON
files and the reused backbone. The backbone SHA-256 is
\begin{quote}\small\ttfamily
@HASH@
\end{quote}
The current batched extraction evaluates the original checkpoint at 85.19\%
test accuracy, whereas Assignment 1 recorded 85.18\%. This one-image discrepancy
is retained transparently: the current extraction uses vectorized GPU preprocessing
and a different batch size from the earlier pipeline. Its precise numerical cause
has not been isolated, so exact reproduction of the historical evaluation is not
claimed. This reference is not a new training run and is not used to select a head.
The independent audit above concerns the new IVON probabilities and reproduces
those arrays exactly in the current pipeline.

The minimal reproduction sequence, after installing the dependencies and obtaining
the CIFAR-10 Python dataset, is:
\begin{quote}\small\ttfamily
python scripts/experiment.py --stage smoke --data-root DATA\_ROOT\\
python scripts/experiment.py --stage full --data-root DATA\_ROOT\\
python scripts/experiment.py --stage audit --data-root DATA\_ROOT\\
python scripts/retime.py\\
python scripts/analyze.py\\
python scripts/build\_report.py
\end{quote}
The shipped archive already contains the backbone and split indices. It excludes
the CIFAR-10 image files, virtual environments, and temporary rendering cache.
Feature caches may be regenerated. Exact reruns require the same data, package
versions, GPU behavior, and evaluation seeds; numerical equality across hardware
is not assumed. The scripts are designed for complete runs and evaluation, not
for resuming interrupted IVON training midway through an epoch.

\section{AI workflow record and requirement mapping}
\begin{table}[H]\centering\small
\caption{Evidence corresponding to the assignment requirements.}
\begin{tabular}{p{.37\textwidth}p{.54\textwidth}}\toprule
Requirement & Evidence in this project \\\midrule
How AI finds the algorithm & Section 2: queries, candidates, selection rationale,
publication and source checks \\
Algorithm description & Section 3: likelihood, posterior, variational objective,
IVON state, posterior predictive average \\
How AI implements it & Section 4: code integration, checks, smoke test and audit \\
Experimental settings/results & Sections 5--7, raw JSON/NPZ, generated figures \\
What was learned & Section 8: reflection draft for student review \\
Webpage link of source code & Local archive prepared; project webpage publication
and URL insertion still required \\\bottomrule
\end{tabular}\end{table}

The actual workflow encountered a local execution-sandbox initialization failure;
approved shell execution was used to access the specified project and run the
experiment. No experimental metric was invented to bypass that obstacle.
The official IVON implementation downloaded successfully, GPU/data checks passed,
and the numerical smoke test passed before formal training. A timing review
identified unnecessary random-state bookkeeping on the deterministic inference
path. This was removed and inference timing was repeated; model selection and
saved probabilities were unchanged. The independent reload audit was retained.

The AI authored the new experiment, plotting, statistical analysis, and report
scripts. It reused the previously generated StarNet adaptation and the authors'
licensed IVON optimizer. The supplied project logs summarize actual actions;
they are not presented as verbatim transcripts of unrecorded interactions.
\end{document}
'''
for key,value in {'TABLE':table,'BOOT':boottable,'TIMING':timing,'SEEDTABLE':seedtable,'TEMPS':temps,'TORCH':E['torch'].replace('_',r'\_'),'FEATURESECONDS':f"{F['extraction_seconds']:.1f}",'HASH':P['backbone_sha256'][:32]+r'\\'+P['backbone_sha256'][32:]}.items():
    paper=paper.replace('@'+key+'@',value)
paper=paper.replace(r'\begin{figure}[H]',r'\begin{figure}[htbp]')
paper=paper.replace(r'\section{Discussion and limitations}',r'\FloatBarrier'+'\n'+r'\section{Discussion and limitations}')
report=ROOT/'report';report.mkdir(exist_ok=True)
(report/'main.tex').write_text(paper,encoding='utf8')
if '--source-only' in sys.argv:
    print('LaTeX source generated');sys.exit(0)
old=Path(r'D:\CISC3024 project 1')
compiler=os.environ.get('TECTONIC_EXE',str(old/'tools/tectonic/tectonic.exe'))
if not Path(compiler).exists():
    compiler=shutil.which('tectonic')
if not compiler:raise SystemExit('LaTeX source saved. Install Tectonic or set TECTONIC_EXE to compile.')
cache=ROOT/'tmp/tectonic-cache'
if not cache.exists() and (old/'report/.tectonic-cache').exists():shutil.copytree(old/'report/.tectonic-cache',cache)
env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(cache);env['XDG_CACHE_HOME']=str(cache);env['XDG_CONFIG_HOME']=str(ROOT/'tmp/tectonic-config')
out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
subprocess.run([compiler,'main.tex','--outdir',str(out),'--keep-logs','--keep-intermediates'],cwd=report,env=env,check=True)
shutil.copy2(out/'main.pdf',out/'IVON_StarNet_CIFAR10_Report.pdf')
print('Report compiled:',out/'IVON_StarNet_CIFAR10_Report.pdf')
