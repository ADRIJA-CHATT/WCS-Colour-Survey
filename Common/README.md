# Common code

This directory contains the shared data, metric, splitting, and plotting utilities used by both speaker-level analyses.

## Modules

- `config.py`: canonical paths and reproducibility constants.
- `data.py`: loads the locally stored canonical WCS table and checks the expected 110-language/330-chip structure.
- `download_wcs.py`: one-time acquisition of the public WCS source files; `run_all.py` never downloads data.
- `load_wcs.py`: legacy-compatible WCS parsers retained for older scripts.
- `metrics.py`: speaker statistics, exact hypergeometric rarefaction, Miller--Madow entropy/MI corrections, language prevalence vocabularies, bootstrap intervals, and all-valid-pair resolution summaries.
- `splits.py`: deterministic approximately 80/20 speaker holdout within every language.
- `plotting.py`: minimal shared figure-saving helper.

## Statistical conventions

### Speaker identity

WCS speaker numbers are unique only within language. Therefore the canonical individual key is

\[
(\texttt{language\_id},\texttt{speaker\_id}).
\]

All speaker-level aggregation and splitting uses this composite identity.

### Missing responses

The WCS design targets 330 chips per speaker, but a cleaned speaker can have fewer valid responses. Project 1 only forms a pair when both labels are observed; Project 2 normalizes its empirical contingency table over the speaker's observed responses.

### Repertoire comparability

Raw speaker label counts depend on the number of observed responses. The descriptive standardized repertoire therefore uses exact without-replacement hypergeometric rarefaction to a common target of 313 valid responses (95% of the intended 330) whenever the speaker has at least 313 valid responses.

### Information-theoretic bias correction

For a contingency table with \(r\) non-empty rows, \(c\) non-empty columns, \(q\) non-empty cells, and \(n\) observations, the first-order Miller--Madow correction obtained from

\[
I(X;Y)=H(X)+H(Y)-H(X,Y)
\]

is

\[
\widehat I_{MM}=\widehat I_{MLE}+\frac{r+c-q-1}{2n}.
\]

The code implements this sign convention explicitly and tests it.

## Source status

The code was audited against the supplied result archive. The uploaded result snapshot was generated during an earlier code revision. In particular, the snapshot's Project 2 information-efficiency values were produced before the Miller--Madow MI sign correction now present in `metrics.py`. Consequently those Project 2 numerical results are **provisional** until the corrected pipeline is rerun on the canonical WCS dataset. See `../RESULTS_AUDIT.md`.

## Output policy

`results/` contains only machine-generated analysis outputs. Project-level figure captions belong in `project_1_geometry/FIGURE_CAPTIONS.md` and `project_2_information/FIGURE_CAPTIONS.md`; the analysis scripts do not write documentation files into `results/`.
