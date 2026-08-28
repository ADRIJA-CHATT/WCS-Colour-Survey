# Project 2 — Speaker-level colour-information representation

## Scientific question

Does the colour-label vocabulary of a speaker's primary language relate to how much physical-colour information is retained by that speaker's linguistic labels?

## Primary hypothesis

\(H_0\): language colour vocabulary is unrelated to speaker-level colour-information efficiency.

\(H_1\): language colour vocabulary is related to speaker-level colour-information efficiency.

The primary test is a one-degree-of-freedom language-clustered Wald test of the slope in the pre-specified linear fractional-response model.

## Information-theoretic response

Exact chip identity is not used because each observed chip has one recorded label for a speaker, making the empirical mapping deterministic. Instead, the 330 WCS chips are partitioned into 20 physical bins using CIELAB coordinates only.

For speaker \(s\),

\[I_s(B;Y)=\sum_{b,y}p_s(b,y)\lograc{p_s(b,y)}{p_s(b)p_s(y)}.\]

The first-order Miller–Madow correction is

\[\widehat I_{MM}=\widehat I_{MLE}+rac{r+c-q-1}{2n}.\]

Information efficiency is

\[E_s=\widehat I_{MM}/H(B).\]

## Primary regression

We fit the pre-specified linear fractional-response mean model

\[\mathrm{logit}\{\mathrm E(E_s\mid K_l)\}=eta_0+eta_1\log K_l.\]

The null hypothesis is

\[H_0:eta_1=0.\]

A cluster-robust covariance is calculated with language as the independent cluster because all speakers in a language share the same language-level predictor.

The test statistic is

\[W=\left(rac{\hateta_1}{\mathrm{SE}(\hateta_1)}
ight)^2\sim\chi^2_1\quad	ext{under }H_0,\]

under the usual large-cluster regularity conditions.

A fixed quadratic model is reported only as a sensitivity analysis; no cross-validation or model-selection step is used.

## Outputs

- `speaker_information.csv`
- `language_vocabulary.csv`
- `model_coefficients.csv`
- `quadratic_sensitivity_coefficients.csv`
- `model_summary.txt`
- `hypothesis_test.csv`
- `language_summary.csv`
- `physical_partition_sensitivity.csv`

