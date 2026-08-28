# Project 1 — Speaker-level linguistic colour resolution

## Scientific question

Does the colour-label vocabulary of a speaker's primary language relate to how finely that speaker divides physical colour space linguistically?

## Primary hypothesis

\(H_0\): language colour vocabulary is unrelated to speaker-level linguistic colour resolution.

\(H_1\): language colour vocabulary is related to speaker-level linguistic colour resolution.

The null is tested by a language-clustered joint Wald test of all spline-by-language-vocabulary interaction coefficients.

## Model

For speaker \(s\) and chip pair \((i,j)\),

\[Z_{sij}=1\{Y_{si}
e Y_{sj}\}.\]

The physical distance is Euclidean distance in the WCS CIELAB coordinates,

\[d_{ij}=\|\mathbf x_i-\mathbf x_j\|_2.\]

All valid unordered pairs are used. After deterministic aggregation into 30 distance bins,

\[Y_{sb}\sim\mathrm{Binomial}(N_{sb},p_{sb})\]

serves as a working mean-variance model. The conditional mean is

\[\mathrm{logit}(p_{sb})=lpha+f(d_b)+f(d_b)\log K_l,\]

where \(f\) is a pre-specified cubic B-spline with eight knots. Generalized estimating equations use language as the correlation cluster and a robust sandwich covariance.

The binomial distribution is a working distribution: raw pairs are not assumed independent because pairs overlap in chips, repeated pairs belong to the same speaker, and speakers within a language share a predictor. The robust cluster covariance is used precisely because the exact within-language dependence is not modelled.

## Hypothesis test

If there are nine spline basis functions, the language-vocabulary effect is represented by nine interaction coefficients \(\gamma_1,\ldots,\gamma_9\).

\[H_0:\gamma_1=\cdots=\gamma_9=0.\]

The test statistic is the robust Wald statistic

\[W=\hat\gamma^	op\{\mathrm{Var}(\hat\gamma)\}^{-1}\hat\gamma,\]

which is compared with a \(\chi^2_9\) reference distribution under the usual large-cluster regularity conditions.

## Outputs

- `speaker_statistics.csv`
- `language_vocabulary.csv`
- `pair_bins.csv`
- `model_coefficients.csv`
- `model_summary.txt`
- `hypothesis_test.csv`
- `language_resolution_summary.csv`
- descriptive figures and `fitted_distinction_curves.png`
