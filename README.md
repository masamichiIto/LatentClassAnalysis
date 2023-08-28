# Latent Class Analysis(LCA)
## Introduction

## Model

## Log Likelihood
$$
\begin{align}
f(Y_i;\pi_r) &= \prod_{j=1}^J\prod_{k=1}^K(\pi_{jrk})^{Y_{ijk}}\\
P(Y_i|\pi, p) &= \sum_{r=1}^Rp_r\prod_{j=1}^J\prod_{k=1}^K(\pi_{jrk})^{Y_{ijk}}\\
L(\pi, p) &= \prod_{i=1}^N\left(\sum_{r=1}^Rp_r\prod_{j=1}^J\prod_{k=1}^K(\pi_{jrk})^{Y_{ijk}}\right)\\
\ln L(\pi, p) &= \sum_{i=1}^N \ln \left(\sum_{r=1}^Rp_r\prod_{j=1}^J\prod_{k=1}^K(\pi_{jrk})^{Y_{ijk}}\right)\\
\hat{P}(r|Y_i) &= \frac{\hat{p}_r f(Y_i;\hat{\pi}_r)}{\sum_{q=1}^R \hat{p}_qf(Y_i;\hat{\pi}_q)}
\end{align}
$$

## Parameters Estimation via EM Algorithm

## References
- https://en.wikipedia.org/wiki/Latent_class_model
- https://hummedia.manchester.ac.uk/institutes/methods-manchester/docs/lca.pdf
- https://www.youtube.com/watch?v=SOZ0tDvTR58
- https://www.stat.cmu.edu/~brian/720/week08/14-lca2.pdf
- http://www.stat.unipg.it/bacci/Erasmus/Lecture_01.pdf
- (https://www.jstage.jst.go.jp/article/psycholres1954/14/2/14_2_87/_pdf/-char/ja)

## Usage
- [Paste link to Notebook here]
