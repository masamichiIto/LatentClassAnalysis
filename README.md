# Latent Class Analysis(LCA) for Binary Response Data
## Introduction
In this repository, we mainly treat Latent Class Analysis(LCA) for binary responce data as a simple representative for LCA in order to explain its algorithm and its implementation in python.
LCA is used to explore a latent groups structure which determines how observed responses are generated. For example, let's assume there are "unobservable" 2 latent groups, and indviduals belonging to first group are likely respond to optimistic questions and, on the other hand, individuals belonging to another group are likely to pesimmistic questions. LCA explains how such a response difference is happen by by using unobservable(i.e. latent) groups 
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
```math
\begin{align}
f(Y_i;\pi_r) &= \prod_{j=1}^J\prod_{k=1}^K(\pi_{jrk})^{Y_{ijk}}\\
P(Y_i|\pi, p) &= \sum_{r=1}^Rp_r\prod_{j=1}^J\prod_{k=1}^K(\pi_{jrk})^{Y_{ijk}}\\
L(\pi, p) &= \prod_{i=1}^N\left(\sum_{r=1}^Rp_r\prod_{j=1}^J\prod_{k=1}^K(\pi_{jrk})^{Y_{ijk}}\right)\\
\ln L(\pi, p) &= \sum_{i=1}^N \ln \left(\sum_{r=1}^Rp_r\prod_{j=1}^J\prod_{k=1}^K(\pi_{jrk})^{Y_{ijk}}\right)\\
\hat{P}(r|Y_i) &= \frac{\hat{p}_r f(Y_i;\hat{\pi}_r)}{\sum_{q=1}^R \hat{p}_qf(Y_i;\hat{\pi}_q)}
\end{align}
```

## Parameters Estimation via EM Algorithm

## References
- https://en.wikipedia.org/wiki/Latent_class_model
- https://hummedia.manchester.ac.uk/institutes/methods-manchester/docs/lca.pdf
- https://www.youtube.com/watch?v=SOZ0tDvTR58
- https://www.youtube.com/watch?v=btUaJd35hYE
- https://www.stat.cmu.edu/~brian/720/week08/14-lca2.pdf
- http://www.stat.unipg.it/bacci/Erasmus/Lecture_01.pdf
- (https://www.jstage.jst.go.jp/article/psycholres1954/14/2/14_2_87/_pdf/-char/ja)
- https://pop.princeton.edu/sites/g/files/toruqf496/files/documents/2020Jan_LatentClassAnalysis.pdf
- https://arxiv.org/pdf/1705.03864v5.pdf
    - https://github.com/danieledurante/nEM

## Usage
- [Paste link to Notebook here]
