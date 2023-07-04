#  Local Interpretable Model-Agnostic Explanations for Ranking Model Interpretability

## Problem statement
Learning-to-rank (LTR) models are often complex since they are trained
using many parameters to achieve high accuracy. The complexity of ranking
models can sometimes undermine their efficacy, as humans struggle to comprehend the rationale behind a particular order. So-called “black boxes” often
become a problem due to the difficulty of understanding the factors contributing to a model’s output. The absence of transparency can derive to unwanted
issues as it may result in errors, biases, or even unethical behavior. Hence, the
urge to develop interpretable LTR models.

Currently, the state-of-the-art for interpretability of learning-to-rank models is
led by generalized additive models (GAMs). Zhuang et al. presented how
to extend GAMs into ranking models and proposed a novel formulation of them.
The study introduced the notation of neural ranking GAMs based on neural
networks by effectively leveraging list-level context features. The conducted
experiments demonstrated that ranking GAMs outperformed traditional GAMs
while preserving interpretability. These findings have made ranking GAMs the
standard method applied in LTR tasks since they are intrinsically interpretable
models with transparent and self-explainable structures.

Nevertheless, ranking GAMs have some limitations that compromise their
successful incorporation in production settings. For instance, they can be computationally intensive, limiting their scalability in production environments requiring real-time ranking. Additionally, GAMs can struggle to handle highdimensional data as the number of parameters increases with the number of predictors. This can lead to overfitting and poor generalization of unseen data.
In front of these drawbacks, previous alternatives based on post-hoc methods  could be improved to potentially offer a more scalable and efficient solution for real-time ranking.

The present study identifies a significant gap in the literature concerning the capabilities of post-hoc methods, particularly those associated with local interpretable model-agnostic explanations (LIME), compared to novel ranking GAMs. Despite the growing research on this topic, no studies provide a quantitative evaluation of post-hoc techniques’ efficacy compared to state-of-the-art
interpretable ranking methods.
