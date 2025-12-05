# Introduction

The core code and models in the article "Charting the Meta-Trait Landscape of Human Personality".

The primary objective of this code library is to facilitate the factor decomposition of behavioral data using [Orthogonal Projection Non-negative Matrix Factorization (OPNMF)](https://github.com/asotiras/brainparts), and to appropriately visualize the results.

All operations are implemented in MATLAB (version 2021a or later), and the code runs without requiring any additional dependencies.

# Demo Illustrations

## **1.Personality trait decomposition and evaluation**

[![A.png](https://i.postimg.cc/gk6LWcXq/A.png)](https://postimg.cc/svsgPrJv)

## 2.Two-dimensional space of personality meta-traits

[![B.png](https://i.postimg.cc/wBF3ZQcD/B.png)](https://postimg.cc/VSrYbMhN)

# Contents

## 1.Code

Functions designed to perform decomposition, evaluation, and visualization tasks efficiently.

*   fmatrix.m

    The core code for conducting OPNMF-based behavioral data dimensionality reduction (factor decomposition) and evaluation (also allowing the use of classical methods such as PAF and PCA).

*   plot\_fmatrix.m

    The core visualization code for displaying the evaluation results and the factor loading matrix.

*   plot\_factor\_loading.m

    The code for flexibly visualizing the factor loading matrix.

*   plot\_biplot.m

    The code for displaying the projection of data points into a two-dimensional space.

*   plot\_permutation.m

    The code for evaluating (and displaying) the model's fit bias across different groups (such as different gender groups).

*   plot\_alluvial\_factors.m

    The code for showing how the item's affiliation changes between different models.

*   combine\_fmatrix.m

    The code for integrating multiple evaluation results (for parallel computation).

## 2.Data

Demonstration data.

*   data\_evaluation.mat

    Demonstration data for behavioral data evaluation, evaluation and visualization.
*   data\_alluvial.mat

    Demonstration data on the trend of item affiliation changes.
*   data\_HCP\_meta2traits

    Demonstration data on two-dimensional projection (from a two-factor model trained on HCP).

## 3.Demo

Demonstration code for executing different analyses.

*   Script01\_Evaluation\_basic.m

    Basic version of data decomposition and evaluation.
*   Script02\_Model\_generalization.m

    Apply a pre-trained model to another independent sample.
*   Script03\_Evaluation\_advanced.m

    Additional strategies for model evaluation.
*   Script04\_Evaluation\_demographics.m

    Compare the fitting error when applying the model to different groups.
*   Script05\_Evaluation\_combination.m

    Integrate multiple evaluation data.
*   Script06\_Item\_fluctuations.m

    Show how item affiliation changes in different models.
*   Script07\_2D\_space.m

    Project the participant's data onto a two-dimensional behavior space.
*   Script08\_Factor\_loading\_heatmap.m

    Conduct factor analysis using classic models and flexibly visualize the factor structure.

## 4.Model

Pre-trained personality models based on the main datasets.

*   Model\_NEO\_IPIP\_120items.mat

    Personality trait models trained on [IPIP-NEO](https://ipip.ori.org/) (with factor numbers ranging from 2 to 9)
*   Model\_NEO\_FFI\_60items\_HCP\_DYA.mat

    Personality trait models trained on NEO-FFI in the [integrated HCP sample](https://www.humanconnectome.org/) (with factor numbers ranging from 2 to 9)
*   Model\_NEO\_FFI\_60items\_HCP\_Y.mat

    Personality trait models trained on NEO-FFI in the[ HCP young adult sample](https://www.humanconnectome.org/) (with factor numbers ranging from 2 to 9)
*   Model\_NEO\_BBC\_44items.mat

    Personality trait models trained on [BBC-44](https://beta.ukdataservice.ac.uk/datacatalogue/doi/?id=7656#!#1) (with factor numbers ranging from 2 to 9)

    | Dataset  | **Item N** | **Data N**               | **Age**                          |
    | :------- | :--------- | :----------------------- | :------------------------------- |
    | IPIP-NEO | 120        | 619,150 (370,892 Female) | 10-99 (Avg = 25.19; SD = 10.22)  |
    | HCP-D    | 60         | 229 (125 Female)         | 16-21 (Avg = 18.53; SD = 1.81)   |
    | HCP-Y    | 60         | 1,198 (650 Female)       | 22-37 (Avg = 28.84; SD = 3.68)   |
    | HCP-A    | 60         | 725 (406 Female)         | 36-100 (Avg = 59.89; SD = 15.75) |
    | BBC      | 44         | 386,375 (247,551 Female) | 18-130 (Avg = 35.97; SD = 13.86) |

## 5.Utility

Utility functions that allow for the convenient construction of custom analysis frameworks.

### Factor loading congruence

*   get\_ci.m

    Calculate the Concordance Index (CI) between two weight matrices.
*   get\_rmse.m

    Computes the root mean square error between V (items × subject) and the projection of V onto W.
*   get\_re.m

    Computes the mean absolute reconstruction error between V (items × subject) and its projection onto W (items × dimension).

### Item assignment congruence

*   get\_ps.m

    Computes various similarity metrics between two partitions (including both overall and item-level measures).

### Vector projection

*   get\_projection.m

    Project the data points, represented as vectors within the meta-trait space, onto a specific angle.
*   get\_rotation.m

    Project the vectors by rotating the coordinate axes.

# Reference

> Kaixiang, Z., Chen, J., Jinfeng, H., Cheng, W., Jiang, Q., Jianfen, F., Eickhoff, S. & Vatansever, D. (2025) Charting the Big Two landscape of human personality.
>
> Chen, J., Patil, K. R., Weis, S., Sim, K., Nickl-Jockschat, T., Zhou, J., ... & Visser, E. (2020). Neurobiological divergence of the positive and negative schizophrenia subtypes identified on a new factor structure of psychopathology using non-negative factorization: an international machine learning study. *Biological psychiatry*.

