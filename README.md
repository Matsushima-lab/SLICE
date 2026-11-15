# SLICE

This repository implements SLICE, a causal discovery method for two categorical variables when unobserved common causes may exist.

Given paired observations of two discrete variables, SLICE compares four causal explanations: 

- `X -> Y`: `X` causes `Y`
- `X <- Y`: `Y` causes `X`
- `X ⫫ Y`: `X` and `Y` are independent
- `X <- C -> Y`: unobserved common causes influence both `X` and `Y`

SLICE selects the explanation with the shortest codelength based on the MDL principle.


## Citation
If you use this repository in your research, please cite below.

```bibtex
@inbook{doi:10.1137/1.9781611979329.9,
  author = {Masatoshi Kobayashi and Shin Matsushima},
  title = {Causal Model Selection under Arbitrary Unobserved Confounders for Categorical Data with Compact Representation},
  booktitle = {Proceedings of the 2026 SIAM International Conference on Data Mining (SDM)},
  chapter = {},
  pages = {103-120},
  doi = {10.1137/1.9781611979329.9},
  URL = {https://epubs.siam.org/doi/abs/10.1137/1.9781611979329.9},
}
```