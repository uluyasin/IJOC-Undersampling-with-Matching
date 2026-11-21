# Extended results: LaTeX sources

Extended-results document of the paper "Boundary or Coverage, Stable or Optimal? A Computational Study of
Matching-Based Undersampling for Imbalanced Classification". It contains the complete experimental design and all
detailed results; it introduces no new results.

| File / folder | Content |
|---|---|
| `extended_results.tex` | Main file (article class, 11 pt, 0.65-in margins, so that the 7.2-in figures print at native size) |
| `sections/` | `e1_datasets.tex` (benchmark datasets), `e2_design.tex` (experimental design in full), `e3_results.tex` and `e3_additional.tex` (detailed results, additional tables and figures), `e4_implementation.tex` (implementation details) |
| `tables/` | All result tables (LaTeX, booktabs) |
| `figures/` | Figure files (PDF) |
| `references.bib` | Bibliography (natbib, `plainnat`) |

Compile from this folder:

```
pdflatex extended_results
bibtex extended_results
pdflatex extended_results
pdflatex extended_results
```

or `latexmk -pdf extended_results.tex`. Required LaTeX packages: geometry, fontenc, natbib, amsmath, amssymb,
booktabs, multirow, array, adjustbox, graphicx, xcolor, subcaption, algorithm (algorithms bundle), algpseudocode
(algorithmicx), hyperref.
