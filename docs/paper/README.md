# Paper source

LaTeX source for the FPR calibration preprint, targeting arXiv.

## Build

```
make            # build main.pdf
make watch      # rebuild on save
make arxiv      # produce arxiv-submission.tar.gz
make clean      # remove build artifacts
```

Requires `latexmk`, `pdflatex`, and `bibtex` (TeX Live or MacTeX).

## Files

- `main.tex` — paper source
- `references.bib` — bibliography
- `arxiv.sty` — [kourgeorge/arxiv-style](https://github.com/kourgeorge/arxiv-style) template, MIT licensed
- `figures/` — PNG figures

## arXiv submission

`make arxiv` bundles `main.tex`, the pre-built `main.bbl` (arXiv does not run `bibtex`), `references.bib`, `arxiv.sty`, and `figures/` into `arxiv-submission.tar.gz`. Upload that tarball to arXiv.

The markdown source is preserved at `../blog.md`.
