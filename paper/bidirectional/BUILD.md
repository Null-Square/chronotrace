# Build the manuscript

Run `make -C paper check` from the native repository root, or `make check` from the standalone manuscript-source folder. A LaTeX installation must include `elsarticle`, Latin Modern, AMS mathematics, `algorithm`, `algpseudocode`, PGFPlots, `microtype`, and BibTeX.

A minimal Ubuntu setup matching the CI package selection is:

```bash
sudo apt-get install latexmk lmodern texlive-latex-base texlive-latex-extra \
  texlive-fonts-recommended texlive-bibtex-extra texlive-pictures \
  texlive-publishers texlive-science
```

Font and package binaries are not bundled in the manuscript source. The current source is editable and uses standard TeX-distribution packages. Both current and historic scripts remain testable independently of the paper build.
