TEX := main.tex
PDF := main.pdf

.PHONY: all pdf clean check

all: pdf

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

check:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)
	@grep -E "(LaTeX Warning:|Package .* Warning:|Overfull \\hbox|Undefined control sequence|Citation .* undefined|Reference .* undefined)" main.log || true

clean:
	latexmk -C
	rm -f *.bbl *.blg *.run.xml *.bcf
