BIBTEX := $(shell command -v bibtex 2>/dev/null || command -v bibtex8 2>/dev/null)
.PHONY: all pdf check clean
all: pdf
pdf:
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	$(BIBTEX) main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
check: pdf
	! grep -E 'Citation .* undefined|Reference .* undefined|Overfull \\hbox' main.log
clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.fdb_latexmk main.fls main.pdf
