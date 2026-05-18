PAPER = main

.PHONY: all clean watch arxiv

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib arxiv.sty $(wildcard figures/*)
	latexmk -pdf -bibtex -halt-on-error -interaction=nonstopmode $(PAPER).tex

watch:
	latexmk -pdf -bibtex -pvc -halt-on-error $(PAPER).tex

arxiv: all
	rm -f arxiv-submission.tar.gz
	tar czf arxiv-submission.tar.gz \
		$(PAPER).tex \
		$(PAPER).bbl \
		references.bib \
		arxiv.sty \
		figures/

clean:
	latexmk -C
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.fls *.fdb_latexmk \
	      *.synctex.gz arxiv-submission.tar.gz
