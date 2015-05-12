PDFLATEX = pdflatex --halt-on-error --interaction=nonstopmode --file-line-error --shell-es
BIBER = biber

all: report.pdf
report.aux: \
	intro.tex \
	skin_color.tex \
	gesture.tex \
	experiment.tex \
	discussion.tex \
  major.tex \
	fingertip.tex \
	fig2.tex \
	fig3.tex \
	fig4.tex \

report.bbl: myref.bib
%.aux %.bcf: %.tex
	$(PDFLATEX) --draftmode $<
%.bbl: %.aux %.bcf
	$(BIBER) $(basename $<)
%.pdf: %.tex %.aux %.bbl
	$(PDFLATEX) $<

