PDFLATEX = pdflatex --halt-on-error --interaction=nonstopmode --file-line-error --shell-es

all: report.pdf
report.aux: \
	intro.tex \
	skin_color.tex \
	gesture.tex \
	experiment.tex \
	discussion.tex \
  major.tex \
	fingertip.tex \

%.aux: %.tex
	$(PDFLATEX) --draftmode $<

%.pdf: %.tex %.aux
	$(PDFLATEX) $<

