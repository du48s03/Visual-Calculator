PDFLATEX = pdflatex --halt-on-error --interaction=nonstopmode --file-line-error --shell-es

all: report.pdf
report.aux: \
	intro.tex \
	hand_detect.tex \
	gesture.tex \
	experiment.tex \
	discussion.tex \

%.aux: %.tex
	$(PDFLATEX) --draftmode $<

%.pdf: %.tex %.aux
	$(PDFLATEX) $<

