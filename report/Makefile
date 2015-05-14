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
	fig1.tex \
	fig2.tex \
	fig3.tex \
	fig4.tex \
	experiment.tex \
	fig_exp_1_line.tex \
	fig_exp_1_circle.tex \
	fig_exp_1_points.tex \
	fig_exp_2_line.tex \
	fig_exp_2_circle.tex \
	fig_exp_2_points.tex \
	fig_exp_2_sd.tex \
	device.tex \
	fig_line_error.tex \

report.bbl: myref.bib
%.aux %.bcf: %.tex
	$(PDFLATEX) --draftmode $<
%.bbl: %.aux %.bcf
	$(BIBER) $(basename $<)
%.pdf: %.tex %.aux %.bbl
	$(PDFLATEX) $<

.DELETE_ON_ERROR:
