# Report tooling

Everything under `report/` is regenerated from here. Run from the repository
root with the project's virtual environment and `PYTHONPATH=.`.

- `report_figures.py <out_dir>`: figures 2, 6, 7 and 8 from the run registry
  and the prediction cache (every-day layout, both items).
- `report_figures_data.py <out_dir>`: figures 3 and 4, the demand
  classification of the twenty series and the fast mover's weekly pattern
  and calendar-event effects, from the panel.
- `report_figure1.py`: figure 5, the first study's error by store. Runs in
  the first-study checkout (tag `first-study`) because that cache holds the
  weekly-layout runs; it writes into this repository's `report/figures/`.
- `ladder_tables.py <png>`: figure 1 and the ladder and appendix tables,
  printed as markdown for pasting into `report/report.md`.
- `brief_figure2.py <png>`: the brief's weekly-error-by-store figure.
- `build_pdf.py [source.md] [compact]`: markdown to PDF with reportlab. The
  brief uses `compact`.
- `build_odt.py <source.md> <out.odt> [compact]`: the same markdown to an
  editable .odt with the full-resolution figures.

Figures save at 300 dpi. The markdown subset the builders understand is
described at the top of `build_pdf.py`.
