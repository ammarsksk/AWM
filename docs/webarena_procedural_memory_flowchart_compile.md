# Paper Flowchart Files

Main standalone TikZ file:

```text
docs/webarena_procedural_memory_flowchart_paper.tex
```

Includable paper snippet:

```text
docs/webarena_procedural_memory_flowchart_paper_snippet.tex
```

Compile the standalone figure with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error docs/webarena_procedural_memory_flowchart_paper.tex
```

If compiling from inside `docs/`:

```bash
cd docs
pdflatex -interaction=nonstopmode -halt-on-error webarena_procedural_memory_flowchart_paper.tex
```

The current machine does not have `pdflatex`, `xelatex`, `lualatex`, or `tectonic`
installed, so the PDF could not be rendered locally here. The source is designed
to compile directly on Overleaf or any TeX Live installation with TikZ.

