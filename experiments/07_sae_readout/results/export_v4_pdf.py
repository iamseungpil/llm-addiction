#!/usr/bin/env python3
"""Export V4 interim report to PDF via LaTeX."""

import subprocess
import os
import re

REPORT_MD = "/home/jovyan/llm-addiction/sae_v3_analysis/results/sae_v4_interim_report.md"
FIGURES_DIR = "/home/jovyan/llm-addiction/sae_v3_analysis/results/figures"
OUTPUT_DIR = "/home/jovyan/llm-addiction/sae_v3_analysis/results"
TEX_FILE = os.path.join(OUTPUT_DIR, "sae_v4_interim_report.tex")
PDF_FILE = os.path.join(OUTPUT_DIR, "sae_v4_interim_report.pdf")


def md_to_latex(md_text):
    """Convert markdown to LaTeX body content."""
    lines = md_text.split('\n')
    latex_lines = []
    in_table = False
    in_code = False
    code_lang = ""
    table_rows = []

    for line in lines:
        # Code blocks
        if line.strip().startswith('```'):
            if in_code:
                in_code = False
                latex_lines.append("\\end{verbatim}")
                continue
            else:
                in_code = True
                code_lang = line.strip()[3:]
                latex_lines.append("\\begin{verbatim}")
                continue

        if in_code:
            latex_lines.append(line)
            continue

        # Skip title (handled separately)
        if line.startswith('# ') and not line.startswith('## '):
            continue

        # Headers
        if line.startswith('## '):
            if in_table:
                latex_lines.append(flush_table(table_rows))
                table_rows = []
                in_table = False
            text = escape_latex(line[3:].strip())
            latex_lines.append(f"\\section{{{text}}}")
            continue
        if line.startswith('### '):
            if in_table:
                latex_lines.append(flush_table(table_rows))
                table_rows = []
                in_table = False
            text = escape_latex(line[4:].strip())
            latex_lines.append(f"\\subsection{{{text}}}")
            continue

        # Horizontal rule
        if line.strip() == '---':
            if in_table:
                latex_lines.append(flush_table(table_rows))
                table_rows = []
                in_table = False
            latex_lines.append("\\bigskip\\hrule\\bigskip")
            continue

        # Images
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line.strip())
        if img_match:
            if in_table:
                latex_lines.append(flush_table(table_rows))
                table_rows = []
                in_table = False
            caption = escape_latex(img_match.group(1))
            path = img_match.group(2)
            # Convert relative path to absolute
            if not path.startswith('/'):
                path = os.path.join(FIGURES_DIR, os.path.basename(path))
            latex_lines.append("\\begin{figure}[htbp]")
            latex_lines.append("\\centering")
            latex_lines.append(f"\\includegraphics[width=\\textwidth]{{{path}}}")
            latex_lines.append(f"\\caption{{{caption}}}")
            latex_lines.append("\\end{figure}")
            continue

        # Tables
        if '|' in line and line.strip().startswith('|'):
            cells = [c.strip() for c in line.strip().split('|')[1:-1]]
            # Skip separator lines
            if all(set(c) <= set('-: ') for c in cells):
                continue
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(cells)
            continue
        else:
            if in_table:
                latex_lines.append(flush_table(table_rows))
                table_rows = []
                in_table = False

        # Bold
        line = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', line)
        # Italic
        line = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', line)
        # Inline code
        line = re.sub(r'`([^`]+)`', r'\\texttt{\1}', line)

        # Bullet points
        if line.strip().startswith('- '):
            text = escape_latex_partial(line.strip()[2:])
            latex_lines.append(f"\\begin{{itemize}}")
            latex_lines.append(f"\\item {text}")
            latex_lines.append(f"\\end{{itemize}}")
            continue

        # Regular text
        if line.strip():
            latex_lines.append(escape_latex_partial(line))
        else:
            latex_lines.append("")

    if in_table:
        latex_lines.append(flush_table(table_rows))

    return '\n'.join(latex_lines)


def escape_latex(text):
    """Escape LaTeX special characters."""
    text = text.replace('\\', '\\textbackslash{}')
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    text = text.replace('$', '\\$')
    text = text.replace('#', '\\#')
    text = text.replace('_', '\\_')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    text = text.replace('~', '\\textasciitilde{}')
    text = text.replace('^', '\\textasciicircum{}')
    return text


def escape_latex_partial(text):
    """Escape LaTeX but preserve already-formatted LaTeX commands."""
    # Don't escape if it contains LaTeX commands
    if '\\' in text and ('textbf' in text or 'textit' in text or 'texttt' in text or 'begin' in text):
        # Only escape & % # ~ ^ in these lines
        text = text.replace('&', '\\&')
        text = text.replace('%', '\\%')
        text = text.replace('#', '\\#')
        # Don't escape _ $ { } since they may be part of LaTeX formatting
        return text
    return escape_latex(text)


def flush_table(rows):
    """Convert table rows to LaTeX tabular."""
    if not rows:
        return ""

    n_cols = len(rows[0])
    col_spec = '|' + 'l|' * n_cols

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    for i, row in enumerate(rows):
        escaped = [escape_latex(cell) for cell in row]
        lines.append(' & '.join(escaped) + ' \\\\')
        if i == 0:
            lines.append("\\hline")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return '\n'.join(lines)


def generate_latex():
    """Generate full LaTeX document."""
    with open(REPORT_MD) as f:
        md_text = f.read()

    body = md_to_latex(md_text)

    latex_doc = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{float}
\usepackage{longtable}
\usepackage{fancyhdr}
\usepackage{titlesec}

\pagestyle{fancy}
\fancyhf{}
\rhead{V4 SAE Analysis — Interim Report}
\lhead{LLM Gambling Addiction}
\cfoot{\thepage}

\titleformat{\section}{\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{1em}{}

\title{\textbf{V4 Improved SAE Analysis}\\[0.5em]
\large Interim Report for Paper Incorporation\\[0.3em]
\normalsize ``Can Large Language Models Develop Gambling Addiction?''}
\author{Seungpil Lee, Donghyeon Shin, Yunjeong Lee, Sundong Kim\\
\small GIST (Gwangju Institute of Science and Technology)}
\date{March 8, 2026}

\begin{document}
\maketitle
\tableofcontents
\newpage

""" + body + r"""

\end{document}
"""
    return latex_doc


def main():
    print("Generating LaTeX...")
    latex = generate_latex()

    with open(TEX_FILE, 'w') as f:
        f.write(latex)
    print(f"LaTeX saved: {TEX_FILE}")

    print("Compiling PDF (pass 1)...")
    result = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', '-output-directory', OUTPUT_DIR, TEX_FILE],
        capture_output=True, text=True, timeout=120
    )

    print("Compiling PDF (pass 2 for TOC)...")
    result = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', '-output-directory', OUTPUT_DIR, TEX_FILE],
        capture_output=True, text=True, timeout=120
    )

    if os.path.exists(PDF_FILE):
        size = os.path.getsize(PDF_FILE)
        print(f"PDF generated: {PDF_FILE} ({size/1024:.0f} KB)")
    else:
        print("PDF generation failed. Trying pandoc fallback...")
        result = subprocess.run(
            ['pandoc', REPORT_MD, '-o', PDF_FILE,
             '--pdf-engine=pdflatex',
             '-V', 'geometry:margin=2cm',
             '-V', 'fontsize=11pt',
             '--toc',
             f'--resource-path={FIGURES_DIR}'],
            capture_output=True, text=True, timeout=120
        )
        if os.path.exists(PDF_FILE):
            size = os.path.getsize(PDF_FILE)
            print(f"PDF generated (pandoc): {PDF_FILE} ({size/1024:.0f} KB)")
        else:
            print(f"Pandoc also failed: {result.stderr[:500]}")

    # Cleanup aux files
    for ext in ['.aux', '.log', '.out', '.toc']:
        auxf = os.path.join(OUTPUT_DIR, f"sae_v4_interim_report{ext}")
        if os.path.exists(auxf):
            os.remove(auxf)


if __name__ == '__main__':
    main()
