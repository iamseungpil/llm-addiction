#!/usr/bin/env python3
"""Convert V13 comprehensive study MD to LaTeX PDF."""
import re, os, subprocess
from pathlib import Path

MD_PATH = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v13_comprehensive_study.md")
TEX_PATH = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v13_comprehensive_study.tex")
PDF_PATH = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v13_comprehensive_study.pdf")
FIG_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/figures")

def md_to_latex(md_text):
    """Convert markdown to LaTeX body content."""
    lines = md_text.split('\n')
    tex_lines = []
    in_table = False
    in_code = False
    table_lines = []

    for line in lines:
        # Code blocks
        if line.strip().startswith('```'):
            if in_code:
                tex_lines.append('\\end{verbatim}')
                in_code = False
            else:
                tex_lines.append('\\begin{verbatim}')
                in_code = True
            continue
        if in_code:
            tex_lines.append(line)
            continue

        # Tables
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
            continue
        elif in_table:
            tex_lines.append(convert_table(table_lines))
            in_table = False
            table_lines = []

        # Headers
        if line.startswith('# ') and not line.startswith('## '):
            title = line[2:].strip()
            tex_lines.append(f'\\section*{{{escape_latex(title)}}}')
            continue
        if line.startswith('## '):
            tex_lines.append(f'\\section{{{escape_latex(line[3:].strip())}}}')
            continue
        if line.startswith('### '):
            tex_lines.append(f'\\subsection{{{escape_latex(line[4:].strip())}}}')
            continue

        # Images
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if img_match:
            caption, path = img_match.groups()
            # Make path relative
            if 'figures/' in path:
                fig_name = path.split('figures/')[-1]
                full_path = str(FIG_DIR / fig_name)
            else:
                full_path = path
            tex_lines.append('\\begin{figure}[H]')
            tex_lines.append('\\centering')
            tex_lines.append(f'\\includegraphics[width=\\textwidth]{{{full_path}}}')
            tex_lines.append(f'\\caption{{{escape_latex(caption)}}}')
            tex_lines.append('\\end{figure}')
            continue

        # Horizontal rules
        if line.strip() == '---':
            tex_lines.append('\\bigskip\\hrule\\bigskip')
            continue

        # Bullet points
        if line.strip().startswith('- '):
            tex_lines.append(f'\\item {escape_latex(line.strip()[2:])}')
            continue

        # Numbered items
        num_match = re.match(r'(\d+)\. (.+)', line.strip())
        if num_match:
            tex_lines.append(f'\\item {escape_latex(num_match.group(2))}')
            continue

        tex_lines.append(escape_latex(line))

    if in_table:
        tex_lines.append(convert_table(table_lines))

    # Wrap bullet items in itemize
    result = '\n'.join(tex_lines)
    result = wrap_items(result)
    return result


def escape_latex(text):
    """Escape special LaTeX characters, handle bold/italic, then escape rest."""
    # First: convert markdown bold/italic BEFORE escaping special chars
    text = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', text)
    text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)
    # Now escape special chars, but skip inside LaTeX commands
    result = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text) and text[i+1:].startswith('text'):
            # Find the closing brace of the LaTeX command
            brace_start = text.find('{', i)
            brace_end = text.find('}', brace_start) if brace_start != -1 else -1
            if brace_end != -1:
                result.append(text[i:brace_end+1])
                i = brace_end + 1
                continue
        if text[i] == '&':
            result.append('\\&')
        elif text[i] == '%':
            result.append('\\%')
        elif text[i] == '#':
            result.append('\\#')
        elif text[i] == '$':
            result.append('\\$')
        elif text[i] == '_':
            result.append('\\_')
        else:
            result.append(text[i])
        i += 1
    return ''.join(result)


def convert_table(lines):
    """Convert MD table to LaTeX tabular."""
    if len(lines) < 2:
        return ''

    # Parse header
    header = [c.strip() for c in lines[0].split('|')[1:-1]]
    n_cols = len(header)
    if n_cols == 0:
        return ''

    # Build column spec
    col_spec = '|' + '|'.join(['l'] * n_cols) + '|'

    tex = []
    tex.append('\\begin{table}[H]')
    tex.append('\\centering')
    tex.append('\\small')
    tex.append(f'\\begin{{tabular}}{{{col_spec}}}')
    tex.append('\\hline')

    # Header row
    header_escaped = [escape_latex(h) for h in header]
    tex.append(' & '.join(header_escaped) + ' \\\\')
    tex.append('\\hline')

    # Data rows (skip separator line)
    for line in lines[2:]:
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) == n_cols:
            cells_escaped = [escape_latex(c) for c in cells]
            tex.append(' & '.join(cells_escaped) + ' \\\\')

    tex.append('\\hline')
    tex.append('\\end{tabular}')
    tex.append('\\end{table}')
    return '\n'.join(tex)


def wrap_items(text):
    """Wrap consecutive \\item lines in itemize environment."""
    lines = text.split('\n')
    result = []
    in_items = False
    for line in lines:
        if line.strip().startswith('\\item '):
            if not in_items:
                result.append('\\begin{itemize}')
                in_items = True
            result.append(line)
        else:
            if in_items:
                result.append('\\end{itemize}')
                in_items = False
            result.append(line)
    if in_items:
        result.append('\\end{itemize}')
    return '\n'.join(result)


def build_tex(body):
    """Wrap body in full LaTeX document."""
    preamble = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{float}
\usepackage{hyperref}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{xcolor}

\tolerance=1000
\emergencystretch=3em
\hfuzz=2pt

\title{Neural Basis of Risky Decision-Making in LLMs\\[0.5em]\large V13: Comprehensive Evidence Report}
\author{Seungpil Lee, Donghyeon Shin, Yunjeong Lee, Sundong Kim\\GIST}
\date{March 30, 2026}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""
    return preamble + body + '\n\\end{document}\n'


def main():
    print("Reading MD...")
    md_text = MD_PATH.read_text()

    # Remove the title line (will use \maketitle)
    md_text = re.sub(r'^# .+\n', '', md_text)
    md_text = re.sub(r'^\*\*Authors\*\*.*\n', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^\*\*Date\*\*.*\n', '', md_text, flags=re.MULTILINE)

    print("Converting to LaTeX...")
    body = md_to_latex(md_text)

    print("Building document...")
    tex = build_tex(body)
    TEX_PATH.write_text(tex)
    print(f"  Written: {TEX_PATH}")

    # Compile
    build_dir = TEX_PATH.parent
    print("Compiling PDF (pass 1)...")
    for i in range(2):
        r = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-output-directory', str(build_dir), str(TEX_PATH)],
            capture_output=True, text=True, cwd=str(build_dir), timeout=120
        )
        if r.returncode != 0 and i == 0:
            # Check for common errors
            errors = [l for l in r.stdout.split('\n') if '!' in l and 'Emergency' not in l]
            if errors:
                print(f"  Errors: {errors[:5]}")
        print(f"  Pass {i+1} done")

    if PDF_PATH.exists():
        size_mb = PDF_PATH.stat().st_size / 1024 / 1024
        print(f"\nSuccess! PDF: {PDF_PATH} ({size_mb:.1f} MB)")
    else:
        print(f"\nPDF not created. Check {TEX_PATH.with_suffix('.log')}")


if __name__ == '__main__':
    main()
