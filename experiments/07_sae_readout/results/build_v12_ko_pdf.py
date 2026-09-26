#!/usr/bin/env python3
"""Convert V12 Korean study MD to LaTeX PDF via XeLaTeX with Korean fonts."""
import re, subprocess
from pathlib import Path

MD_PATH = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v12_cross_model_study_ko.md")
TEX_PATH = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v12_cross_model_study_ko.tex")
PDF_PATH = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v12_cross_model_study_ko.pdf")
FIG_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/figures")


def escape_latex(text):
    if '\\' in text and any(cmd in text for cmd in ['\\textbf', '\\textit', '\\texttt',
                                                      '\\item', '\\section', '\\subsection',
                                                      '\\begin', '\\end', '\\includegraphics',
                                                      '\\caption', '\\centering']):
        return text
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    text = text.replace('#', '\\#')
    text = text.replace('_', '\\_')
    return text


def convert_table(lines):
    if len(lines) < 2:
        return ''
    header = [c.strip() for c in lines[0].split('|')[1:-1]]
    n_cols = len(header)
    if n_cols == 0:
        return ''
    col_spec = '|' + '|'.join(['l'] * n_cols) + '|'
    tex = ['\\begin{table}[H]', '\\centering', '\\small',
           f'\\begin{{tabular}}{{{col_spec}}}', '\\hline']
    tex.append(' & '.join([escape_latex(h) for h in header]) + ' \\\\')
    tex.append('\\hline')
    for line in lines[2:]:
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) == n_cols:
            tex.append(' & '.join([escape_latex(c) for c in cells]) + ' \\\\')
    tex.extend(['\\hline', '\\end{tabular}', '\\end{table}'])
    return '\n'.join(tex)


def md_to_latex(md_text):
    lines = md_text.split('\n')
    tex_lines = []
    in_table = False
    in_code = False
    table_lines = []

    for line in lines:
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

        if line.startswith('# ') and not line.startswith('## '):
            tex_lines.append(f'\\section*{{{escape_latex(line[2:].strip())}}}')
            continue
        if line.startswith('## '):
            tex_lines.append(f'\\section{{{escape_latex(line[3:].strip())}}}')
            continue
        if line.startswith('### '):
            tex_lines.append(f'\\subsection{{{escape_latex(line[4:].strip())}}}')
            continue

        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if img_match:
            caption, path = img_match.groups()
            if 'figures/' in path:
                fig_name = path.split('figures/')[-1]
                full_path = str(FIG_DIR / fig_name)
            else:
                full_path = path
            tex_lines.extend([
                '\\begin{figure}[H]', '\\centering',
                f'\\includegraphics[width=\\textwidth]{{{full_path}}}',
                f'\\caption{{{escape_latex(caption)}}}',
                '\\end{figure}'
            ])
            continue

        line = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', line)
        line = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', line)
        line = re.sub(r'`([^`]+)`', r'\\texttt{\1}', line)

        if line.strip() == '---':
            tex_lines.append('\\bigskip\\hrule\\bigskip')
            continue
        if line.strip().startswith('- '):
            tex_lines.append(f'\\item {escape_latex(line.strip()[2:])}')
            continue
        num_match = re.match(r'(\d+)\. (.+)', line.strip())
        if num_match:
            tex_lines.append(f'\\item {escape_latex(num_match.group(2))}')
            continue

        tex_lines.append(escape_latex(line))

    if in_table:
        tex_lines.append(convert_table(table_lines))

    result = '\n'.join(tex_lines)
    # Wrap items in itemize
    lines_out = result.split('\n')
    final = []
    in_items = False
    for line in lines_out:
        if line.strip().startswith('\\item '):
            if not in_items:
                final.append('\\begin{itemize}')
                in_items = True
            final.append(line)
        else:
            if in_items:
                final.append('\\end{itemize}')
                in_items = False
            final.append(line)
    if in_items:
        final.append('\\end{itemize}')
    return '\n'.join(final)


def build_tex(body):
    preamble = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{float}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{fontspec}
\usepackage{kotex}

\setmainfont{Noto Sans CJK KR}

\tolerance=1000
\emergencystretch=3em
\hfuzz=2pt

\title{대규모 언어모델의 위험 의사결정에 대한\\교차 모델-교차 도메인 신경 기반 분석\\[0.5em]\large V12: 방향 조향을 통한 포괄적 인과 검증}
\author{이승필, 신동현, 이윤정, 김선동\\GIST}
\date{2026년 3월 29일}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""
    return preamble + body + '\n\\end{document}\n'


def main():
    print("Reading Korean MD...")
    md_text = MD_PATH.read_text()
    md_text = re.sub(r'^# .+\n# .+\n', '', md_text)
    md_text = re.sub(r'^\*\*저자\*\*.*\n', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^\*\*작성일\*\*.*\n', '', md_text, flags=re.MULTILINE)

    print("Converting to LaTeX...")
    body = md_to_latex(md_text)
    tex = build_tex(body)
    TEX_PATH.write_text(tex)
    print(f"  Written: {TEX_PATH}")

    build_dir = TEX_PATH.parent

    # Try XeLaTeX first (Korean font support)
    print("Compiling with XeLaTeX...")
    for i in range(2):
        r = subprocess.run(
            ['xelatex', '-interaction=nonstopmode', '-output-directory', str(build_dir), str(TEX_PATH)],
            capture_output=True, text=True, cwd=str(build_dir), timeout=120
        )
        if r.returncode != 0 and i == 0:
            errors = [l for l in r.stdout.split('\n') if '!' in l][:5]
            if errors:
                print(f"  Errors: {errors}")
        print(f"  Pass {i+1} done")

    if PDF_PATH.exists():
        size_mb = PDF_PATH.stat().st_size / 1024 / 1024
        print(f"\nSuccess! PDF: {PDF_PATH} ({size_mb:.1f} MB)")
    else:
        # Fallback to pdflatex
        print("XeLaTeX failed, trying pdflatex...")
        for i in range(2):
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', str(build_dir), str(TEX_PATH)],
                capture_output=True, text=True, cwd=str(build_dir), timeout=120
            )
        if PDF_PATH.exists():
            print(f"Success (pdflatex fallback): {PDF_PATH}")
        else:
            print(f"PDF not created. Check {TEX_PATH.with_suffix('.log')}")


if __name__ == '__main__':
    main()
