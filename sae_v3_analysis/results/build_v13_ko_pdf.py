#!/usr/bin/env python3
"""Convert V13 Korean study MD to LaTeX PDF via XeLaTeX."""
import re, subprocess
from pathlib import Path

MD_PATH = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v13_comprehensive_study_ko.md")
TEX_PATH = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v13_comprehensive_study_ko.tex")
PDF_PATH = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v13_comprehensive_study_ko.pdf")
FIG_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/figures")

def escape_latex(text):
    """Escape special LaTeX characters, handle bold/italic."""
    text = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', text)
    text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)
    result = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text) and text[i+1:].startswith('text'):
            brace_start = text.find('{', i)
            brace_end = text.find('}', brace_start) if brace_start != -1 else -1
            if brace_end != -1:
                result.append(text[i:brace_end+1])
                i = brace_end + 1
                continue
        if text[i] == '&': result.append('\\&')
        elif text[i] == '%': result.append('\\%')
        elif text[i] == '#': result.append('\\#')
        elif text[i] == '$': result.append('\\$')
        elif text[i] == '_': result.append('\\_')
        else: result.append(text[i])
        i += 1
    return ''.join(result)

def convert_table(lines):
    if len(lines) < 2: return ''
    header = [c.strip() for c in lines[0].split('|')[1:-1]]
    n_cols = len(header)
    if n_cols == 0: return ''
    col_spec = '|' + '|'.join(['l'] * n_cols) + '|'
    tex = ['\\begin{table}[H]', '\\centering', '\\small', f'\\begin{{tabular}}{{{col_spec}}}', '\\hline']
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
    in_table = False; in_code = False; table_lines = []
    for line in lines:
        if line.strip().startswith('```'):
            if in_code: tex_lines.append('\\end{verbatim}'); in_code = False
            else: tex_lines.append('\\begin{verbatim}'); in_code = True
            continue
        if in_code: tex_lines.append(line); continue
        if '|' in line and line.strip().startswith('|'):
            if not in_table: in_table = True; table_lines = []
            table_lines.append(line); continue
        elif in_table:
            tex_lines.append(convert_table(table_lines)); in_table = False; table_lines = []
        if line.startswith('# ') and not line.startswith('## '):
            tex_lines.append(f'\\section*{{{escape_latex(line[2:].strip())}}}'); continue
        if line.startswith('## '):
            raw = re.sub(r'^\d+\.\s+', '', line[3:].strip())
            tex_lines.append(f'\\section{{{escape_latex(raw)}}}'); continue
        if line.startswith('### '):
            raw = re.sub(r'^\d+\.\d+\s+', '', line[4:].strip())
            tex_lines.append(f'\\subsection{{{escape_latex(raw)}}}'); continue
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if img_match:
            caption, path = img_match.groups()
            full_path = str(FIG_DIR / path.split('figures/')[-1]) if 'figures/' in path else path
            tex_lines.extend(['\\begin{figure}[H]', '\\centering',
                f'\\includegraphics[width=\\textwidth]{{{full_path}}}',
                f'\\caption{{{escape_latex(caption)}}}', '\\end{figure}'])
            continue
        if line.strip() == '---': tex_lines.append('\\bigskip\\hrule\\bigskip'); continue
        if line.strip().startswith('- '):
            tex_lines.append(f'\\item {escape_latex(line.strip()[2:])}'); continue
        num_match = re.match(r'(\d+)\. (.+)', line.strip())
        if num_match:
            tex_lines.append(f'\\item {escape_latex(num_match.group(2))}'); continue
        tex_lines.append(escape_latex(line))
    if in_table: tex_lines.append(convert_table(table_lines))
    # Wrap items
    out = []; in_items = False
    for line in tex_lines:
        if line.strip().startswith('\\item '):
            if not in_items: out.append('\\begin{itemize}'); in_items = True
            out.append(line)
        else:
            if in_items: out.append('\\end{itemize}'); in_items = False
            out.append(line)
    if in_items: out.append('\\end{itemize}')
    return '\n'.join(out)

def main():
    print("Reading Korean MD...")
    md_text = MD_PATH.read_text()
    md_text = re.sub(r'^# .+\n(# .+\n)?', '', md_text)
    md_text = re.sub(r'^\*\*저자\*\*.*\n', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^\*\*작성일\*\*.*\n', '', md_text, flags=re.MULTILINE)
    
    print("Converting to LaTeX...")
    body = md_to_latex(md_text)
    
    preamble = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}\usepackage{longtable}\usepackage{graphicx}\usepackage{float}
\usepackage{hyperref}\usepackage{amsmath}\usepackage{xcolor}
\usepackage{fontspec}\usepackage{kotex}
\setmainfont{Noto Sans CJK KR}
\tolerance=1000\emergencystretch=3em\hfuzz=2pt
\title{LLM의 위험 의사결정에 대한 신경 기반 연구\\[0.5em]\large V13: 종합 증거 보고서}
\author{이승필, 신동현, 이윤정, 김선동\\GIST}
\date{2026년 3월 30일}
\begin{document}
\maketitle\tableofcontents\newpage
"""
    tex = preamble + body + '\n\\end{document}\n'
    TEX_PATH.write_text(tex)
    
    print("Compiling with XeLaTeX...")
    build_dir = TEX_PATH.parent
    for i in range(2):
        r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-output-directory', str(build_dir), str(TEX_PATH)],
            capture_output=True, text=True, cwd=str(build_dir), timeout=120)
        print(f"  Pass {i+1} done")
    
    if PDF_PATH.exists():
        print(f"\nSuccess! {PDF_PATH} ({PDF_PATH.stat().st_size/1024/1024:.1f} MB)")
    else:
        print("PDF not created. Trying pdflatex fallback...")
        for i in range(2):
            subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory', str(build_dir), str(TEX_PATH)],
                capture_output=True, text=True, cwd=str(build_dir), timeout=120)
        if PDF_PATH.exists(): print(f"Fallback success: {PDF_PATH}")

if __name__ == '__main__':
    main()
