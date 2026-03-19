"""Export V5 study document to PDF via LaTeX"""
import subprocess
import os
import re
import sys

STUDY_MD = "/home/jovyan/llm-addiction/sae_v3_analysis/results/sae_v5_comprehensive_study.md"
FIG_DIR = "/home/jovyan/llm-addiction/sae_v3_analysis/results/figures"
OUTPUT_DIR = "/home/jovyan/llm-addiction/sae_v3_analysis/results"

def md_to_latex(md_content):
    """Convert markdown to LaTeX"""
    lines = md_content.split('\n')
    latex_lines = []
    in_code = False
    in_table = False
    table_lines = []

    for line in lines:
        # Code blocks
        if line.strip().startswith('```'):
            if in_code:
                latex_lines.append('\\end{verbatim}')
                in_code = False
            else:
                lang = line.strip()[3:].strip()
                latex_lines.append('\\begin{verbatim}')
                in_code = True
            continue

        if in_code:
            latex_lines.append(line)
            continue

        # Tables
        if '|' in line and line.strip().startswith('|'):
            if '---' in line:
                continue  # Skip separator lines
            cells = [c.strip() for c in line.strip().split('|')[1:-1]]
            if not in_table:
                in_table = True
                n_cols = len(cells)
                col_spec = '|' + 'l|' * n_cols
                latex_lines.append('\\begin{table}[H]')
                latex_lines.append('\\centering')
                latex_lines.append('\\small')
                latex_lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
                latex_lines.append('\\hline')
                # Header
                escaped = [escape_latex(c) for c in cells]
                latex_lines.append(' & '.join([f'\\textbf{{{c}}}' for c in escaped]) + ' \\\\')
                latex_lines.append('\\hline')
            else:
                escaped = [escape_latex(c) for c in cells]
                latex_lines.append(' & '.join(escaped) + ' \\\\')
            continue
        else:
            if in_table:
                latex_lines.append('\\hline')
                latex_lines.append('\\end{tabular}')
                latex_lines.append('\\end{table}')
                in_table = False

        # Images
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if img_match:
            caption = escape_latex(img_match.group(1))
            path = img_match.group(2)
            # Make path absolute or relative to figures dir
            if not os.path.isabs(path):
                path = os.path.join(FIG_DIR, os.path.basename(path))
            latex_lines.append('\\begin{figure}[H]')
            latex_lines.append('\\centering')
            latex_lines.append(f'\\includegraphics[width=\\textwidth]{{{path}}}')
            latex_lines.append(f'\\caption{{{caption}}}')
            latex_lines.append('\\end{figure}')
            continue

        # Headers
        if line.startswith('# ') and not line.startswith('## '):
            title = escape_latex(line[2:].strip())
            latex_lines.append(f'\\section*{{{title}}}')
            continue
        if line.startswith('## '):
            title = escape_latex(line[3:].strip())
            latex_lines.append(f'\\section{{{title}}}')
            continue
        if line.startswith('### '):
            title = escape_latex(line[4:].strip())
            latex_lines.append(f'\\subsection{{{title}}}')
            continue
        if line.startswith('#### '):
            title = escape_latex(line[5:].strip())
            latex_lines.append(f'\\subsubsection{{{title}}}')
            continue

        # Bold
        line = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', line)
        # Italic
        line = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', line)
        # Inline code
        line = re.sub(r'`([^`]+)`', r'\\texttt{\1}', line)

        # Horizontal rules
        if line.strip() == '---':
            latex_lines.append('\\vspace{1em}\\hrule\\vspace{1em}')
            continue

        # Bullet points
        if line.strip().startswith('- '):
            content = escape_latex(line.strip()[2:])
            # Re-apply bold/italic after escaping
            content = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', content)
            latex_lines.append(f'\\begin{{itemize}}\\item {content}\\end{{itemize}}')
            continue

        # Numbered items
        num_match = re.match(r'^(\d+)\. (.+)', line.strip())
        if num_match:
            content = escape_latex(num_match.group(2))
            latex_lines.append(f'\\begin{{enumerate}}\\item[{num_match.group(1)}.] {content}\\end{{enumerate}}')
            continue

        # Regular text
        if line.strip():
            latex_lines.append(escape_latex(line))
        else:
            latex_lines.append('')

    if in_table:
        latex_lines.append('\\hline')
        latex_lines.append('\\end{tabular}')
        latex_lines.append('\\end{table}')

    return '\n'.join(latex_lines)


def escape_latex(text):
    """Escape LaTeX special characters"""
    # Don't escape if already has LaTeX commands
    if '\\' in text and ('\\textbf' in text or '\\textit' in text or '\\texttt' in text):
        return text
    replacements = [
        ('\\', '\\textbackslash{}'),
        ('&', '\\&'),
        ('%', '\\%'),
        ('$', '\\$'),
        ('#', '\\#'),
        ('_', '\\_'),
        ('{', '\\{'),
        ('}', '\\}'),
        ('~', '\\textasciitilde{}'),
        ('^', '\\textasciicircum{}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def create_latex_document(body):
    """Wrap body in full LaTeX document"""
    return r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{kotex}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage[margin=2cm]{geometry}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{longtable}

\title{V5 종합 SAE/활성화 분석: 신경 메커니즘 연구 보고서}
\author{이승필, 신동현, 이윤정, 김선동 (GIST)}
\date{2026-03-09}

\begin{document}
\maketitle
\tableofcontents
\newpage

""" + body + r"""

\end{document}
"""


def main():
    print("Reading V5 study markdown...")
    with open(STUDY_MD) as f:
        md_content = f.read()

    # Try pandoc first (best quality)
    tex_path = os.path.join(OUTPUT_DIR, "sae_v5_comprehensive_study.tex")
    pdf_path = os.path.join(OUTPUT_DIR, "sae_v5_comprehensive_study.pdf")

    print("Attempting pandoc conversion...")
    try:
        result = subprocess.run([
            'pandoc', STUDY_MD,
            '-o', pdf_path,
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=2cm',
            '-V', 'fontsize=11pt',
            '-V', 'mainfont=Noto Sans CJK KR',
            '-V', 'CJKmainfont=Noto Sans CJK KR',
            '--resource-path', FIG_DIR,
            '--toc',
            '-V', 'toc-title=목차',
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print(f"PDF generated successfully: {pdf_path}")
            return
        else:
            print(f"Pandoc failed: {result.stderr[:500]}")
    except FileNotFoundError:
        print("pandoc not found, trying alternative...")
    except subprocess.TimeoutExpired:
        print("pandoc timed out")

    # Fallback: Try with lualatex
    print("Trying pandoc with lualatex...")
    try:
        result = subprocess.run([
            'pandoc', STUDY_MD,
            '-o', pdf_path,
            '--pdf-engine=lualatex',
            '-V', 'geometry:margin=2cm',
            '-V', 'fontsize=11pt',
            '--resource-path', FIG_DIR,
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print(f"PDF generated successfully: {pdf_path}")
            return
        else:
            print(f"lualatex failed: {result.stderr[:500]}")
    except Exception as e:
        print(f"lualatex attempt failed: {e}")

    # Fallback 2: Try pdflatex
    print("Trying pandoc with pdflatex...")
    try:
        result = subprocess.run([
            'pandoc', STUDY_MD,
            '-o', pdf_path,
            '--pdf-engine=pdflatex',
            '-V', 'geometry:margin=2cm',
            '-V', 'fontsize=11pt',
            '--resource-path', FIG_DIR,
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print(f"PDF generated successfully: {pdf_path}")
            return
        else:
            print(f"pdflatex failed: {result.stderr[:500]}")
    except Exception as e:
        print(f"pdflatex attempt failed: {e}")

    # Fallback 3: weasyprint HTML
    print("Trying weasyprint HTML conversion...")
    try:
        # First convert to HTML
        html_path = os.path.join(OUTPUT_DIR, "sae_v5_comprehensive_study.html")
        result = subprocess.run([
            'pandoc', STUDY_MD,
            '-o', html_path,
            '--standalone',
            '--resource-path', FIG_DIR,
            '--self-contained',
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            # Then HTML to PDF
            result2 = subprocess.run([
                'weasyprint', html_path, pdf_path
            ], capture_output=True, text=True, timeout=120)
            if result2.returncode == 0:
                print(f"PDF generated successfully via weasyprint: {pdf_path}")
                return
            else:
                print(f"weasyprint failed: {result2.stderr[:500]}")
    except Exception as e:
        print(f"weasyprint attempt failed: {e}")

    print(f"\nAll PDF conversion methods failed.")
    print(f"Markdown saved at: {STUDY_MD}")
    print(f"You can convert manually using:")
    print(f"  pandoc {STUDY_MD} -o {pdf_path} --pdf-engine=xelatex")


if __name__ == '__main__':
    main()
