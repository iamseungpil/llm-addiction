#!/usr/bin/env python3
"""Convert V14 markdown report to a PDF."""

import re
import subprocess
from pathlib import Path


MD_PATH = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v14_causal_validation_study.md")
TEX_PATH = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v14_causal_validation_study.tex")
PDF_PATH = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v14_causal_validation_study.pdf")
FIG_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/figures")


def escape_latex(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", text)
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    out = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1:].startswith("text"):
            brace_start = text.find("{", i)
            brace_end = text.find("}", brace_start) if brace_start != -1 else -1
            if brace_end != -1:
                out.append(text[i:brace_end + 1])
                i = brace_end + 1
                continue
        if text[i] == "&":
            out.append("\\&")
        elif text[i] == "%":
            out.append("\\%")
        elif text[i] == "#":
            out.append("\\#")
        elif text[i] == "$":
            out.append("\\$")
        elif text[i] == "_":
            out.append("\\_")
        else:
            out.append(text[i])
        i += 1
    return "".join(out)


def convert_table(lines: list[str]) -> str:
    if len(lines) < 2:
        return ""
    header = [c.strip() for c in lines[0].split("|")[1:-1]]
    n_cols = len(header)
    if n_cols == 0:
        return ""
    col_spec = "|" + "|".join(["l"] * n_cols) + "|"
    tex = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(escape_latex(h) for h in header) + " \\\\",
        "\\hline",
    ]
    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) == n_cols:
            tex.append(" & ".join(escape_latex(c) for c in cells) + " \\\\")
    tex.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(tex)


def wrap_items(text: str) -> str:
    lines = text.split("\n")
    result = []
    in_items = False
    for line in lines:
        if line.strip().startswith("\\item "):
            if not in_items:
                result.append("\\begin{itemize}")
                in_items = True
            result.append(line)
        else:
            if in_items:
                result.append("\\end{itemize}")
                in_items = False
            result.append(line)
    if in_items:
        result.append("\\end{itemize}")
    return "\n".join(result)


def md_to_latex(md_text: str) -> str:
    lines = md_text.split("\n")
    tex_lines = []
    in_table = False
    in_code = False
    table_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                tex_lines.append("\\end{verbatim}")
                in_code = False
            else:
                tex_lines.append("\\begin{verbatim}")
                in_code = True
            continue
        if in_code:
            tex_lines.append(line)
            continue
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
            continue
        elif in_table:
            tex_lines.append(convert_table(table_lines))
            in_table = False
            table_lines = []
        if line.startswith("# ") and not line.startswith("## "):
            tex_lines.append(f"\\section*{{{escape_latex(line[2:].strip())}}}")
            continue
        if line.startswith("## "):
            raw = re.sub(r"^\d+\.\s+", "", line[3:].strip())
            tex_lines.append(f"\\section{{{escape_latex(raw)}}}")
            continue
        if line.startswith("### "):
            raw = re.sub(r"^\d+\.\d+\s+", "", line[4:].strip())
            tex_lines.append(f"\\subsection{{{escape_latex(raw)}}}")
            continue
        img_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if img_match:
            caption, path = img_match.groups()
            if "figures/" in path:
                fig_name = path.split("figures/")[-1]
                full_path = str(FIG_DIR / fig_name)
            else:
                full_path = path
            tex_lines.extend([
                "\\begin{figure}[H]",
                "\\centering",
                f"\\includegraphics[width=\\textwidth]{{{full_path}}}",
                f"\\caption{{{escape_latex(caption)}}}",
                "\\end{figure}",
            ])
            continue
        if line.strip() == "---":
            tex_lines.append("\\bigskip\\hrule\\bigskip")
            continue
        if line.strip().startswith("- "):
            tex_lines.append(f"\\item {escape_latex(line.strip()[2:])}")
            continue
        num_match = re.match(r"(\d+)\. (.+)", line.strip())
        if num_match:
            tex_lines.append(f"\\item {escape_latex(num_match.group(2))}")
            continue
        tex_lines.append(escape_latex(line))
    if in_table:
        tex_lines.append(convert_table(table_lines))
    return wrap_items("\n".join(tex_lines))


def build_tex(body: str) -> str:
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

\title{Neural Basis of Risky Decision-Making in LLMs\\[0.5em]\large V14: Rigorous Causal Validation Report}
\author{Seungpil Lee, Donghyeon Shin, Yunjeong Lee, Sundong Kim\\GIST}
\date{March 31, 2026}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""
    return preamble + body + "\n\\end{document}\n"


def main() -> None:
    md_text = MD_PATH.read_text()
    md_text = re.sub(r"^# .+\n", "", md_text)
    md_text = re.sub(r"^\*\*Authors\*\*.*\n", "", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"^\*\*Date\*\*.*\n", "", md_text, flags=re.MULTILINE)
    body = md_to_latex(md_text)
    TEX_PATH.write_text(build_tex(body))
    build_dir = TEX_PATH.parent
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(build_dir), str(TEX_PATH)],
            capture_output=True,
            text=True,
            cwd=str(build_dir),
            timeout=180,
            check=False,
        )
    if PDF_PATH.exists():
        size_mb = PDF_PATH.stat().st_size / 1024 / 1024
        print(f"Success! PDF: {PDF_PATH} ({size_mb:.1f} MB)")
    else:
        raise SystemExit(f"PDF not created. Check {TEX_PATH.with_suffix('.log')}")


if __name__ == "__main__":
    main()
