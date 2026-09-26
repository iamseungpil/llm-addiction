#!/usr/bin/env python3
"""Convert V9 study markdown to PDF via WeasyPrint with embedded figures."""
import re, base64, sys
from pathlib import Path
from weasyprint import HTML

STUDY = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v9_cross_model_study.md")
FIG_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/figures")
OUT_PDF = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sae_v9_cross_model_study.pdf")

def md_to_html(md_text):
    """Minimal markdown→HTML converter for tables, headers, bold, images, code blocks."""
    lines = md_text.split('\n')
    html_lines = []
    in_table = False
    in_code = False
    in_ul = False

    for line in lines:
        # Code blocks
        if line.strip().startswith('```'):
            if in_code:
                html_lines.append('</code></pre>')
                in_code = False
            else:
                lang = line.strip().lstrip('`').strip()
                html_lines.append(f'<pre><code class="{lang}">')
                in_code = True
            continue
        if in_code:
            html_lines.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            continue

        # Close table if needed
        if in_table and not line.strip().startswith('|'):
            html_lines.append('</tbody></table>')
            in_table = False

        # Close list if needed
        if in_ul and not line.strip().startswith('- '):
            html_lines.append('</ul>')
            in_ul = False

        # Horizontal rule
        if line.strip() == '---':
            html_lines.append('<hr>')
            continue

        # Headers
        m = re.match(r'^(#{1,4})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            html_lines.append(f'<h{level}>{_inline(m.group(2))}</h{level}>')
            continue

        # Images
        m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line.strip())
        if m:
            alt, src = m.group(1), m.group(2)
            img_path = FIG_DIR / Path(src).name if not Path(src).is_absolute() else Path(src)
            if not img_path.exists():
                img_path = FIG_DIR.parent / src
            if img_path.exists():
                b64 = base64.b64encode(img_path.read_bytes()).decode()
                html_lines.append(f'<figure><img src="data:image/png;base64,{b64}" style="max-width:100%"><figcaption>{alt}</figcaption></figure>')
            else:
                html_lines.append(f'<p><em>[Image: {alt} — {src}]</em></p>')
            continue

        # Blockquote
        if line.strip().startswith('>'):
            html_lines.append(f'<blockquote>{_inline(line.strip()[1:].strip())}</blockquote>')
            continue

        # Table
        if line.strip().startswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if all(set(c) <= set('-: ') for c in cells):
                continue  # separator row
            if not in_table:
                html_lines.append('<table><thead><tr>')
                for c in cells:
                    html_lines.append(f'<th>{_inline(c)}</th>')
                html_lines.append('</tr></thead><tbody>')
                in_table = True
            else:
                html_lines.append('<tr>')
                for c in cells:
                    html_lines.append(f'<td>{_inline(c)}</td>')
                html_lines.append('</tr>')
            continue

        # Unordered list
        if line.strip().startswith('- '):
            if not in_ul:
                html_lines.append('<ul>')
                in_ul = True
            html_lines.append(f'<li>{_inline(line.strip()[2:])}</li>')
            continue

        # Numbered list
        m = re.match(r'^(\d+)\.\s+(.*)', line.strip())
        if m:
            html_lines.append(f'<p><strong>{m.group(1)}.</strong> {_inline(m.group(2))}</p>')
            continue

        # Empty line
        if not line.strip():
            html_lines.append('')
            continue

        # Paragraph
        html_lines.append(f'<p>{_inline(line)}</p>')

    if in_table:
        html_lines.append('</tbody></table>')
    if in_ul:
        html_lines.append('</ul>')

    return '\n'.join(html_lines)


def _inline(text):
    """Process inline markdown: bold, italic, code, strikethrough."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    text = re.sub(r'~~(.+?)~~', r'<del>\1</del>', text)
    return text


CSS = """
@font-face {
    font-family: KoreanFont;
    src: url("file:///usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf");
}
@page {
    size: A4;
    margin: 2cm 1.8cm;
    @bottom-center { content: counter(page); font-size: 9pt; color: #666; }
}
body {
    font-family: KoreanFont, 'DejaVu Sans', sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #222;
}
h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 6px; margin-top: 20px; page-break-before: avoid; }
h2 { font-size: 14pt; border-bottom: 1px solid #999; padding-bottom: 4px; margin-top: 16px; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 12px; page-break-after: avoid; }
h4 { font-size: 11pt; margin-top: 10px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; page-break-inside: avoid; }
th { background: #f0f0f0; border: 1px solid #ccc; padding: 4px 6px; text-align: center; font-weight: bold; }
td { border: 1px solid #ccc; padding: 3px 6px; text-align: center; }
td:first-child, th:first-child { text-align: left; }
blockquote { border-left: 3px solid #0072B2; padding-left: 10px; color: #555; margin: 8px 0; }
pre { background: #f8f8f8; border: 1px solid #ddd; padding: 8px; font-size: 8.5pt; overflow-x: auto; page-break-inside: avoid; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; background: #f0f0f0; padding: 1px 3px; }
pre code { background: none; padding: 0; }
figure { text-align: center; margin: 12px 0; page-break-inside: avoid; }
figure img { max-width: 100%; border: 1px solid #eee; }
figcaption { font-size: 9pt; color: #666; margin-top: 4px; font-style: italic; }
hr { border: none; border-top: 1px solid #ccc; margin: 16px 0; }
strong { color: #111; }
del { color: #999; text-decoration: line-through; }
ul, ol { margin: 4px 0; padding-left: 20px; }
li { margin: 2px 0; }
"""


def main():
    print("Reading study markdown...")
    md = STUDY.read_text(encoding='utf-8')

    print("Converting to HTML...")
    body = md_to_html(md)
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>{CSS}</style>
</head><body>
{body}
</body></html>"""

    print(f"Generating PDF ({OUT_PDF})...")
    HTML(string=html).write_pdf(str(OUT_PDF))
    size_mb = OUT_PDF.stat().st_size / 1e6
    print(f"Done! {OUT_PDF} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
