#!/usr/bin/env python3
"""Render the canonical Markdown reports without changing their scientific text.

Requires pandoc, XeLaTeX, Noto Serif CJK SC and DejaVu Sans Mono. The contact report's wide tables are split by column
for legibility; all original table cells are preserved in the PDF.
No network calls are made by this script.
"""
from pathlib import Path
import argparse
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def split_contact_tables(source):
    lines = source.splitlines()
    out = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith('|'):
            out.append(lines[i])
            i += 1
            continue
        table = []
        while i < len(lines) and lines[i].startswith('|'):
            table.append([cell.strip() for cell in lines[i].strip().strip('|').split('|')])
            i += 1
        n = len(table[0])
        if n == 6:
            groups = [('Prediction error', [0, 1, 2]), ('Planning outcomes', [0, 3, 4, 5])]
        elif n == 5:
            groups = [('Executed costs by candidate budget', [0, 1, 2, 3]),
                      ('Paired budget changes', [0, 4])]
        else:
            groups = [('', list(range(n)))]
        for caption, indices in groups:
            if caption:
                out += ['', r'\Needspace{27\baselineskip}', '', '### ' + caption, '']
            if not caption:
                out += ['', r'\Needspace{27\baselineskip}', '']
            for row in table:
                out.append('| ' + ' | '.join(row[j] for j in indices) + ' |')
            out.append('')
    return '\n'.join(out) + '\n'


def render(which):
    name = 'REPORT' if which == 'contact' else 'report'
    source = ROOT / which / (name + '.md')
    output = ROOT / which / (name + '.pdf')
    text = source.read_text(encoding='utf-8')
    if which == 'contact':
        text = split_contact_tables(text)
        text = text.replace('(x_t-g)^2+0.1v_t^2+0.02a_t^2', r'$(x_t-g)^2+0.1v_t^2+0.02a_t^2$')
        text = text.replace('# Contact conditional error and planning performance in a synthetic world model', r'\section*{Contact conditional error and planning performance\\in a synthetic world model}')
    with tempfile.TemporaryDirectory(prefix='world-model-report-') as tmp:
        tmp = Path(tmp)
        src = tmp / 'render.md'
        src.write_text(text, encoding='utf-8')
        header = tmp / 'header.tex'
        header.write_text(r'''\usepackage{xurl}
\usepackage{fvextra}
\usepackage{needspace}
\DefineVerbatimEnvironment{verbatim}{Verbatim}{breaklines,fontsize=\small}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}
\setlength{\emergencystretch}{3em}
''', encoding='utf-8')
        subprocess.run([
            'pandoc', str(src), '--from=markdown+tex_math_dollars-superscript-subscript', '--standalone',
            '--pdf-engine=xelatex', '--resource-path=' + str(source.parent),
            '--include-in-header=' + str(header), '-V', 'documentclass=article',
            '-V', 'geometry:margin=0.85in', '-V', 'papersize=a4', '-V', 'fontsize=11pt',
            '-V', 'mainfont=Noto Serif CJK SC', '-V', 'monofont=DejaVu Sans Mono',
            '-V', 'colorlinks=true',
            '-V', 'urlcolor=blue', '-V', 'linkcolor=blue', '-o', str(output)
        ], check=True)
    print(output.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', choices=['theory', 'contact', 'all'], default='all')
    args = parser.parse_args()
    for item in (['theory', 'contact'] if args.report == 'all' else [args.report]):
        render(item)


if __name__ == '__main__':
    main()
