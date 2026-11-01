#!/usr/bin/env python3
"""Build the checked-in Russian manual from Markdown with Pandoc/XeLaTeX.

No BEM jobs or data collection are started. PANDOC and PANDOC_DATA_DIR can
point to a local tool installation; default tools are resolved from PATH.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    pandoc = os.environ.get('PANDOC') or shutil.which('pandoc')
    latex = shutil.which('xelatex')
    if not pandoc or not latex:
        raise SystemExit('Install Pandoc and XeLaTeX (with DejaVu fonts) first.')
    with tempfile.TemporaryDirectory(prefix='bem-manual-') as directory:
        temp = Path(directory)
        header = temp/'header.tex'
        header.write_text(r'''
\usepackage{fvextra}
\RecustomVerbatimEnvironment{verbatim}{Verbatim}{breaklines=true,fontsize=\small}
\usepackage{xurl}
\usepackage{etoolbox}
\usepackage[section]{placeins}
\AtBeginEnvironment{longtable}{\small}
\makeatletter
\def\maxheight{\ifdim\Gin@nat@height>.72\textheight .72\textheight\else\Gin@nat@height\fi}
\makeatother
\setlength{\emergencystretch}{3em}
\sloppy
\raggedbottom
\AtBeginDocument{\hypersetup{colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!55!black}}
''')
        command = [str(pandoc), 'MANUAL.md', '--standalone', '--toc',
                   '--toc-depth=2', '--from=markdown', '--to=latex',
                   '--no-highlight', '--lua-filter=docs/manual.lua',
                   '--include-in-header='+str(header),
                   '-V', 'documentclass=report', '-V', 'fontsize=11pt',
                   '-V', 'papersize=a4', '-V', 'geometry:margin=22mm',
                   '-V', 'mainfont=DejaVu Serif', '-V', 'sansfont=DejaVu Sans',
                   '-V', 'monofont=DejaVu Sans Mono', '-o', 'MANUAL.tex']
        if os.environ.get('PANDOC_DATA_DIR'):
            command += ['--data-dir='+os.environ['PANDOC_DATA_DIR']]
        subprocess.run(command, cwd=ROOT, check=True)
        # Use a temporary build directory so a documentation rebuild leaves
        # only the intended .tex/.pdf outputs in the source tree.
        for _ in range(2):
            result = subprocess.run([latex, '-interaction=nonstopmode',
                '-halt-on-error', '-output-directory='+str(temp), 'MANUAL.tex'],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                print(result.stdout[-12000:])
                raise SystemExit(result.returncode)
        log = (temp/'MANUAL.log').read_text(errors='replace')
        warnings = [line for line in log.splitlines()
                    if 'Overfull' in line or 'Missing character' in line]
        for line in warnings:
            print(line)
        shutil.copyfile(temp/'MANUAL.pdf', ROOT/'MANUAL.pdf')
    print('Built MANUAL.tex and MANUAL.pdf from MANUAL.md')


if __name__ == '__main__':
    main()
