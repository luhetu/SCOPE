#!/usr/bin/env python3
"""Patch pycocotools numpy compatibility issues"""
import re

cocoeval_file = '/home3/dnrx52/anaconda3/envs/vitseg/lib/python3.7/site-packages/pycocotools/cocoeval.py'

with open(cocoeval_file, 'r') as f:
    content = f.read()

# Fix all np.round() calls
content = re.sub(
    r'np\.linspace\(([^,]+),\s*([^,]+),\s*np\.round\(([^)]+)\)\s*\+\s*1',
    r'np.linspace(\1, \2, int(np.round(\3)) + 1',
    content
)

with open(cocoeval_file, 'w') as f:
    f.write(content)

print(f'✅ Patched: {cocoeval_file}')
