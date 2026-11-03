#!/usr/bin/env python3
"""Verify that extracted resource files match the distributed SHA256SUMS.txt."""
from pathlib import Path
import hashlib
import sys

root=Path(__file__).resolve().parent
manifest=root/'SHA256SUMS.txt'
expected={}
for lineno, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(),1):
    if not line.strip():
        continue
    if len(line)<67 or line[64:66]!='  ':
        raise SystemExit(f'Malformed checksum manifest line {lineno}')
    digest=line[:64]
    rel=line[66:]
    if not rel or rel.startswith('/') or '..' in Path(rel).parts or rel in expected:
        raise SystemExit(f'Unsafe or duplicate checksum manifest entry line {lineno}')
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise SystemExit(f'Invalid digest on line {lineno}')
    expected[rel]=digest

actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
missing=set(expected)-actual
extra=actual-set(expected)
fail=[]
for rel,digest in expected.items():
    p=root/rel
    if not p.is_file():
        continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024),b''):
            h.update(chunk)
    if h.hexdigest()!=digest:
        fail.append(rel)
print(f'Checksum audit: {len(expected)} entries, {len(missing)} missing, {len(extra)} extra, {len(fail)} mismatched')
for kind,items in [('MISSING',missing),('EXTRA',extra),('MISMATCH',fail)]:
    for item in sorted(items): print(f'  {kind}: {item}')
if missing or extra or fail:
    sys.exit(1)
print('PASS: all resource files match SHA256SUMS.txt')
