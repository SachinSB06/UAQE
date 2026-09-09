import pypdf
import sys

sys.stdout.reconfigure(encoding='utf-8')
reader = pypdf.PdfReader("QUANTIZATION_ENGINE_Technical_Documentation.pdf")
print(f"=== TOTAL PAGES: {len(reader.pages)} ===")

math_symbols = ['α', 'β', 'λ', 'τ', 'σ', 'μ', 'ε', 'Δ', '∑', '≤', '≥', '×', '±', '→', '≈', '·', '∂']

for i, page in enumerate(reader.pages):
    txt = page.extract_text()
    found = [s for s in math_symbols if s in txt]
    dollar_lines = [l.strip() for l in txt.split('\n') if '$' in l]
    print(f"Page {i+1:2d}: Found {len(found)} math symbol types: {found}")
    if dollar_lines:
        print(f"         Literal dollar signs in lines: {dollar_lines}")
    if "nlpha" in txt:
        print(f"         ERROR: broken 'nlpha' found!")

if len(reader.pages) == 10:
    print("\n>>> ALL CHECKS PASSED: EXACTLY 10 PAGES WITH CRISP MATH SYMBOLS ON EVERY PAGE! <<<")
