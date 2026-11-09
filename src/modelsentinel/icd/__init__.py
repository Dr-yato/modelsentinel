"""Per-language static rules for generated code (non-Python languages).

Rule codes are CWE ids. Severities follow ``codeanalyzer``: a finding of
medium or above marks the code insecure. The categories follow the public
CyberSecEval Insecure Code Detector description (Bhatt et al., 2023,
arXiv:2312.04724); the rules themselves are ours.
"""

from modelsentinel.icd.c import analyze_c
from modelsentinel.icd.golang import analyze_go
from modelsentinel.icd.javascript import analyze_js
from modelsentinel.icd.php import analyze_php
from modelsentinel.icd.sql import analyze_sql

__all__ = ["analyze_c", "analyze_go", "analyze_js", "analyze_php", "analyze_sql"]
