#!/usr/bin/env python3
"""
Emoseum 메인 엔트리 포인트
"""

import sys
from pathlib import Path

# src 디렉토리를 Python 경로에 추가
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from cli.main import main

if __name__ == "__main__":
    main()