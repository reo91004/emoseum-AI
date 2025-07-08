#!/usr/bin/env python3
"""
Emoseum 피드백 CLI 엔트리 포인트
"""

import sys
from pathlib import Path

# src 디렉토리를 Python 경로에 추가
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from cli.feedback_cli import GentleFeedbackCLI
import argparse

def main():
    parser = argparse.ArgumentParser(description="Emoseum 피드백 수집 시스템")
    parser.add_argument("--user-id", required=True, help="사용자 ID")
    parser.add_argument("--emotion-id", type=int, required=True, help="감정 ID")
    
    args = parser.parse_args()
    
    cli = GentleFeedbackCLI()
    cli.collect_gentle_feedback(args.user_id, args.emotion_id)

if __name__ == "__main__":
    main()