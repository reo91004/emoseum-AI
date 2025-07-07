#!/usr/bin/env python3
"""
감정 기반 디지털 치료 이미지 생성 시스템 - CLI 인터페이스
- SD-1.5 기반 경량화 이미지 생성
- VAD 모델 기반 완벽한 감정 분석
- LoRA 개인화 어댑터
- DRaFT+ 강화학습
- CLI 기반 터미널 인터페이스
"""

import sys
import argparse
import logging
from pathlib import Path

# 모듈 임포트
from config import device, logger
from core.therapy_system import EmotionalImageTherapySystem

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("emotion_therapy.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


def check_system_requirements():
    """시스템 요구사항 및 라이브러리 설치 상태 확인"""
    import torch
    import numpy as np
    from emoseum.config import (
        TRANSFORMERS_AVAILABLE,
        DIFFUSERS_AVAILABLE,
        PEFT_AVAILABLE,
    )

    print("🔍 시스템 요구사항 확인")
    print("=" * 50)

    # Python 버전
    python_version = sys.version_info
    print(
        f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}"
    )

    # 디바이스 정보
    print(f"🔧 디바이스: {device} ({device.type})")

    if device.type == "mps":
        print("🍎 Apple Silicon 최적화 활성화")
    elif device.type == "cuda":
        print(f"🚀 CUDA 가능 (GPU: {torch.cuda.get_device_name()})")
    else:
        print("💻 CPU 모드")

    # 메모리 정보
    if device.type == "mps":
        print("💾 통합 메모리 (Apple Silicon)")
    elif device.type == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 GPU 메모리: {gpu_memory:.1f}GB")

    # 라이브러리 상태
    print("\n📚 라이브러리 상태:")
    libraries = {
        "PyTorch": torch.__version__,
        "Transformers": TRANSFORMERS_AVAILABLE,
        "Diffusers": DIFFUSERS_AVAILABLE,
        "PEFT": PEFT_AVAILABLE,
        "PIL": True,  # 기본 라이브러리
        "NumPy": np.__version__,
    }

    for lib, status in libraries.items():
        if isinstance(status, bool):
            status_str = "✅ 설치됨" if status else "❌ 미설치"
        else:
            status_str = f"✅ v{status}"
        print(f"  • {lib}: {status_str}")

    # 설치 권장사항
    missing_libs = []
    if not TRANSFORMERS_AVAILABLE:
        missing_libs.append("transformers")
    if not DIFFUSERS_AVAILABLE:
        missing_libs.append("diffusers")
    if not PEFT_AVAILABLE:
        missing_libs.append("peft")

    if missing_libs:
        print(f"\n⚠️ 누락된 라이브러리: {', '.join(missing_libs)}")
        print("설치 명령어:")
        print(f"pip install {' '.join(missing_libs)}")
    else:
        print("\n✅ 모든 필수 라이브러리가 설치되어 있습니다!")

    # 디렉토리 상태
    print(f"\n📁 작업 디렉토리:")
    dirs_to_check = ["generated_images", "user_loras"]
    for dir_name in dirs_to_check:
        dir_path = Path(dir_name)
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"  • {dir_name}/: ✅ ({file_count}개 파일)")
        else:
            print(f"  • {dir_name}/: 📁 생성 예정")

    # 데이터베이스 상태
    db_path = Path("user_profiles.db")
    if db_path.exists():
        size_mb = db_path.stat().st_size / 1024 / 1024
        print(f"  • user_profiles.db: ✅ ({size_mb:.2f}MB)")
    else:
        print(f"  • user_profiles.db: 📄 생성 예정")

    print("=" * 50)


def show_usage_examples():
    """사용 예시 표시"""

    print("💡 사용 예시")
    print("=" * 50)

    examples = [
        {
            "title": "1. 기본 이미지 생성",
            "command": 'python main.py --user-id "alice" --text "오늘 하루 정말 행복했다"',
            "description": "사용자의 감정 일기를 분석하여 치료용 이미지 생성",
        },
        {
            "title": "2. 상세 프롬프트와 함께 생성",
            "command": 'python main.py --user-id "bob" --text "스트레스가 심하다" --prompt "평온한 자연 풍경"',
            "description": "기본 프롬프트에 사용자 정의 프롬프트 추가",
        },
        {
            "title": "3. 고품질 이미지 생성",
            "command": 'python main.py --user-id "carol" --text "창의적인 기분" --steps 30 --guidance 9.0',
            "description": "더 많은 스텝과 높은 가이던스로 고품질 이미지 생성",
        },
        {
            "title": "4. 피드백 제공",
            "command": 'python main.py --user-id "alice" --emotion-id 1 --feedback-score 4.5 --comments "정말 좋다"',
            "description": "생성된 이미지에 대한 피드백 제공으로 개인화 학습",
        },
        {
            "title": "5. 치료 인사이트 조회",
            "command": 'python main.py --user-id "alice" --insights',
            "description": "사용자의 감정 상태와 치료 진행 상황 확인",
        },
        {
            "title": "6. 감정 히스토리 조회",
            "command": 'python main.py --user-id "alice" --history 10',
            "description": "최근 감정 기록과 생성된 이미지 히스토리 확인",
        },
        {
            "title": "7. 시스템 정리",
            "command": 'python main.py --user-id "admin" --cleanup 30',
            "description": "30일 이상 된 이미지 파일 정리",
        },
    ]

    for example in examples:
        print(f"\n{example['title']}:")
        print(f"  💡 {example['description']}")
        print(f"  💻 {example['command']}")

    print("\n" + "=" * 50)


def main():
    """메인 CLI 인터페이스"""

    parser = argparse.ArgumentParser(
        description="감정 기반 디지털 치료 이미지 생성 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python %(prog)s --user-id "alice" --text "오늘 기분이 좋다" --prompt "자연 풍경"
  python %(prog)s --user-id "bob" --text "스트레스 받는다" --feedback-score 4.2
  python %(prog)s --user-id "carol" --insights
        """,
    )

    # 기본 인자들
    parser.add_argument("--user-id", required=True, help="사용자 ID")
    parser.add_argument("--text", help="감정 일기 텍스트")
    parser.add_argument("--prompt", default="", help="추가 프롬프트")

    # 생성 옵션들
    parser.add_argument("--steps", type=int, default=15, help="추론 스텝 수 (기본: 15)")
    parser.add_argument(
        "--guidance", type=float, default=7.5, help="가이던스 스케일 (기본: 7.5)"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="이미지 너비 (기본: 512)"
    )
    parser.add_argument(
        "--height", type=int, default=512, help="이미지 높이 (기본: 512)"
    )

    # 피드백 옵션들
    parser.add_argument("--feedback-score", type=float, help="피드백 점수 (1.0-5.0)")
    parser.add_argument("--emotion-id", type=int, help="피드백할 감정 ID")
    parser.add_argument("--comments", help="피드백 코멘트")
    parser.add_argument(
        "--no-training", action="store_true", help="피드백 시 학습 비활성화"
    )

    # 조회 옵션들
    parser.add_argument("--insights", action="store_true", help="치료 인사이트 조회")
    parser.add_argument("--history", type=int, help="감정 히스토리 조회 (개수)")

    # 시스템 옵션들
    parser.add_argument(
        "--model", default="runwayml/stable-diffusion-v1-5", help="모델 경로"
    )
    parser.add_argument("--cleanup", type=int, help="오래된 이미지 정리 (일 수)")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")

    args = parser.parse_args()

    # 로그 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 시스템 초기화
    print("🚀 감정 기반 디지털 치료 시스템 시작")
    print(f"🔧 디바이스: {device}")
    print("-" * 60)

    try:
        system = EmotionalImageTherapySystem(model_path=args.model)

        # 1. 이미지 생성 모드
        if args.text:
            print(f"👤 사용자: {args.user_id}")
            print(f"📝 입력 텍스트: {args.text}")
            print(f"🎨 프롬프트: {args.prompt}")
            print()

            result = system.generate_therapeutic_image(
                user_id=args.user_id,
                input_text=args.text,
                base_prompt=args.prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                width=args.width,
                height=args.height,
            )

            if result["success"]:
                metadata = result["metadata"]
                emotion = metadata["emotion"]

                print("✅ 이미지 생성 완료!")
                print(
                    f"😊 감정 분석: V={emotion['valence']:.3f}, A={emotion['arousal']:.3f}, D={emotion['dominance']:.3f}"
                )
                print(f"🎯 최종 프롬프트: {metadata['final_prompt']}")
                print(f"💾 저장 경로: {metadata['image_path']}")
                print(f"🆔 감정 ID: {metadata['emotion_id']} (피드백용)")
                print()

                # 이미지 표시 (가능한 경우)
                try:
                    import subprocess
                    import platform

                    if platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", metadata["image_path"]], check=False)
                        print("🖼️ 이미지를 기본 뷰어로 열었습니다.")
                    elif platform.system() == "Windows":
                        subprocess.run(
                            ["start", metadata["image_path"]], shell=True, check=False
                        )
                        print("🖼️ 이미지를 기본 뷰어로 열었습니다.")
                    elif platform.system() == "Linux":
                        subprocess.run(
                            ["xdg-open", metadata["image_path"]], check=False
                        )
                        print("🖼️ 이미지를 기본 뷰어로 열었습니다.")
                except Exception:
                    print("💡 생성된 이미지를 확인하려면 위 경로를 열어보세요.")

            else:
                print(f"❌ 이미지 생성 실패: {result['error']}")
                return 1

        # 2. 피드백 모드
        elif args.feedback_score is not None:
            if args.emotion_id is None:
                print("❌ 피드백을 위해서는 --emotion-id가 필요합니다.")
                return 1

            print(f"👤 사용자: {args.user_id}")
            print(f"🆔 감정 ID: {args.emotion_id}")
            print(f"⭐ 피드백 점수: {args.feedback_score}")
            if args.comments:
                print(f"💬 코멘트: {args.comments}")
            print()

            result = system.process_feedback(
                user_id=args.user_id,
                emotion_id=args.emotion_id,
                feedback_score=args.feedback_score,
                comments=args.comments,
                enable_training=not args.no_training,
            )

            if result["success"]:
                print("✅ 피드백 처리 완료!")
                print(f"📊 총 상호작용: {result['total_interactions']}회")
                print(f"📝 총 피드백: {result['total_feedbacks']}회")

                if result["training_performed"]:
                    training_result = result["training_result"]
                    if "total_reward" in training_result:
                        print(
                            f"🤖 개인화 학습 완료: 보상 {training_result['total_reward']:.3f}"
                        )
                    else:
                        print(
                            f"🤖 개인화 학습 완료: {training_result.get('mode', 'unknown')}"
                        )
                else:
                    print("ℹ️ 학습은 수행되지 않았습니다.")

                # 간단한 인사이트 표시
                insights = result["therapeutic_insights"]
                if "emotional_state" in insights:
                    mood = insights["emotional_state"]["current_mood"]
                    trend = insights["emotional_state"]["mood_trend"]
                    print(f"😊 현재 기분: {mood}")
                    print(f"📈 기분 트렌드: {trend:+.3f}")

            else:
                print(f"❌ 피드백 처리 실패: {result['error']}")
                return 1

        # 3. 인사이트 조회 모드
        elif args.insights:
            print(f"👤 사용자: {args.user_id}")
            print("📊 치료 인사이트 조회")
            print("-" * 40)

            insights = system.get_user_insights(args.user_id)

            if insights.get("status") == "insufficient_data":
                print("ℹ️ 충분한 데이터가 수집되지 않았습니다.")
                print("💡 더 많은 감정 일기를 작성하고 피드백을 제공해주세요.")
            else:
                # 감정 상태
                emotional_state = insights["emotional_state"]
                print(f"😊 현재 기분: {emotional_state['current_mood']}")
                print(f"📈 기분 트렌드: {emotional_state['mood_trend']:+.3f}")
                print(f"🎯 감정 안정성: {emotional_state['stability']:.3f}")
                print()

                # 진행 지표
                progress = insights["progress_indicators"]
                print("📊 진행 지표:")
                print(f"  • 참여도: {progress['engagement_level']:.1%}")
                print(f"  • 회복 지표: {progress['recovery_indicator']:.3f}")
                print(f"  • 총 상호작용: {progress['total_interactions']}회")
                print(f"  • 피드백 수: {progress['feedback_count']}회")
                print()

                # 추천사항
                recommendations = insights["recommendations"]
                print("💡 추천사항:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
                print()

                # 개인화 선호도
                preferences = insights["preference_summary"]
                print("🎨 개인화 선호도:")
                for key, value in preferences.items():
                    if isinstance(value, (int, float)):
                        print(f"  • {key}: {value:+.2f}")
                    else:
                        print(f"  • {key}: {value}")

        # 4. 히스토리 조회 모드
        elif args.history is not None:
            print(f"👤 사용자: {args.user_id}")
            print(f"📚 최근 {args.history}개 감정 히스토리")
            print("-" * 60)

            history = system.get_emotion_history(args.user_id, args.history)

            if not history:
                print("ℹ️ 감정 히스토리가 없습니다.")
            else:
                for i, record in enumerate(reversed(history), 1):
                    emotion = record["emotion"]
                    timestamp = record["timestamp"][:19].replace("T", " ")

                    print(f"[{i}] {timestamp}")
                    print(
                        f"    📝 입력: {record['input_text'][:60]}{'...' if len(record['input_text']) > 60 else ''}"
                    )
                    print(
                        f"    😊 감정: V={emotion.valence:.2f}, A={emotion.arousal:.2f}, D={emotion.dominance:.2f}"
                    )
                    if record.get("image_path"):
                        print(f"    🖼️ 이미지: {record['image_path']}")
                    print(f"    🆔 ID: {record.get('id', 'N/A')}")
                    print()

        # 5. 정리 모드
        elif args.cleanup is not None:
            print(f"🧹 {args.cleanup}일 이상 된 이미지 파일 정리")
            print("-" * 40)

            cleaned_count = system.cleanup_old_images(args.cleanup)
            print(f"✅ {cleaned_count}개 파일 정리 완료")

        # 6. 도움말 (인자가 없는 경우)
        else:
            print("❓ 사용법:")
            print()
            print("1. 이미지 생성:")
            print('   python main.py --user-id "alice" --text "오늘 기분이 좋다"')
            print()
            print("2. 피드백 제공:")
            print(
                '   python main.py --user-id "alice" --emotion-id 1 --feedback-score 4.5'
            )
            print()
            print("3. 치료 인사이트 조회:")
            print('   python main.py --user-id "alice" --insights')
            print()
            print("4. 히스토리 조회:")
            print('   python main.py --user-id "alice" --history 5')
            print()
            print("5. 도움말:")
            print("   python main.py --help")
            print()
            print("💡 자세한 옵션은 --help를 참조하세요.")

    except KeyboardInterrupt:
        print("\n⚠️ 사용자가 중단했습니다.")
        return 130
    except Exception as e:
        logger.error(f"❌ 시스템 오류: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    print("-" * 60)
    print("✅ 작업 완료")
    return 0


if __name__ == "__main__":
    # 시스템 정보 표시 (verbose 모드거나 도움말인 경우)
    if len(sys.argv) == 1 or "--help" in sys.argv or "-h" in sys.argv:
        check_system_requirements()
        print()
        show_usage_examples()
        print()

    # 메인 프로그램 실행
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        sys.exit(1)
