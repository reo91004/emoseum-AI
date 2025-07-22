# main.py

import logging
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

from src.act_therapy_system import ACTTherapySystem

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("emoseum.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class EmoseumCLI:
    """Emoseum CLI 인터페이스"""

    def __init__(self, data_dir: str = "data", model_path: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_path or "runwayml/stable-diffusion-v1-5"
        self.therapy_system = ACTTherapySystem(
            data_dir=str(self.data_dir), model_path=model_path
        )

        self.current_user = None
        self.current_journey = None

    def run(self):
        """메인 실행 루프"""
        self._print_welcome()

        while True:
            try:
                if not self.current_user:
                    self._handle_user_selection()
                else:
                    self._handle_main_menu()
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break
            except Exception as e:
                logger.error(f"오류 발생: {e}")
                print(f"오류가 발생했습니다: {e}")

    def _print_welcome(self):
        """환영 메시지"""
        print("\n" + "=" * 60)
        print("Emoseum - ACT 기반 디지털 치료 시스템".center(60))
        print("=" * 60)
        print("감정을 시각화하고 희망을 찾아가는 여정".center(60))
        print("=" * 60 + "\n")

    def _handle_user_selection(self):
        """사용자 선택/생성"""
        print("\n=== 사용자 관리 ===")
        print("1. 기존 사용자로 로그인")
        print("2. 신규 사용자 등록")
        print("0. 종료")

        choice = input("\n선택하세요: ").strip()

        if choice == "1":
            self._login_user()
        elif choice == "2":
            self._register_user()
        elif choice == "0":
            sys.exit(0)
        else:
            print("잘못된 선택입니다.")

    def _login_user(self):
        """기존 사용자 로그인"""
        user_id = input("사용자 ID를 입력하세요: ").strip()

        user = self.therapy_system.user_manager.get_user(user_id)
        if user:
            self.current_user = user_id
            print(f"\n환영합니다, {user_id}님!")
            self._show_user_status()
        else:
            print("사용자를 찾을 수 없습니다.")

    def _register_user(self):
        """신규 사용자 등록"""
        print("\n=== 신규 사용자 등록 ===")
        user_id = input("사용할 ID를 입력하세요: ").strip()

        if not user_id:
            print("유효한 ID를 입력해주세요.")
            return

        try:
            result = self.therapy_system.onboard_new_user(user_id)
            self.current_user = user_id

            print(f"\n{user_id}님, 환영합니다!")
            print("다음 단계를 진행해주세요:")
            for step in result["next_steps"]:
                print(f"  {step}")

            # 바로 초기 설정으로 이동
            self._initial_setup()

        except Exception as e:
            print(f"등록 실패: {e}")

    def _initial_setup(self):
        """초기 설정 (심리검사 + 시각 선호도)"""
        print("\n=== 초기 설정 ===")

        # 1. 심리검사
        if input("\n심리검사를 진행하시겠습니까? (y/n): ").lower() == "y":
            self._conduct_assessment()

        # 2. 시각 선호도
        if input("\n시각 선호도를 설정하시겠습니까? (y/n): ").lower() == "y":
            self._set_visual_preferences()

    def _conduct_assessment(self):
        """심리검사 실시"""
        print("\n=== 심리검사 ===")
        print("각 항목에 대해 0-27 범위의 점수를 입력해주세요.")

        try:
            phq9 = int(input("PHQ-9 (우울증 선별도구) 점수: "))
            cesd = int(input("CES-D (우울척도) 점수: "))
            meaq = int(input("MEAQ (경험회피척도) 점수: "))
            ciss = int(input("CISS (대처방식척도) 점수: "))

            result = self.therapy_system.conduct_psychometric_assessment(
                self.current_user, phq9, cesd, meaq, ciss
            )

            print("\n=== 검사 결과 ===")
            print(f"대처 스타일: {result['coping_style']}")
            print(f"심각도: {result['severity_level']}")
            print(f"\n해석:")
            for key, value in result["interpretation"].items():
                print(f"  - {value}")
            print(f"\n권장사항:")
            for rec in result["recommendations"]:
                print(f"  - {rec}")

        except ValueError:
            print("올바른 숫자를 입력해주세요.")

    def _set_visual_preferences(self):
        """시각 선호도 설정"""
        print("\n=== 시각 선호도 설정 ===")

        # 아트 스타일
        print("\n화풍 선택:")
        print("1. painting (회화)")
        print("2. photography (사진)")
        print("3. abstract (추상화)")
        art_choice = input("선택 (1-3): ").strip()
        art_styles = ["painting", "photography", "abstract"]
        art_style = art_styles[int(art_choice) - 1] if art_choice in "123" else "painting"

        # 색감
        print("\n색감 선택:")
        print("1. warm (따뜻한)")
        print("2. cool (차가운)")
        print("3. pastel (파스텔)")
        color_choice = input("선택 (1-3): ").strip()
        color_tones = ["warm", "cool", "pastel"]
        color_tone = color_tones[int(color_choice) - 1] if color_choice in "123" else "warm"

        # 복잡도
        print("\n복잡도 선택:")
        print("1. simple (단순한)")
        print("2. balanced (균형잡힌)")
        print("3. complex (복잡한)")
        complexity_choice = input("선택 (1-3): ").strip()
        complexities = ["simple", "balanced", "complex"]
        complexity = complexities[int(complexity_choice) - 1] if complexity_choice in "123" else "balanced"

        result = self.therapy_system.set_visual_preferences(
            self.current_user, art_style, color_tone, complexity
        )

        print("\n시각 선호도가 저장되었습니다.")
        print(f"설정된 선호도: {result['preferences_set']}")

    def _handle_main_menu(self):
        """메인 메뉴"""
        print(f"\n=== 메인 메뉴 ({self.current_user}) ===")
        print("1. 새로운 감정 여정 시작")
        print("2. 나의 미술관 보기")
        print("3. 치료적 인사이트")
        print("4. 설정 변경")
        print("5. 고급 모델 관리 (Level 3)")
        print("9. 로그아웃")
        print("0. 종료")

        choice = input("\n선택하세요: ").strip()

        if choice == "1":
            self._start_emotion_journey()
        elif choice == "2":
            self._view_gallery()
        elif choice == "3":
            self._view_insights()
        elif choice == "4":
            self._change_settings()
        elif choice == "5":
            self._manage_advanced_models()
        elif choice == "9":
            self.current_user = None
            print("로그아웃되었습니다.")
        elif choice == "0":
            sys.exit(0)
        else:
            print("잘못된 선택입니다.")

    def _start_emotion_journey(self):
        """새로운 감정 여정 시작"""
        print("\n=== 새로운 감정 여정 ===")
        print("오늘의 감정을 자유롭게 기록해주세요.")
        print("(입력을 마치려면 빈 줄에서 Enter를 두 번 누르세요)")

        diary_lines = []
        empty_count = 0

        while True:
            line = input()
            if not line:
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
                diary_lines.append(line)

        diary_text = "\n".join(diary_lines).strip()

        if not diary_text:
            print("일기 내용이 비어있습니다.")
            return

        print("\n감정을 분석하고 있습니다...")

        try:
            # Step 1-2: The Moment → Reflection
            result = self.therapy_system.process_emotion_journey(
                self.current_user, diary_text
            )

            self.current_journey = result["gallery_item_id"]

            print("\n=== Reflection 이미지 생성 완료 ===")
            print(f"감정 키워드: {', '.join(result['emotion_analysis']['keywords'])}")
            print(f"이미지가 생성되었습니다: {result['reflection_image']['image_path']}")
            print(f"\n{result['guided_message']}")

            # Step 3: Defusion (방명록)
            if input("\n방명록을 작성하시겠습니까? (y/n): ").lower() == "y":
                self._write_guestbook()

        except Exception as e:
            logger.error(f"감정 여정 처리 실패: {e}")
            print(f"처리 중 오류가 발생했습니다: {e}")

    def _write_guestbook(self):
        """방명록 작성"""
        if not self.current_journey:
            print("진행 중인 여정이 없습니다.")
            return

        print("\n=== 방명록 작성 ===")
        print("생성된 이미지를 보고 떠오르는 제목을 지어주세요.")

        title = input("제목: ").strip()
        if not title:
            print("제목을 입력해주세요.")
            return

        print("\n이미지와 관련된 태그를 입력해주세요 (쉼표로 구분)")
        tags_input = input("태그: ").strip()
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]

        try:
            result = self.therapy_system.complete_guestbook(
                self.current_user, self.current_journey, title, tags
            )

            print("\n=== 방명록 작성 완료 ===")
            print(f"제목: {result['guestbook']['title']}")
            print(f"태그: {', '.join(result['guestbook']['tags'])}")
            print(f"\n{result['guided_question']}")

            # Step 4: Closure (희망 이미지)
            if input("\n희망 이미지를 생성하시겠습니까? (y/n): ").lower() == "y":
                self._create_hope_image(result["guided_question"])

        except Exception as e:
            logger.error(f"방명록 작성 실패: {e}")
            print(f"처리 중 오류가 발생했습니다: {e}")

    def _create_hope_image(self, guided_question: str):
        """희망 이미지 생성"""
        if not self.current_journey:
            print("진행 중인 여정이 없습니다.")
            return

        print("\n=== 희망 이미지 생성 ===")
        print(f"질문: {guided_question}")

        response = input("\n답변 (선택사항, Enter로 건너뛰기): ").strip()

        print("\n희망 이미지를 생성하고 있습니다...")

        try:
            result = self.therapy_system.create_hope_image(
                self.current_user, self.current_journey, response
            )

            print("\n=== 감정 여정 완료 ===")
            print(result["completion_message"])
            print("\n다음 활동:")
            for rec in result["next_recommendations"]:
                print(f"  - {rec}")

            self.current_journey = None

        except Exception as e:
            logger.error(f"희망 이미지 생성 실패: {e}")
            print(f"처리 중 오류가 발생했습니다: {e}")

    def _view_gallery(self):
        """미술관 보기"""
        print("\n=== 나의 미술관 ===")

        try:
            gallery = self.therapy_system.get_user_gallery(self.current_user, limit=10)

            if gallery["total_items"] == 0:
                print("아직 작품이 없습니다. 감정 여정을 시작해보세요!")
                return

            print(f"총 {gallery['total_items']}개의 작품")
            print("\n최근 작품들:")

            for i, item in enumerate(gallery["items"], 1):
                print(f"\n[{i}] {item['created_date']}")
                print(f"    감정: {', '.join(item['emotion_keywords'])}")
                if item["guestbook_title"]:
                    print(f"    제목: {item['guestbook_title']}")
                    print(f"    태그: {', '.join(item['guestbook_tags'])}")
                print(f"    완성도: {'완료' if item['hope_image_path'] else '진행중'}")

            # 분석 정보
            if "analytics" in gallery and gallery["analytics"]:
                analytics = gallery["analytics"]
                print(f"\n=== 갤러리 분석 ===")
                if "date_range" in analytics:
                    print(f"활동 기간: {analytics['date_range']['span_days']}일")
                if "completion_rate" in analytics:
                    print(f"완성률: {analytics['completion_rate']:.1f}%")

        except Exception as e:
            logger.error(f"갤러리 조회 실패: {e}")
            print(f"조회 중 오류가 발생했습니다: {e}")

    def _view_insights(self):
        """치료적 인사이트 보기"""
        print("\n=== 치료적 인사이트 ===")

        try:
            insights = self.therapy_system.get_therapeutic_insights(self.current_user)

            # 사용자 프로필
            profile = insights["user_profile"]
            print(f"\n회원 가입일: {profile['member_days']}일 전")
            print(f"총 작품 수: {profile['total_journeys']}개")

            # 감정 여정
            if "emotional_journey" in insights:
                trends = insights["emotional_journey"]
                if trends:
                    print("\n감정 변화 추이:")
                    for emotion, data in trends.items():
                        if "trend" in data:
                            trend_text = {
                                "improving": "개선 중",
                                "declining": "하락 중",
                                "stable": "안정적"
                            }.get(data["trend"], data["trend"])
                            print(f"  - {emotion}: {trend_text}")

            # 성장 인사이트
            if "growth_insights" in insights:
                growth = insights["growth_insights"]
                if growth:
                    print("\n성장 포인트:")
                    for insight in growth[:3]:
                        print(f"  - {insight}")

            # 권장사항
            if "recommendations" in insights:
                next_actions = insights["recommendations"]["next_actions"]
                if next_actions:
                    print("\n다음 단계 권장사항:")
                    for action in next_actions:
                        print(f"  - {action}")

            # 요약
            print(f"\n요약: {insights['summary']}")

        except Exception as e:
            logger.error(f"인사이트 조회 실패: {e}")
            print(f"조회 중 오류가 발생했습니다: {e}")

    def _change_settings(self):
        """설정 변경"""
        print("\n=== 설정 변경 ===")
        print("1. 시각 선호도 변경")
        print("2. 주기적 심리검사")
        print("0. 돌아가기")

        choice = input("\n선택하세요: ").strip()

        if choice == "1":
            self._set_visual_preferences()
        elif choice == "2":
            self._conduct_assessment()
        elif choice == "0":
            return
        else:
            print("잘못된 선택입니다.")

    def _manage_advanced_models(self):
        """고급 모델 관리 (Level 3)"""
        print("\n=== 고급 모델 관리 (Level 3) ===")

        try:
            # 준비 상태 확인
            readiness = self.therapy_system.check_advanced_training_readiness(
                self.current_user
            )

            print(f"\n총 작품 수: {readiness['total_gallery_items']}")
            print(f"완성된 여정: {readiness['complete_journeys']}")

            # LoRA 상태
            lora = readiness["lora_training"]
            print(f"\n[LoRA 개인화]")
            print(f"  준비 상태: {'준비됨' if lora['ready'] else '준비 안됨'}")
            print(f"  데이터 크기: {lora['data_size']}개")
            print(f"  예상 성능: {lora['estimated_performance']}")

            # DRaFT+ 상태
            draft = readiness["draft_training"]
            print(f"\n[DRaFT+ 강화학습]")
            print(f"  준비 상태: {'준비됨' if draft['ready'] else '준비 안됨'}")
            print(f"  데이터 크기: {draft['data_size']}개")
            print(f"  예상 성능: {draft['estimated_performance']}")

            # 권장사항
            if readiness["recommendations"]:
                print("\n권장사항:")
                for rec in readiness["recommendations"]:
                    print(f"  - {rec}")

            # 훈련 시작 옵션
            if readiness["overall_readiness"] == "ready":
                if input("\n고급 모델 훈련을 시작하시겠습니까? (y/n): ").lower() == "y":
                    self._trigger_advanced_training()

        except Exception as e:
            logger.error(f"고급 모델 관리 실패: {e}")
            print(f"처리 중 오류가 발생했습니다: {e}")

    def _trigger_advanced_training(self):
        """고급 모델 훈련 실행"""
        print("\n훈련 유형 선택:")
        print("1. LoRA만")
        print("2. DRaFT+만")
        print("3. 둘 다")

        choice = input("선택 (1-3): ").strip()
        training_types = {
            "1": "lora",
            "2": "draft",
            "3": "both"
        }
        training_type = training_types.get(choice, "both")

        print(f"\n{training_type} 모델 훈련을 시작합니다...")
        print("이 과정은 시간이 걸릴 수 있습니다.")

        try:
            result = self.therapy_system.trigger_advanced_training(
                self.current_user, training_type
            )

            if result["success"]:
                print("\n훈련이 성공적으로 완료되었습니다!")
                for model, details in result["results"].items():
                    print(f"\n[{model.upper()}]")
                    for key, value in details.items():
                        print(f"  {key}: {value}")
            else:
                print(f"\n훈련 실패: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"훈련 실행 실패: {e}")
            print(f"훈련 중 오류가 발생했습니다: {e}")

    def _show_user_status(self):
        """사용자 상태 표시"""
        try:
            stats = self.therapy_system.user_manager.get_user_stats(self.current_user)
            print(f"\n활동 기간: {stats['member_days']}일")
            print(f"총 여정 수: {stats['total_journeys']}개")

            if stats["latest_test"]:
                print(f"최근 검사: {stats['latest_test']['test_date']}")
                print(f"대처 스타일: {stats['latest_test']['coping_style']}")

            if stats["needs_periodic_test"]:
                print("\n[알림] 주기적 심리검사 시기가 되었습니다.")

        except Exception as e:
            logger.error(f"사용자 상태 조회 실패: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Emoseum - ACT 기반 디지털 치료 시스템")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="데이터 저장 디렉토리 (기본값: data)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Stable Diffusion 모델 경로",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드 활성화",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cli = EmoseumCLI(data_dir=args.data_dir, model_path=args.model_path)
    cli.run()


if __name__ == "__main__":
    main()