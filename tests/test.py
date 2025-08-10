# tests/test.py

import requests
import json
import uuid
from datetime import datetime
from typing import Dict, Any

try:
    from bson import ObjectId

    BSON_AVAILABLE = True
except ImportError:
    BSON_AVAILABLE = False
    ObjectId = None


class EmoSeumAPITester:
    def __init__(self, base_url: str = "http://49.143.8.98:8081"):
        self.base_url = base_url
        self.token = None
        self.user_id = None
        self.session_id = None

        print("\n[INIT] EmoSeumAPITester 초기화됨")
        print(f"[INIT] base_url: {self.base_url}")
        print(f"[INIT] token: {self.token}")
        print(f"[INIT] user_id: {self.user_id}")
        print(f"[INIT] session_id: {self.session_id}")

    def print_response(self, name: str, response: requests.Response):
        """응답을 보기 좋게 출력"""
        print(f"\n{'='*50}")
        print(f"테스트: {name}")
        print(f"상태 코드: {response.status_code}")
        print(f"응답 시간: {response.elapsed.total_seconds():.3f}초")

        try:
            data = response.json()
            print(f"응답 내용:\n{json.dumps(data, indent=2, ensure_ascii=False)}")
        except:
            print(f"응답 내용: {response.text}")

        print(f"{'='*50}\n")

    def test_auth_flow(self):
        """인증 플로우 테스트"""
        # 랜덤 유저 ID 생성
        self.user_id = f"test_user_{uuid.uuid4().hex[:8]}"

        # 1. 회원가입
        register_data = {"user_id": self.user_id}
        response = requests.post(f"{self.base_url}/auth/register", json=register_data)
        self.print_response("회원가입", response)

        if response.status_code == 200:
            self.token = response.json()["access_token"]

        # 2. 로그인 테스트 (새로운 유저로는 실패해야 함)
        response = requests.post(f"{self.base_url}/auth/login", json=register_data)
        self.print_response("로그인", response)

        # 3. 로그아웃
        response = requests.delete(f"{self.base_url}/auth/logout")
        self.print_response("로그아웃", response)

    def get_headers(self) -> Dict[str, str]:
        """인증 헤더 반환"""
        if not self.token:
            raise ValueError("토큰이 없음. 먼저 인증을 수행하세요.")
        return {"Authorization": f"Bearer {self.token}"}

    def test_user_endpoints(self):
        """사용자 관련 엔드포인트 테스트"""
        headers = self.get_headers()

        # 1. 프로필 조회
        response = requests.get(f"{self.base_url}/users/profile", headers=headers)
        self.print_response("사용자 프로필 조회", response)

        # 2. 사용자 상태 조회
        response = requests.get(f"{self.base_url}/users/status", headers=headers)
        self.print_response("사용자 상태 조회", response)

        # 3. 심리측정 평가
        assessment_data = {
            "phq9_score": 10,
            "cesd_score": 25,
            "meaq_score": 35,
            "ciss_score": 45,
        }
        response = requests.post(
            f"{self.base_url}/users/assessment", json=assessment_data, headers=headers
        )
        self.print_response("심리측정 평가", response)

        # 4. 시각적 선호도 업데이트
        preferences_data = {
            "preferred_styles": ["painting", "abstract"],
            "color_preferences": ["warm", "pastel"],
            "complexity_level": "medium",
        }
        response = requests.put(
            f"{self.base_url}/users/visual-preferences",
            json=preferences_data,
            headers=headers,
        )
        self.print_response("시각적 선호도 업데이트", response)

        # 5. 프로필 설정 업데이트
        settings_data = {"language": "ko", "notifications": True}
        response = requests.put(
            f"{self.base_url}/users/profile", json=settings_data, headers=headers
        )
        self.print_response("프로필 설정 업데이트", response)

    def test_therapy_endpoints(self):
        """치료 세션 관련 엔드포인트 테스트"""
        headers = self.get_headers()

        # 1. 치료 세션 시작
        response = requests.post(f"{self.base_url}/therapy/sessions", headers=headers)
        self.print_response("치료 세션 시작", response)

        if response.status_code == 200:
            self.session_id = response.json()["session_id"]

        if not self.session_id:
            print("세션 ID가 없어서 치료 관련 테스트를 건너뜁니다.")
            return

        # 2. 일기 작성 (새로운 감정 분석 필드 테스트)
        diary_data = {
            "diary_text": "Today I felt really angry and disappointed. I wasn't in a good mood from the morning, and work didn't go well either. I felt confused about my future and surprised by how much this affected me.",
            "diary_id": (
                str(ObjectId())
                if BSON_AVAILABLE
                else f"test_diary_{uuid.uuid4().hex[:24]}"
            ),
        }
        response = requests.post(
            f"{self.base_url}/therapy/sessions/{self.session_id}/diary",
            json=diary_data,
            headers=headers,
        )
        self.print_response("일기 작성", response)

        # 새로운 감정 분석 필드 검증
        if response.status_code == 200:
            emotion_analysis = response.json().get("emotion_analysis", {})
            print("\n🔍 감정 분석 결과 검증:")
            print(f"  - Keywords: {emotion_analysis.get('keywords', [])}")
            print(f"  - VAD Scores: {emotion_analysis.get('vad_scores', [])}")
            print(
                f"  - Primary Emotion: {emotion_analysis.get('primary_emotion', 'N/A')}"
            )
            print(
                f"  - Normalized All Keys: {list(emotion_analysis.get('normalized_all', {}).keys())[:5]}..."
            )
            print(
                f"  - Emotion Categories: {emotion_analysis.get('emotion_categories', {})}"
            )

            # 데이터 검증
            normalized_all = emotion_analysis.get("normalized_all", {})
            emotion_categories = emotion_analysis.get("emotion_categories", {})

            if normalized_all:
                total_normalized = sum(normalized_all.values())
                print(f"  - Normalized Total (should be ~1.0): {total_normalized:.3f}")
                print(f"  - Normalized Count (should be 28): {len(normalized_all)}")

            if emotion_categories:
                total_categories = sum(emotion_categories.values())
                print(f"  - Categories Total (should be ~1.0): {total_categories:.3f}")
                print(f"  - Categories: {list(emotion_categories.keys())}")

            print("✅ 새로운 감정 분석 필드 검증 완료\n")

        # 일기 작성이 성공하면 새로운 session_id(gallery_item_id)로 업데이트
        if response.status_code == 200:
            self.session_id = response.json()["session_id"]
            print(f"세션 ID가 업데이트되었습니다: {self.session_id}")

        # 3. 성찰 이미지 생성
        response = requests.post(
            f"{self.base_url}/therapy/sessions/{self.session_id}/reflect",
            headers=headers,
        )
        self.print_response("성찰 이미지 생성", response)

        # 4. 작품 제목 작성
        artwork_title_data = {
            "title": "Today's Therapy Reflection",
            "reflection": "Today's therapy session was very helpful. Thank you.",
        }
        response = requests.post(
            f"{self.base_url}/therapy/sessions/{self.session_id}/artwork-title",
            json=artwork_title_data,
            headers=headers,
        )
        self.print_response("작품 제목 작성", response)

        # 5. 도슨트 메시지 생성
        response = requests.post(
            f"{self.base_url}/therapy/sessions/{self.session_id}/docent",
            headers=headers,
        )
        self.print_response("도슨트 메시지 생성", response)

        # 6. 세션 상세 정보 조회 (새로운 필드 포함 검증)
        response = requests.get(
            f"{self.base_url}/therapy/sessions/{self.session_id}", headers=headers
        )
        self.print_response("세션 상세 정보 조회", response)

        # 세션 상세 정보에서 새로운 감정 분석 필드 검증
        if response.status_code == 200:
            session_data = response.json()
            emotion_analysis = session_data.get("emotion_analysis", {})

            print("\n🔍 세션 상세 정보에서 감정 분석 필드 검증:")
            if emotion_analysis.get("normalized_all"):
                print(
                    f"  - Normalized All 포함: ✅ ({len(emotion_analysis['normalized_all'])}개 감정)"
                )
            else:
                print("  - Normalized All 누락: ❌")

            if emotion_analysis.get("emotion_categories"):
                print(
                    f"  - Emotion Categories 포함: ✅ ({list(emotion_analysis['emotion_categories'].keys())})"
                )
            else:
                print("  - Emotion Categories 누락: ❌")

            print("✅ 세션 상세 정보 감정 분석 필드 검증 완료\n")

    def test_diary_exploration_endpoints(self):
        """일기 심화 탐색 관련 엔드포인트 테스트"""
        headers = self.get_headers()

        # 1. 일기 심화 탐색 질문 생성 (새로운 감정 분석 필드 테스트)
        exploration_data = {
            "diary_text": "오늘은 정말 우울했다. 아침부터 기분이 좋지 않았고, 일도 잘 풀리지 않았다. 모든 게 다 잘못되는 것 같아서 답답하고 화가 난다.",
            "emotion_keywords": ["우울", "화남", "답답함"],
        }
        response = requests.post(
            f"{self.base_url}/therapy/diary/explore",
            json=exploration_data,
            headers=headers,
        )
        self.print_response("일기 심화 탐색 질문 생성", response)

        # 일기 심화 탐색에서 새로운 감정 분석 필드 검증
        if response.status_code == 200:
            exploration_result = response.json()
            emotion_analysis = exploration_result.get("emotion_analysis", {})

            print("\n🔍 일기 심화 탐색에서 감정 분석 필드 검증:")
            if emotion_analysis:
                print(f"  - 감정 분석 데이터 포함: ✅")
                if emotion_analysis.get("normalized_all"):
                    print(
                        f"  - Normalized All: ✅ ({len(emotion_analysis['normalized_all'])}개)"
                    )
                if emotion_analysis.get("emotion_categories"):
                    print(
                        f"  - Categories: ✅ ({list(emotion_analysis['emotion_categories'].keys())})"
                    )
            else:
                print("  - 감정 분석 데이터 누락: ❌")

            print("✅ 일기 심화 탐색 감정 분석 필드 검증 완료\n")

        # 2. 일기 심화 탐색 질문 생성 (감정 키워드 없이)
        exploration_data_simple = {
            "diary_text": "Today was a good day. I met my friends and we had a great time together."
        }
        response = requests.post(
            f"{self.base_url}/therapy/diary/explore",
            json=exploration_data_simple,
            headers=headers,
        )
        self.print_response("일기 심화 탐색 질문 생성 (간단)", response)

        # 3. 질문 카테고리 정보 조회
        response = requests.get(
            f"{self.base_url}/therapy/diary/explore/categories", headers=headers
        )
        self.print_response("질문 카테고리 정보 조회", response)

        # 4. 안전 가이드라인 조회
        response = requests.get(
            f"{self.base_url}/therapy/diary/explore/safety", headers=headers
        )
        self.print_response("안전 가이드라인 조회", response)

        # 5. 후속 질문 생성 테스트
        follow_up_data = {
            "diary_text": "오늘은 정말 우울했다. 아침부터 기분이 좋지 않았고, 일도 잘 풀리지 않았다.",
            "previous_question": "Can you describe in more detail the specific situation that led to this emotion?",
            "user_response": "I think it started when I had an argument with my colleague in the morning. It made me feel frustrated and isolated.",
            "emotion_keywords": ["우울", "좌절", "고립감"],
        }
        response = requests.post(
            f"{self.base_url}/therapy/diary/explore/follow-up",
            json=follow_up_data,
            headers=headers,
        )
        self.print_response("후속 질문 생성", response)

    def test_new_emotion_analysis_features(self):
        """새로운 감정 분석 기능 집중 테스트"""
        headers = self.get_headers()

        print(f"\n{'='*50}")
        print("새로운 감정 분석 기능 집중 테스트")
        print(f"{'='*50}")

        # 다양한 감정을 포함한 테스트 케이스들
        test_cases = [
            {
                "name": "긍정적 감정 위주",
                "diary": "Today was absolutely amazing! I felt so proud of my achievements and grateful for my supportive friends. The excitement and joy I experienced made me realize how much I love my life.",
            },
            {
                "name": "부정적 감정 위주",
                "diary": "I'm feeling incredibly angry and disappointed today. The sadness and grief are overwhelming me, and I can't shake this feeling of disgust and fear about my future.",
            },
            {
                "name": "혼합된 감정",
                "diary": "I'm confused about my feelings today. While I'm curious about new opportunities, I also feel nervous and surprised by unexpected changes. There's some relief mixed with realization.",
            },
            {
                "name": "중립적 감정",
                "diary": "Today was just a regular day. Nothing particularly exciting or upsetting happened. I went through my usual routine and felt pretty neutral about everything.",
            },
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📊 테스트 케이스 {i}: {test_case['name']}")

            # 새 세션 시작
            response = requests.post(
                f"{self.base_url}/therapy/sessions", headers=headers
            )
            if response.status_code != 200:
                print(f"❌ 세션 시작 실패: {test_case['name']}")
                continue

            session_id = response.json()["session_id"]

            # 일기 작성 및 감정 분석
            diary_data = {
                "diary_text": test_case["diary"],
                "diary_id": f"test_diary_{uuid.uuid4().hex[:24]}",
            }

            response = requests.post(
                f"{self.base_url}/therapy/sessions/{session_id}/diary",
                json=diary_data,
                headers=headers,
            )

            if response.status_code != 200:
                print(f"❌ 일기 작성 실패: {test_case['name']}")
                continue

            # 감정 분석 결과 상세 검증
            emotion_analysis = response.json().get("emotion_analysis", {})
            session_id = response.json()["session_id"]  # 업데이트된 session_id

            print(f"  📝 일기 내용 (첫 50자): {test_case['diary'][:50]}...")
            print(f"  🎯 상위 감정들: {emotion_analysis.get('keywords', [])}")
            print(f"  🧠 주요 감정: {emotion_analysis.get('primary_emotion', 'N/A')}")

            # 정규화 검증
            normalized_all = emotion_analysis.get("normalized_all", {})
            if normalized_all:
                total = sum(normalized_all.values())
                top_5_emotions = sorted(
                    normalized_all.items(), key=lambda x: x[1], reverse=True
                )[:5]
                print(f"  📊 정규화 합계: {total:.3f} (목표: 1.0)")
                print(
                    f"  🔝 상위 5개 정규화 점수: {[(k, f'{v:.3f}') for k, v in top_5_emotions]}"
                )

            # 카테고리별 분석
            categories = emotion_analysis.get("emotion_categories", {})
            if categories:
                cat_total = sum(categories.values())
                print(f"  📂 카테고리 합계: {cat_total:.3f} (목표: 1.0)")
                sorted_categories = sorted(
                    categories.items(), key=lambda x: x[1], reverse=True
                )
                print(
                    f"  📊 카테고리 점수: {[(k, f'{v:.3f}') for k, v in sorted_categories]}"
                )

            # 검증 결과
            validations = []
            if normalized_all and len(normalized_all) == 28:
                validations.append("✅ 28개 감정 정규화 완료")
            else:
                validations.append("❌ 28개 감정 정규화 실패")

            if normalized_all and 0.99 <= sum(normalized_all.values()) <= 1.01:
                validations.append("✅ 정규화 합계 올바름")
            else:
                validations.append("❌ 정규화 합계 오류")

            if categories and len(categories) == 4:
                validations.append("✅ 4개 카테고리 생성 완료")
            else:
                validations.append("❌ 4개 카테고리 생성 실패")

            if categories and 0.99 <= sum(categories.values()) <= 1.01:
                validations.append("✅ 카테고리 합계 올바름")
            else:
                validations.append("❌ 카테고리 합계 오류")

            for validation in validations:
                print(f"  {validation}")

            print(f"  {'='*30}")

        print(f"\n{'='*50}")
        print("✅ 새로운 감정 분석 기능 집중 테스트 완료")
        print(f"{'='*50}\n")

    def test_stepwise_diary_exploration(self):
        """단계적 일기 심화 탐색 플로우 테스트"""
        headers = self.get_headers()

        print(f"\n{'='*50}")
        print("단계적 일기 심화 탐색 플로우 테스트 시작")
        print(f"{'='*50}")

        # 테스트용 일기 내용
        diary_text = "Today I had a really difficult day at work. My manager criticized my project in front of the whole team, and I felt embarrassed and angry. I've been working on this project for weeks, and it felt like all my effort was dismissed."
        emotion_keywords = ["embarrassed", "angry", "dismissed"]

        # 1단계: 초기 질문 생성
        print("\n🔍 1단계: 초기 탐색 질문 생성")
        exploration_data = {
            "diary_text": diary_text,
            "emotion_keywords": emotion_keywords,
        }
        response = requests.post(
            f"{self.base_url}/therapy/diary/explore",
            json=exploration_data,
            headers=headers,
        )
        self.print_response("1단계 - 초기 질문", response)

        if response.status_code != 200:
            print("❌ 초기 질문 생성 실패 - 단계적 테스트 중단")
            return

        # 첫 번째 질문 추출
        try:
            first_result = response.json()
            if not first_result.get("questions"):
                print("❌ 생성된 질문이 없음 - 단계적 테스트 중단")
                return
            first_question = first_result["questions"][0]["question"]
        except Exception as e:
            print(f"❌ 응답 파싱 실패: {e} - 단계적 테스트 중단")
            return

        # 2단계: 첫 번째 질문에 대한 답변 시뮬레이션 및 후속 질문
        print("\n🔍 2단계: 첫 번째 답변 후 후속 질문")
        user_response_1 = "When my manager criticized my work in front of everyone, I felt my face getting hot and my heart racing. I wanted to defend myself but I was too shocked to speak up. It made me question whether I'm good enough for this job."

        follow_up_data_1 = {
            "diary_text": diary_text,
            "previous_question": first_question,
            "user_response": user_response_1,
            "emotion_keywords": emotion_keywords + ["shocked", "questioning"],
        }

        response = requests.post(
            f"{self.base_url}/therapy/diary/explore/follow-up",
            json=follow_up_data_1,
            headers=headers,
        )
        self.print_response("2단계 - 첫 번째 후속 질문", response)

        if response.status_code != 200:
            print("❌ 첫 번째 후속 질문 생성 실패")
            return

        # 두 번째 질문 추출
        try:
            second_result = response.json()
            if not second_result.get("questions"):
                print("❌ 두 번째 질문이 없음")
                return
            second_question = second_result["questions"][0]["question"]
        except Exception as e:
            print(f"❌ 두 번째 응답 파싱 실패: {e}")
            return

        # 3단계: 두 번째 질문에 대한 답변 시뮬레이션 및 후속 질문
        print("\n🔍 3단계: 두 번째 답변 후 후속 질문")
        user_response_2 = "I've always been a perfectionist, and I think that's why criticism hits me so hard. My self-worth is tied to my work performance. When someone criticizes my work, it feels like they're criticizing me as a person."

        follow_up_data_2 = {
            "diary_text": diary_text,
            "previous_question": second_question,
            "user_response": user_response_2,
            "emotion_keywords": emotion_keywords + ["perfectionist", "self-worth"],
        }

        response = requests.post(
            f"{self.base_url}/therapy/diary/explore/follow-up",
            json=follow_up_data_2,
            headers=headers,
        )
        self.print_response("3단계 - 두 번째 후속 질문", response)

        # 4단계: 탐색 완료 시뮬레이션
        print("\n🔍 4단계: 탐색 완료 시나리오")
        if response.status_code == 200:
            try:
                third_result = response.json()
                if third_result.get("questions"):
                    third_question = third_result["questions"][0]["question"]

                    # 마지막 답변 시뮬레이션
                    user_response_3 = "I realize that I need to separate my personal worth from my work performance. Maybe I can view criticism as an opportunity to improve rather than a personal attack. I should also communicate better with my manager about my concerns."

                    follow_up_data_3 = {
                        "diary_text": diary_text,
                        "previous_question": third_question,
                        "user_response": user_response_3,
                        "emotion_keywords": emotion_keywords
                        + ["realization", "improvement"],
                    }

                    response = requests.post(
                        f"{self.base_url}/therapy/diary/explore/follow-up",
                        json=follow_up_data_3,
                        headers=headers,
                    )
                    self.print_response("4단계 - 최종 후속 질문", response)

            except Exception as e:
                print(f"❌ 세 번째 응답 파싱 실패: {e}")

        print(f"\n{'='*50}")
        print("✅ 단계적 일기 심화 탐색 플로우 테스트 완료")
        print(f"{'='*50}\n")

    def test_gallery_endpoints(self):
        """갤러리 관련 엔드포인트 테스트"""
        headers = self.get_headers()

        # 1. 갤러리 아이템 목록 조회
        response = requests.get(f"{self.base_url}/gallery/items", headers=headers)
        self.print_response("갤러리 아이템 목록 조회", response)

        # 2. 갤러리 분석 데이터 조회
        response = requests.get(
            f"{self.base_url}/gallery/analytics?days=30", headers=headers
        )
        self.print_response("갤러리 분석 데이터 조회", response)

        # 3. 갤러리 데이터 내보내기
        response = requests.get(
            f"{self.base_url}/gallery/export?format=json", headers=headers
        )
        self.print_response("갤러리 데이터 내보내기", response)

    def test_training_endpoints(self):
        """트레이닝 관련 엔드포인트 테스트"""
        headers = self.get_headers()

        # 1. 트레이닝 자격 확인
        response = requests.get(
            f"{self.base_url}/training/eligibility", headers=headers
        )
        self.print_response("트레이닝 자격 확인", response)

        # 2. LoRA 트레이닝 시작 (실제로는 시작하지 않고 응답만 확인)
        training_data = {"model_name": "test_model", "training_type": "lora"}
        # 주석 처리: 실제 트레이닝은 시간이 오래 걸림
        # response = requests.post(f"{self.base_url}/training/lora",
        #                        json=training_data, headers=headers)
        # self.print_response("LoRA 트레이닝 시작", response)

    def test_system_endpoints(self):
        """시스템 관련 엔드포인트 테스트 (인증 불필요)"""
        # 1. 시스템 상태 확인
        response = requests.get(f"{self.base_url}/system/status")
        self.print_response("시스템 상태 확인", response)

        # 2. 헬스 체크
        response = requests.get(f"{self.base_url}/system/health")
        self.print_response("헬스 체크", response)

        # 3. API 비용 정보
        response = requests.get(f"{self.base_url}/system/costs?days=7")
        self.print_response("API 비용 정보", response)

        # 4. 시스템 메트릭
        response = requests.get(f"{self.base_url}/system/metrics")
        self.print_response("시스템 메트릭", response)

    def run_all_tests(self):
        """모든 테스트 실행"""
        print(f"EmoSeum API 테스트 시작 - {datetime.now()}")
        print(f"베이스 URL: {self.base_url}")

        try:
            # 시스템 엔드포인트 (인증 불필요)
            print("\n🔧 시스템 엔드포인트 테스트")
            self.test_system_endpoints()

            # 인증 플로우
            print("\n🔐 인증 플로우 테스트")
            self.test_auth_flow()

            if not self.token:
                print("❌ 인증 실패로 인해 나머지 테스트를 건너뜁니다.")
                return

            # 사용자 관련 엔드포인트
            print("\n👤 사용자 엔드포인트 테스트")
            self.test_user_endpoints()

            # 치료 세션 관련 엔드포인트
            print("\n🧠 치료 세션 엔드포인트 테스트")
            self.test_therapy_endpoints()

            # 새로운 감정 분석 기능 집중 테스트
            print("\n🧠 새로운 감정 분석 기능 집중 테스트")
            self.test_new_emotion_analysis_features()

            # 일기 심화 탐색 관련 엔드포인트
            print("\n📝 일기 심화 탐색 엔드포인트 테스트")
            self.test_diary_exploration_endpoints()

            # 단계적 일기 심화 탐색 플로우 테스트
            print("\n🔄 단계적 일기 심화 탐색 플로우 테스트")
            self.test_stepwise_diary_exploration()

            # 갤러리 관련 엔드포인트
            print("\n🖼️ 갤러리 엔드포인트 테스트")
            self.test_gallery_endpoints()

            # 트레이닝 관련 엔드포인트
            print("\n🏋️ 트레이닝 엔드포인트 테스트")
            self.test_training_endpoints()

            print("\n✅ 모든 테스트 완료")

        except Exception as e:
            print(f"\n❌ 테스트 중 오류 발생: {e}")

        finally:
            print(f"\n테스트 종료 - {datetime.now()}")
            if self.user_id:
                print(f"사용된 테스트 유저 ID: {self.user_id}")


def main():
    """메인 함수"""
    tester = EmoSeumAPITester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
