# tests/test.py

import requests
import json
import uuid
from datetime import datetime
from typing import Dict, Any


class EmoSeumAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token = None
        self.user_id = None
        self.session_id = None
    
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
            "ciss_score": 45
        }
        response = requests.post(f"{self.base_url}/users/assessment", 
                               json=assessment_data, headers=headers)
        self.print_response("심리측정 평가", response)
        
        # 4. 시각적 선호도 업데이트
        preferences_data = {
            "preferred_styles": ["painting", "abstract"],
            "color_preferences": ["warm", "pastel"],
            "complexity_level": "medium"
        }
        response = requests.put(f"{self.base_url}/users/visual-preferences", 
                              json=preferences_data, headers=headers)
        self.print_response("시각적 선호도 업데이트", response)
        
        # 5. 프로필 설정 업데이트
        settings_data = {
            "language": "ko",
            "notifications": True
        }
        response = requests.put(f"{self.base_url}/users/profile", 
                              json=settings_data, headers=headers)
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
        
        # 2. 일기 작성
        diary_data = {
            "diary_text": "Today I felt really depressed. I wasn't in a good mood from the morning, and work didn't go well either."
        }
        response = requests.post(f"{self.base_url}/therapy/sessions/{self.session_id}/diary", 
                               json=diary_data, headers=headers)
        self.print_response("일기 작성", response)
        
        # 일기 작성이 성공하면 새로운 session_id(gallery_item_id)로 업데이트
        if response.status_code == 200:
            self.session_id = response.json()["session_id"]
            print(f"세션 ID가 업데이트되었습니다: {self.session_id}")
        
        # 3. 성찰 이미지 생성
        response = requests.post(f"{self.base_url}/therapy/sessions/{self.session_id}/reflect", 
                               headers=headers)
        self.print_response("성찰 이미지 생성", response)
        
        # 4. 방명록 작성
        guestbook_data = {
            "title": "Today's Therapy Reflection",
            "tags": ["grateful", "helpful", "therapy"],
            "reflection": "Today's therapy session was very helpful. Thank you."
        }
        response = requests.post(f"{self.base_url}/therapy/sessions/{self.session_id}/guestbook", 
                               json=guestbook_data, headers=headers)
        self.print_response("방명록 작성", response)
        
        # 5. 큐레이터 메시지 생성
        response = requests.post(f"{self.base_url}/therapy/sessions/{self.session_id}/curator", 
                               headers=headers)
        self.print_response("큐레이터 메시지 생성", response)
        
        # 6. 세션 상세 정보 조회
        response = requests.get(f"{self.base_url}/therapy/sessions/{self.session_id}", 
                              headers=headers)
        self.print_response("세션 상세 정보 조회", response)
    
    def test_gallery_endpoints(self):
        """갤러리 관련 엔드포인트 테스트"""
        headers = self.get_headers()
        
        # 1. 갤러리 아이템 목록 조회
        response = requests.get(f"{self.base_url}/gallery/items", headers=headers)
        self.print_response("갤러리 아이템 목록 조회", response)
        
        # 2. 갤러리 분석 데이터 조회
        response = requests.get(f"{self.base_url}/gallery/analytics?days=30", headers=headers)
        self.print_response("갤러리 분석 데이터 조회", response)
        
        # 3. 갤러리 데이터 내보내기
        response = requests.get(f"{self.base_url}/gallery/export?format=json", headers=headers)
        self.print_response("갤러리 데이터 내보내기", response)
    
    def test_training_endpoints(self):
        """트레이닝 관련 엔드포인트 테스트"""
        headers = self.get_headers()
        
        # 1. 트레이닝 자격 확인
        response = requests.get(f"{self.base_url}/training/eligibility", headers=headers)
        self.print_response("트레이닝 자격 확인", response)
        
        # 2. LoRA 트레이닝 시작 (실제로는 시작하지 않고 응답만 확인)
        training_data = {
            "model_name": "test_model",
            "training_type": "lora"
        }
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
