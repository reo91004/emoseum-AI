# test_colab_integration.py

"""
Colab GoEmotions 통합 테스트 스크립트
"""

import os
from dotenv import load_dotenv
from src.services.emotion_analyzer import get_emotion_analyzer

load_dotenv()

def test_emotion_services():
    """다양한 감정 분석 서비스 테스트"""
    
    print("=== Colab GoEmotions 통합 테스트 ===")
    
    test_text = "I'm still furious about what happened today. We argued over something small, but it turned into a mess."
    
    # 환경변수 확인
    emotion_service = os.getenv("EMOTION_ANALYSIS_SERVICE", "local_goEmotions")
    colab_url = os.getenv("COLAB_NOTEBOOK_URL", "")
    
    print(f"현재 설정:")
    print(f"  EMOTION_ANALYSIS_SERVICE: {emotion_service}")
    print(f"  COLAB_NOTEBOOK_URL: {colab_url}")
    
    # 각 서비스 테스트
    services_to_test = ["local_goEmotions", "colab_goEmotions"]
    
    for service_type in services_to_test:
        print(f"\n--- {service_type} 테스트 ---")
        
        try:
            analyzer = get_emotion_analyzer(service_type)
            print(f"✅ {service_type} 분석기 초기화 성공")
            
            result = analyzer.analyze_emotions(test_text)
            
            if "error" in result:
                print(f"❌ 분석 실패: {result['error']}")
            else:
                print(f"감정 키워드: {result['keywords']}")
                print(f"주요 감정: {result['primary_emotion']}")
                print(f"감정 강도: {result['emotional_intensity']}")
                print(f"신뢰도: {result['confidence']:.3f}")
                
        except Exception as e:
            print(f"❌ {service_type} 테스트 실패: {e}")

def test_colab_server_endpoints():
    """Colab 서버 엔드포인트 직접 테스트"""
    import requests
    
    colab_url = os.getenv("COLAB_NOTEBOOK_URL", "").rstrip('/')
    if not colab_url:
        print("❌ COLAB_NOTEBOOK_URL이 설정되지 않았습니다.")
        return
    
    print(f"\n=== Colab 서버 엔드포인트 테스트: {colab_url} ===")
    
    # Health check
    try:
        response = requests.get(f"{colab_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check 성공:")
            print(f"  Status: {health_data.get('status')}")
            print(f"  GoEmotions loaded: {health_data.get('goEmotions_loaded')}")
            print(f"  GPU available: {health_data.get('gpu_available')}")
        else:
            print(f"❌ Health check 실패: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Health check 오류: {e}")
    
    # Emotion analysis
    try:
        payload = {
            "text": "I'm excited about the new project!",
            "threshold": 0.3
        }
        
        response = requests.post(
            f"{colab_url}/analyze_emotion",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"✅ 감정 분석 성공:")
                print(f"  Keywords: {result.get('keywords')}")
                print(f"  Primary emotion: {result.get('primary_emotion')}")
            else:
                print(f"❌ 감정 분석 실패: {result.get('error')}")
        else:
            print(f"❌ 감정 분석 HTTP 오류: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 감정 분석 요청 오류: {e}")

if __name__ == "__main__":
    test_emotion_services()
    test_colab_server_endpoints()