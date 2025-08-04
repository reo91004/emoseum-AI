# test_env_variables.py

"""
환경변수 설정 및 CLI/API 동작 검증 스크립트
"""

import os
from dotenv import load_dotenv

def test_environment_variables():
    """환경변수 설정 확인"""
    print("=== 환경변수 설정 확인 ===")
    
    # .env 파일 로드
    load_dotenv()
    
    # 이미지 생성 서비스
    image_service = os.getenv("IMAGE_GENERATION_SERVICE", "local")
    colab_url = os.getenv("COLAB_NOTEBOOK_URL", "")
    
    print(f"이미지 생성 서비스: {image_service}")
    print(f"Colab URL: {colab_url}")
    
    if image_service == "colab" and not colab_url:
        print("⚠️  경고: IMAGE_GENERATION_SERVICE가 'colab'이지만 COLAB_NOTEBOOK_URL이 설정되지 않았습니다.")
    
    # 감정 분석 서비스
    emotion_service = os.getenv("EMOTION_ANALYSIS_SERVICE", "local")
    goemotion_model = os.getenv("GOEMOTION_MODEL", "")
    
    print(f"감정 분석 서비스: {emotion_service}")
    print(f"GoEmotions 모델: {goemotion_model}")
    
    if emotion_service == "colab" and not colab_url:
        print("⚠️  경고: EMOTION_ANALYSIS_SERVICE가 'colab'이지만 COLAB_NOTEBOOK_URL이 설정되지 않았습니다.")

def test_cli_initialization():
    """CLI 초기화 테스트"""
    print("\n=== CLI ACTTherapySystem 초기화 테스트 ===")
    
    try:
        from src.core.act_therapy_system import ACTTherapySystem
        
        # 시스템 초기화 (실제 모델 로드 없이 테스트)
        print("ACTTherapySystem 초기화 중...")
        system = ACTTherapySystem()
        
        # 이미지 생성기 타입 확인
        image_gen_type = type(system.image_generator).__name__
        print(f"✅ 이미지 생성기: {image_gen_type}")
        
        # 환경변수 확인
        image_service = os.getenv("IMAGE_GENERATION_SERVICE", "local")
        emotion_service = os.getenv("EMOTION_ANALYSIS_SERVICE", "local")
        
        print(f"   설정: IMAGE_GENERATION_SERVICE={image_service}")
        print(f"   설정: EMOTION_ANALYSIS_SERVICE={emotion_service}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI 초기화 실패: {e}")
        return False

def test_emotion_analyzer_initialization():
    """감정 분석기 초기화 테스트"""
    print("\n=== 감정 분석기 초기화 테스트 ===")
    
    try:
        from src.services.emotion_analyzer import get_emotion_analyzer
        
        emotion_service = os.getenv("EMOTION_ANALYSIS_SERVICE", "local")
        
        if emotion_service == "local":
            print("로컬 GoEmotions 분석기 테스트...")
            analyzer = get_emotion_analyzer("local_goEmotions")
            analyzer_type = type(analyzer).__name__
            print(f"✅ 로컬 분석기: {analyzer_type}")
            
        elif emotion_service == "colab":
            print("Colab GoEmotions 분석기 테스트...")
            analyzer = get_emotion_analyzer("colab_goEmotions")
            analyzer_type = type(analyzer).__name__
            print(f"✅ Colab 분석기: {analyzer_type}")
            print(f"   Colab URL: {analyzer.colab_url}")
            
        return True
        
    except Exception as e:
        print(f"❌ 감정 분석기 초기화 실패: {e}")
        return False

def test_api_initialization():
    """API 초기화 테스트"""
    print("\n=== API 서비스 초기화 테스트 ===")
    
    try:
        # API 이미지 서비스 테스트
        from api.services.image_service import get_image_service
        
        print("API 이미지 서비스 초기화 중...")
        image_service = get_image_service()
        service_type = type(image_service).__name__
        print(f"✅ API 이미지 서비스: {service_type}")
        
        # API 감정 분석 서비스 테스트
        from api.services.emotion_service import get_emotion_service
        
        print("API 감정 분석 서비스 초기화 중...")
        emotion_service = get_emotion_service()
        service_type = type(emotion_service).__name__
        print(f"✅ API 감정 분석 서비스: {service_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ API 초기화 실패: {e}")
        return False

def print_service_combinations():
    """가능한 서비스 조합 출력"""
    print("\n=== 가능한 서비스 조합 ===")
    print("1. 완전 로컬:")
    print("   IMAGE_GENERATION_SERVICE=local")
    print("   EMOTION_ANALYSIS_SERVICE=local")
    print()
    print("2. 완전 Colab:")
    print("   IMAGE_GENERATION_SERVICE=colab")
    print("   EMOTION_ANALYSIS_SERVICE=colab")
    print("   COLAB_NOTEBOOK_URL=https://your-ngrok-url.ngrok-free.app")
    print()
    print("3. 이미지만 Colab:")
    print("   IMAGE_GENERATION_SERVICE=colab")
    print("   EMOTION_ANALYSIS_SERVICE=local")
    print("   COLAB_NOTEBOOK_URL=https://your-ngrok-url.ngrok-free.app")
    print()
    print("4. 감정 분석만 Colab:")
    print("   IMAGE_GENERATION_SERVICE=local")
    print("   EMOTION_ANALYSIS_SERVICE=colab")
    print("   COLAB_NOTEBOOK_URL=https://your-ngrok-url.ngrok-free.app")
    print()
    print("5. GPT 감정 분석 (기존 방식):")
    print("   IMAGE_GENERATION_SERVICE=local 또는 colab")
    print("   EMOTION_ANALYSIS_SERVICE=gpt")

if __name__ == "__main__":
    print("🔧 Emoseum 환경변수 및 서비스 초기화 테스트")
    print("=" * 60)
    
    # 환경변수 확인
    test_environment_variables()
    
    # CLI 초기화 테스트  
    cli_ok = test_cli_initialization()
    
    # 감정 분석기 초기화 테스트
    emotion_ok = test_emotion_analyzer_initialization()
    
    # API 초기화 테스트
    api_ok = test_api_initialization()
    
    # 가능한 조합 출력
    print_service_combinations()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("🎯 테스트 결과 요약:")
    print(f"   CLI 초기화: {'✅ 성공' if cli_ok else '❌ 실패'}")
    print(f"   감정 분석기: {'✅ 성공' if emotion_ok else '❌ 실패'}")
    print(f"   API 초기화: {'✅ 성공' if api_ok else '❌ 실패'}")
    
    if all([cli_ok, emotion_ok, api_ok]):
        print("\n🎉 모든 테스트 통과! 시스템이 정상적으로 설정되었습니다.")
    else:
        print("\n⚠️  일부 테스트가 실패했습니다. 환경변수 설정을 확인해주세요.")