# test_goEmotions.py

"""
GoEmotions 감정 분석 테스트 스크립트
"""

from src.services.emotion_analyzer import get_emotion_analyzer


def test_goEmotions_analysis():
    """GoEmotions 감정 분석 테스트"""

    print("=== GoEmotions 감정 분석 테스트 ===")

    # 분석기 초기화
    try:
        analyzer = get_emotion_analyzer()
        print(f"✅ GoEmotions 모델 로드 성공: {analyzer.model_name}")
        print(f"   디바이스: {analyzer.device}")
    except Exception as e:
        print(f"❌ GoEmotions 모델 로드 실패: {e}")
        return

    # 테스트 텍스트들
    test_texts = [
        "I'm still furious about what happened today. We argued over something small, but it turned into a mess. I felt unheard, like my feelings didn't matter.",
        "오늘 정말 행복한 하루였어. 친구들과 함께 웃으며 시간을 보냈고, 모든 것이 완벽했어.",
        "I feel confused and don't know what to do next. Everything seems uncertain.",
        "Thank you so much for helping me. I really appreciate your support.",
        "I'm excited about the new project starting tomorrow!",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n--- 테스트 {i} ---")
        print(f"입력 텍스트: {text[:50]}...")

        try:
            result = analyzer.analyze_emotions(text)

            print(f"감정 키워드: {result['keywords']}")
            print(f"주요 감정: {result['primary_emotion']}")
            print(f"감정 강도: {result['emotional_intensity']}")
            print(f"VAD 점수: {result['vad_scores']}")
            print(f"신뢰도: {result['confidence']:.3f}")
            print(f"상위 감정: {result['top_emotions']}")

        except Exception as e:
            print(f"❌ 분석 실패: {e}")


if __name__ == "__main__":
    test_goEmotions_analysis()
