# Emoseum Image Generation Model

**감정 기반 치료적 이미지 생성을 위한 AI 모델 구현**

본 저장소는 'Emoseum: AI 기반 감정 갤러리' 프로젝트의 **핵심 이미지 생성 모델**을 구현합니다. 사용자의 감정 일기를 분석하여 치료적 효과가 있는 개인 맞춤형 이미지를 생성하는 AI 모델의 실제 구현체입니다.

## 🧠 핵심 모델 구현

**메인 클래스 (`therapy_system.py`)**

- `HybridEmotionTherapySystem`: 감정 분석과 이미지 생성을 통합한 메인 시스템
- `EmotionEmbedding`: VAD 모델 기반 감정 벡터 표현
- `DRaFT (Direct Reward Fine-Tuning)`: 사용자 피드백 기반 모델 개인화
- `LoRA Adapter`: 개인별 이미지 생성 스타일 맞춤화

**1. 감정 분석 파이프라인**

- 자연어 처리를 통한 감정 키워드 추출
- VAD(Valence-Arousal-Dominance) 모델 기반 감정 상태 수치화
- 개인 감정 패턴 학습 및 저장

**2. 이미지 생성 엔진**

- Stable Diffusion 기반 맞춤형 감정 이미지 생성
- 감정 상태와 연동된 프롬프트 자동 생성
- 개인화된 LoRA 어댑터 적용

**3. 개인화 학습 시스템**

- 사용자 피드백 기반 DRaFT 파인튜닝
- 개인별 선호도 반영 모델 업데이트
- 지속적 학습을 통한 치료 효과 최적화

## 🔬 연구 배경

본 모델은 다음과 같은 선행 연구들을 기반으로 개발되었습니다:

1. **AI 기반 글쓰기**: 감정 조절 능력 향상에 긍정적 영향
2. **텍스트 기반 AI 이미지 생성**: 메타인지 능력 자극을 통한 인지 왜곡 완화
3. **미술 치료**: 시각적 예술 창작을 통한 자기 인식과 감정 표현 촉진
4. **한국 우울증 현황**: 지속적인 유병률 증가에 따른 효과적 대응 방안 필요

## 🎯 모델 특징

**자기 통찰력 증진**

- 감정 상태의 시각적 표현을 통한 메타인지 능력 향상
- 개인 감정 패턴의 객관적 분석 및 추적
- 정서 변화와 회복 과정의 체계적 모니터링

**개인화 학습**

- 사용자별 맞춤형 LoRA 어댑터 생성
- 피드백 기반 지속적 모델 개선
- 개인 선호도와 치료 효과 최적화

**치료적 효과**

- 감정 표현 및 인식 능력 향상
- 우울 상태 조기 감지 및 대응
- 예방적 정신건강 관리 지원

## 🔧 기술 스택

**AI/ML 프레임워크**

- **PyTorch**: 딥러닝 모델 구현
- **Transformers**: 자연어 처리 및 감정 분석
- **Diffusers**: Stable Diffusion 이미지 생성
- **PEFT**: LoRA 기반 개인화 파인튜닝

**감정 분석**

- **CLIP**: 텍스트-이미지 연동 임베딩
- **Custom VAD Model**: Valence-Arousal-Dominance 분석

**데이터 처리**

- **SQLite**: 사용자 프로필 및 학습 데이터 저장
- **NumPy/SciPy**: 수치 연산 및 데이터 처리

## 📋 모델 요구사항

- Python 3.8+
- NVIDIA GPU (권장, CUDA 11.8+)
- 최소 12GB VRAM (개인화 학습 시)
- 최소 16GB RAM
- 최소 15GB 디스크 공간 (모델 가중치 저장용)

## 🚀 모델 실행

### 1. 환경 설정

```bash
git clone <repository-url>
cd Emoseum-image-gen
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. 모델 테스트 (웹 데모)

```bash
# 웹 인터페이스를 통한 모델 테스트
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. 직접 모델 사용

```python
from therapy_system import HybridEmotionTherapySystem

# 시스템 초기화
therapy_system = HybridEmotionTherapySystem()

# 이미지 생성
image, emotion, metadata = therapy_system.generate_therapeutic_image(
    user_id="test_user",
    diary_text="오늘은 기분이 좋지 않았다. 모든 일이 잘못되는 것 같다.",
    base_prompt="digital art, therapeutic image"
)

# 피드백 학습
therapy_system.train_with_user_feedback(
    user_id="test_user",
    diary_text="오늘은 기분이 좋지 않았다.",
    feedback_score=3.5
)
```

## 📁 코드 구조

```
Emoseum-image-gen/
├── therapy_system.py       # 🔥 핵심 AI 모델 구현
├── main.py                 # 웹 데모 인터페이스 (테스트용)
├── requirements.txt        # 의존성 패키지
├── user_profiles.db       # 사용자 데이터 (로컬 저장)
├── generated_images/      # 생성된 이미지 샘플
├── user_adapters/         # 개인화 모델 어댑터
├── templates/             # 웹 데모 템플릿 (테스트용)
└── static/               # 웹 데모 리소스 (테스트용)
```

## 🧪 모델 평가

**성능 지표**

- 감정 분석 정확도
- 이미지-텍스트 일치도 (CLIP Score)
- 사용자 만족도 (피드백 점수)
- 개인화 학습 수렴 속도

**테스트 방법**

1. 웹 인터페이스를 통한 실시간 테스트
2. 배치 처리를 통한 대량 데이터 실험
3. 사용자 연구를 통한 치료 효과 검증

## 🔧 개발자 API

### 주요 클래스

**HybridEmotionTherapySystem**

```python
generate_therapeutic_image(user_id, diary_text, base_prompt)
train_with_user_feedback(user_id, diary_text, feedback_score)
update_user_feedback(user_id, image, emotion, diary_text, score)
```

**EmotionEmbedding**

```python
from_text(text) -> EmotionEmbedding
to_vector() -> List[float]
valence, arousal, dominance  # VAD 점수
```

## 📚 참고문헌

[1] AI 기반 글쓰기의 감정 조절 효과 연구  
[2] 텍스트-이미지 생성 AI의 메타인지 능력 향상 효과  
[3] 미술 치료의 정서적 안정 및 자아 정체성 확립 효과  
[4] 한국 우울증 유병률 현황 및 대응 방안  
[5] LoRA 기반 개인화 학습의 치료적 효과  
[6] VAD 모델을 활용한 감정 상태 수치화 연구

## 📄 라이선스

이 모델 구현은 연구 목적으로 개발되었으며, MIT 라이선스 하에 배포됩니다.

**Citation**

```bibtex
@misc{emoseum2025,
  title={Emoseum: AI-based Emotion Gallery for Digital Therapeutics},
  author={Research Team},
  year={2025},
  url={https://github.com/your-repo/Emoseum-image-gen}
}
```
