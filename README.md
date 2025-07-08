# Emoseum 감정 기반 디지털 치료 이미지 생성 시스템

![Emoseum Banner](https://img.shields.io/badge/Emoseum-AI%20감정%20치료-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 개요

Emoseum은 **감정 일기 텍스트**를 분석하여 **개인화된 치료용 이미지**를 생성하는 AI 기반 디지털 치료 시스템입니다. VAD(Valence-Arousal-Dominance) 감정 모델과 Stable Diffusion을 결합하여 사용자의 감정 상태에 맞는 맞춤형 이미지를 생성하고, 강화학습을 통해 지속적으로 개인화됩니다.

### ✨ 주요 특징

- 🧠 **고급 감정 분석**: VAD 모델 기반 3차원 감정 공간 분석
- 🎨 **AI 이미지 생성**: Stable Diffusion 1.5 기반 치료용 이미지 생성
- 👤 **개인화 학습**: LoRA 어댑터와 DRaFT+ 강화학습을 통한 개인 맞춤화
- 📊 **치료 진행 추적**: SQLite 기반 감정 히스토리 및 치료 진행도 관리
- 💻 **CLI 인터페이스**: 터미널 기반 사용자 친화적 인터페이스

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Emoseum 시스템                             │
├─────────────────────────────────────────────────────────────┤
│                                                           │
│  📝 텍스트 입력 → 😊 감정 분석 → 🎨 이미지 생성 → 📊 피드백    │
│                                                           │
├─────────────────────────────────────────────────────────────┤
│  주요 컴포넌트:                                              │
│  • AdvancedEmotionMapper (VAD 감정 분석)                    │
│  • PersonalizedLoRAManager (개인화 어댑터)                   │
│  • DRaFTPlusRewardModel (강화학습 보상)                      │
│  • UserEmotionProfile (사용자 프로파일)                      │
│  • EmotionalImageTherapySystem (통합 시스템)                 │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 설치 및 설정

### 필수 요구사항

- Python 3.8 이상
- PyTorch 2.0 이상
- CUDA 지원 GPU (권장) 또는 Apple Silicon Mac (MPS 지원)
- 최소 8GB RAM (GPU 메모리 포함)

### 라이브러리 설치

```bash
# 기본 의존성 설치
pip install torch torchvision torchaudio

# AI/ML 라이브러리
pip install transformers diffusers accelerate

# 개인화 및 강화학습
pip install peft

# 이미지 처리 및 유틸리티
pip install pillow numpy opencv-python

# 데이터베이스 (SQLite는 Python 기본 포함)
```

### 프로젝트 클론 및 초기화

```bash
git clone https://github.com/your-repo/Emoseum-image-gen.git
cd Emoseum-image-gen

# 의존성 설치
pip install -r requirements.txt

# 시스템 요구사항 확인
python main.py --help
```

## 📁 프로젝트 구조

```
Emoseum-image-gen/
├── src/                      # 메인 소스 코드
│   ├── config/               # 시스템 설정
│   │   ├── __init__.py
│   │   ├── settings.py       # 기본 설정 및 환경변수
│   │   └── logging_config.py # 로깅 설정
│   ├── core/                 # 핵심 시스템
│   │   ├── __init__.py
│   │   └── therapy_system.py # 메인 치료 시스템
│   ├── models/               # AI 모델들
│   │   ├── __init__.py
│   │   ├── emotion.py        # 감정 임베딩
│   │   ├── emotion_mapper.py # 감정 분석 시스템
│   │   ├── user_profile.py   # 사용자 프로파일
│   │   ├── lora_manager.py   # LoRA 개인화 관리
│   │   ├── reward_model.py   # 강화학습 보상 모델
│   │   ├── evaluators.py     # 품질 평가기들
│   │   ├── diversity_evaluator.py # 다양성 평가기
│   │   ├── adaptive_personalization.py # 적응형 개인화
│   │   └── smart_feedback_system.py # 스마트 피드백
│   ├── training/             # 강화학습 모듈
│   │   ├── __init__.py
│   │   ├── trainer.py        # 기본 트레이너
│   │   ├── ddpo_trainer.py   # DDPO 기반 트레이너
│   │   └── lora_trainer.py   # LoRA 트레이너
│   └── utils/                # 유틸리티
│       ├── __init__.py
│       └── database.py       # 데이터베이스 관리
├── cli/                      # CLI 인터페이스
│   ├── __init__.py
│   ├── main.py              # 메인 CLI
│   └── feedback_cli.py      # 피드백 CLI
├── data/                     # 사용자 데이터 (Git 제외)
│   ├── generated_images/     # 생성된 이미지들
│   ├── user_loras/          # 사용자별 LoRA 어댑터
│   └── user_profiles.db     # 사용자 프로파일 DB
├── logs/                     # 로그 파일들
│   └── therapy.log          # 시스템 로그
├── tests/                    # 테스트 코드
│   └── __init__.py
├── main.py                   # 메인 실행 파일
├── feedback.py              # 피드백 실행 파일
├── requirements.txt          # 기본 의존성
├── requirements-dev.txt      # 개발 의존성
├── pyproject.toml           # 프로젝트 설정
└── README.md                # 프로젝트 문서
```

## 💻 사용법

### 기본 이미지 생성

```bash
# 감정 일기를 통한 이미지 생성
python main.py --user-id "alice" --text "오늘 하루 정말 행복했다"

# 추가 프롬프트와 함께 생성
python main.py --user-id "bob" --text "스트레스가 심하다" --prompt "평온한 자연 풍경"

# 고품질 이미지 생성 (더 많은 스텝)
python main.py --user-id "carol" --text "우울한 기분" --steps 25 --guidance 8.0
```

### 사용자 피드백 및 개인화 학습

```bash
# 전용 피드백 도구 사용 (권장)
python feedback.py --user-id "alice" --emotion-id 14

# 또는 메인 CLI에서 피드백
python main.py --user-id "alice" --emotion-id 1 --feedback-score 4.8 --comments "정말 마음에 든다"

# 부정적 피드백을 통한 모델 개선
python main.py --user-id "bob" --emotion-id 2 --feedback-score 2.3
```

### 치료 진행도 및 인사이트

```bash
# 종합 치료 인사이트 확인
python main.py --user-id "alice" --insights

# 감정 히스토리 조회 (최근 10개)
python main.py --user-id "alice" --history 10

# 시스템 정리 (30일 이상 된 이미지 삭제)
python main.py --user-id "admin" --cleanup 30
```

### 고급 옵션

```bash
# 커스텀 모델 사용
python main.py --user-id "alice" --text "행복한 하루" --model "runwayml/stable-diffusion-v1-5"

# 고해상도 이미지 생성
python main.py --user-id "bob" --text "평온함" --width 768 --height 768

# 상세 로그 출력
python main.py --user-id "carol" --text "기쁨" --verbose
```

## 🧠 핵심 기술

### 1. VAD 감정 모델

- **Valence**: 감정의 긍정/부정 정도 (-1.0 ~ 1.0)
- **Arousal**: 감정의 활성화 정도 (-1.0 ~ 1.0)
- **Dominance**: 감정의 지배/통제 정도 (-1.0 ~ 1.0)

```python
# 감정 분석 예시
emotion = EmotionEmbedding(
    valence=0.8,    # 매우 긍정적
    arousal=0.6,    # 다소 흥미로운
    dominance=0.4   # 적당히 통제감 있는
)
```

### 2. 개인화 시스템

- **LoRA 어댑터**: 사용자별 모델 파라미터 세밀 조정
- **선호도 학습**: 색온도, 밝기, 채도, 대비, 복잡성 등 시각적 선호도 학습
- **적응형 프롬프트**: 감정과 개인 선호도를 결합한 동적 프롬프트 생성

### 3. DRaFT+ 강화학습

- **다차원 보상**: 감정 정확도 + 미적 품질 + 개인화 점수 + 다양성
- **실시간 학습**: 사용자 피드백을 즉시 반영하는 온라인 학습
- **안정성 보장**: 점진적 학습률과 정규화를 통한 안정적 개선

## 📊 데이터베이스 구조

### 감정 히스토리 테이블

```sql
CREATE TABLE emotion_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    input_text TEXT,
    valence REAL,
    arousal REAL,
    dominance REAL,
    confidence REAL,
    generated_prompt TEXT,
    image_path TEXT
);
```

### 피드백 히스토리 테이블

```sql
CREATE TABLE feedback_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    emotion_id INTEGER,
    timestamp TEXT NOT NULL,
    feedback_score REAL,
    feedback_type TEXT,
    comments TEXT,
    FOREIGN KEY (emotion_id) REFERENCES emotion_history (id)
);
```

## 🔧 설정 및 커스터마이징

### 감정 어휘 사전 확장

```python
# main.py의 AdvancedEmotionMapper 클래스에서 수정 가능
emotion_lexicon = {
    "기쁨": EmotionEmbedding(0.8, 0.6, 0.4),
    "슬픔": EmotionEmbedding(-0.7, -0.3, -0.5),
    # 새로운 감정 추가...
}
```

### 개인화 가중치 조정

```python
# 사용자 선호도 가중치
preference_weights = {
    "color_temperature": 0.0,  # -1.0 (차가운) to 1.0 (따뜻한)
    "brightness": 0.0,         # -1.0 (어두운) to 1.0 (밝은)
    "saturation": 0.0,         # -1.0 (무채색) to 1.0 (채도 높은)
    "contrast": 0.0,           # -1.0 (낮은 대비) to 1.0 (높은 대비)
    "complexity": 0.0,         # -1.0 (단순) to 1.0 (복잡)
    "art_style": "realistic",  # realistic, abstract, impressionist
    "composition": "balanced", # minimal, balanced, complex
}
```

## 📈 치료 효과 지표

시스템은 다음과 같은 치료 효과 지표를 추적합니다:

- **감정 트렌드**: 시간에 따른 감정 변화 추이
- **안정성 점수**: 감정 변동성의 감소 정도
- **참여도**: 사용자의 지속적 사용 및 피드백 제공률
- **회복 지표**: 긍정적 피드백 비율과 감정 개선 종합 점수

## 🚨 문제 해결

### 일반적인 오류

1. **GPU 메모리 부족**

   ```bash
   # 더 작은 해상도로 생성
   python main.py --user-id "test" --text "테스트" --width 256 --height 256
   ```

2. **라이브러리 누락**

   ```bash
   # 필수 라이브러리 설치 확인
   python main.py --help
   ```

3. **모델 다운로드 실패**
   ```bash
   # 인터넷 연결 확인 및 Hugging Face 계정 설정
   huggingface-cli login
   ```

### 로그 확인

상세한 로그는 `logs/therapy.log` 파일에서 확인할 수 있습니다:

```bash
# 실시간 로그 확인
tail -f logs/therapy.log

# 오류 로그만 확인
grep "ERROR" logs/therapy.log
```

---

> **주의**: 이 시스템은 연구 및 교육 목적으로 개발되었습니다. 실제 의료 또는 심리 치료를 대체하지 않으며, 전문적인 상담이 필요한 경우 의료진과 상의하시기 바랍니다.
