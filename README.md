# README.md

![alt text](<포스터 최종안.jpg>)

# Emoseum

감정 일기 기반 이미지 생성 및 자기 전시 시스템을 활용한 ACT 기반 디지털 치료제

## 📖 프로젝트 소개

Emoseum은 ACT(Acceptance and Commitment Therapy) 이론에 기반한 디지털 치료 시스템이다. 사용자가 작성한 감정 일기를 GPT와 Stable Diffusion을 통해 개인화된 이미지로 시각화하여, 감정 수용과 심리적 유연성을 증진시키는 혁신적인 치료적 경험을 제공한다.

본 시스템은 4단계 ACT 치료 여정(The Moment → Reflection → Defusion → Closure)을 통해 사용자가 자신의 감정을 안전하게 탐색하고 수용할 수 있도록 도우며, 궁극적으로 희망을 찾아가는 과정을 지원한다.

## 📅 개발 기간

**2024년 6월 29일 ~ 8월 14일** (약 7주)

## 👥 개발자 소개

### 팀 구성 및 역할 분담

| 역할           | 개발자 | GitHub                                       | 담당 업무                                                                                                                                      |
| -------------- | ------ | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **AI 개발**    | 박용성 | [@reo91004](https://github.com/reo91004)     | • GPT 프롬프트 엔지니어링 (이미지 생성, 도슨트)<br>• Stable Diffusion 이미지 생성<br>• GoEmotions VAD 감정 분석<br>• LoRA/DRaFT+ 개인화 시스템 |
| **AI 개발**    | 송인규 | [@enqueue01](https://github.com/enqueue01)   | • GPT 프롬프트 엔지니어링 (이미지 생성, 도슨트)<br>• 안전성 검증 시스템<br>• GPT API 서비스<br>• 비용 추적 및 모니터링                         |
| **백엔드**     | 이선진 | [@Seonjin-13](https://github.com/Seonjin-13) | • FastAPI 서버 구축<br>• MongoDB 데이터베이스 설계<br>• 사용자 관리 시스템<br>• Unity 연동 API                                                 |
| **Unity 개발** | 추성재 | [@qOLOp](https://github.com/qOLOp)           | • 모바일 게임 UI/UX<br>• 미술관 인터페이스<br>• 클라이언트-서버 통신<br>• 게임화 요소 구현                                                     |

## 💻 개발 환경

### 설치 및 실행

#### 1. 기본 서버 설정

```bash
# 저장소 클론
git clone https://github.com/Emoseum/emoseum-AI.git
cd emoseum-AI

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에서 필요한 API 키 설정:
# - OPENAI_API_KEY
# - MONGODB_URI
# - SUPABASE_URL
# - SUPABASE_KEY
```

#### 2. Google Colab GPU 서버 실행 (권장)

Google Colab에서 `tests/colab_server.ipynb`를 실행하여 GoEmotions와 Stable Diffusion 서버를 구동:

1. [Google Colab](https://colab.research.google.com)에서 `tests/colab_server.ipynb` 열기
2. ngrok 인증 토큰 설정
3. 모든 셀 실행
4. 생성된 Public URL을 `config/settings.yaml`에 설정

#### 3. 시스템 실행

```bash
# CLI 모드 실행
python main.py

# FastAPI 서버 실행 (Unity 클라이언트용)
python run_api.py

# 또는 uvicorn으로 직접 실행
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## ✨ 주요 기능

### 🎭 ACT 기반 4단계 치료 여정

#### 1. The Moment - 감정 인식

- 자유로운 감정 일기 작성
- GoEmotions 기반 28가지 감정 실시간 분류
- VAD(Valence-Arousal-Dominance) 차원적 감정 분석
- 동적 임계값 기반 의미있는 감정 추출

#### 2. Reflection - 감정 시각화

- GPT 기반 개인화 프롬프트 생성
- Stable Diffusion을 통한 감정 이미지 생성
- 대처 스타일별 맞춤형 시각적 표현
- 1:1 비율 최적화 (모바일 갤러리용)

#### 3. Defusion - 인지적 탈융합

- 방명록 작성을 통한 감정 거리두기
- 작품 제목과 태그를 통한 감정 재구성
- 부정적 사고 패턴 완화
- 감정 객관화 촉진

#### 4. Closure - 희망 발견

- GPT 기반 개인화 도슨트 메시지
- 치료적 통찰과 격려 제공
- 미래 지향적 시각 형성
- 긍정적 재구성 지원

## 🏗 프로젝트 아키텍처

### 시스템 구조

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Unity Client  │    │  FastAPI Server │    │  AI/ML Engine   │    │   Colab GPU     │
│                 │    │                 │    │                 │    │                 │
│ • Game UI       │◄──►│ • RESTful API   │◄──►│ • GPT Service   │◄──►│ • GoEmotions    │
│ • Gallery View  │    │ • User Manager  │    │ • Emotion Analyzer│   │ • Stable Diff   │
│ • Mobile UX     │    │ • Data Storage  │    │ • Personalization│   │ • VAD Mapping   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 모듈 구조

```
emoseum-AI/
├── api/
│   └── main.py                    # FastAPI 서버 엔트리포인트
├── src/
│   ├── core/
│   │   └── act_therapy_system.py  # ACT 치료 시스템 통합 코어
│   ├── managers/
│   │   ├── user_manager.py        # 사용자 프로필 및 심리검사
│   │   ├── gallery_manager.py     # 감정 여정 데이터 관리
│   │   └── personalization_manager.py # 3단계 개인화 엔진
│   ├── services/
│   │   ├── gpt_service.py         # OpenAI GPT API 통합
│   │   ├── emotion_analyzer.py    # GoEmotions VAD 분석
│   │   └── image_generator.py     # Stable Diffusion 서비스
│   ├── therapy/
│   │   ├── prompt_architect.py    # 대처 스타일별 프롬프트
│   │   └── docent_message.py      # 치료적 도슨트 메시지
│   ├── training/
│   │   ├── lora_trainer.py        # LoRA 개인화 훈련
│   │   └── draft_trainer.py       # DRaFT+ 강화학습
│   └── utils/
│       ├── safety_validator.py    # 안전성 검증 시스템
│       └── cost_tracker.py        # API 비용 모니터링
├── config/
│   ├── settings.yaml              # 시스템 설정
│   ├── gpt_prompts.yaml          # GPT 프롬프트 템플릿
│   └── safety_rules.yaml        # 안전 규칙 정의
├── tests/
│   ├── colab_server.ipynb       # Google Colab GPU 서버
│   └── test.py                  # 통합 테스트
└── main.py                       # CLI 인터페이스
```

### 데이터 흐름

```
감정 일기 입력
    ↓
GoEmotions 28가지 감정 분류
    ↓
VAD 차원 매핑 및 강도 계산
    ↓
대처 스타일별 프롬프트 생성 (GPT)
    ↓
개인화 이미지 생성 (Stable Diffusion)
    ↓
방명록 작성 (Defusion)
    ↓
도슨트 메시지 생성 (GPT)
    ↓
개인화 데이터 학습 및 업데이트
```
