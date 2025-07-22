# README.md

# Emoseum - ACT 기반 디지털 치료 시스템

## 개요

Emoseum은 **ACT(Acceptance and Commitment Therapy)** 이론에 기반한 혁신적인 디지털 치료 시스템이다. 사용자의 감정 일기를 AI로 시각화하여 감정 수용과 심리적 유연성을 증진시키고, 궁극적으로 희망을 찾아가는 여정을 돕는다.

### 핵심 특징

- **ACT 4단계 치료 프로세스**: The Moment → Reflection → Defusion → Closure
- **3단계 개인화 시스템**: 기본 프로파일링 → 룰 기반 개인화 → ML 기반 고급 개인화
- **대처 스타일별 맞춤형 접근**: 회피형/직면형/균형형 사용자별 차별화된 감정 표현
- **시각적 치료 경험**: Stable Diffusion을 활용한 개인화된 이미지 생성

## 시스템 아키텍처

### 모듈 구조

```
src/
├── act_therapy_system.py      # ACT 치료 시스템 통합 관리
├── user_manager.py            # 사용자 프로필 및 심리검사 관리
├── prompt_architect.py        # ACT 기반 프롬프트 생성
├── image_generator.py         # Stable Diffusion 이미지 생성
├── gallery_manager.py         # 감정 여정 데이터베이스 관리
├── personalization_manager.py # 룰 기반 개인화 엔진
└── rule_manager.py           # 대처 스타일별 감정 표현 규칙

training/
├── lora_trainer.py           # LoRA 개인화 트레이너 (Level 3)
└── draft_trainer.py          # DRaFT+ 강화학습 트레이너 (Level 3)
```

### 데이터 흐름

```
감정 일기 입력
    ↓
VAD 기반 감정 분석
    ↓
대처 스타일 반영 Reflection 이미지
    ↓
방명록 작성 (Defusion)
    ↓
희망 이미지 생성 (Closure)
    ↓
개인화 학습 및 업데이트
```

## 설치 방법

### 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장) 또는 Apple Silicon Mac
- 최소 8GB RAM

### 설치 단계

1. 저장소 클론
```bash
git clone https://github.com/yourusername/Emoseum-image-gen.git
cd Emoseum-image-gen
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 환경 설정
```bash
cp .env.example .env
# .env 파일을 편집하여 설정 조정
```

## 사용 방법

### 기본 실행

```bash
python main.py
```

### 옵션

```bash
python main.py --data-dir custom_data --model-path your/model/path --debug
```

### 사용 흐름

1. **신규 사용자 등록**
   - 사용자 ID 생성
   - 심리검사 실시 (PHQ-9, CES-D, MEAQ, CISS)
   - 시각적 선호도 설정

2. **감정 여정 시작**
   - 감정 일기 작성
   - Reflection 이미지 자동 생성
   - 방명록 작성 (제목, 태그)
   - Hope 이미지 생성

3. **미술관 탐색**
   - 과거 작품들 열람
   - 감정 변화 추이 확인
   - 치료적 인사이트 확인

4. **고급 개인화** (Level 3)
   - 50개 이상의 긍정적 반응 축적 시 LoRA 훈련
   - 30개 이상의 완성된 여정 시 DRaFT+ 훈련

## 개인화 시스템

### Level 1: 초기 프로파일링
- 심리검사를 통한 대처 스타일 분류
- 시각적 선호도 설정 (화풍, 색감, 복잡도)
- 주기적 재검사 (2주, 4주)

### Level 2: 룰 기반 개인화
- 대처 스타일별 감정 표현 조절
  - 회피형: 부드럽고 은유적 표현
  - 직면형: 직접적이고 강렬한 표현
  - 균형형: 중성적 표현
- 방명록 피드백 기반 선호도 업데이트

### Level 3: ML 기반 고급 개인화
- **LoRA (Low-Rank Adaptation)**
  - 사용자별 개인화 어댑터 훈련
  - 최소 50개 긍정적 반응 필요
- **DRaFT+ (Reward-based Fine-tuning)**
  - 방명록 감정 점수 기반 강화학습
  - 최소 30개 완성된 여정 필요

## 기술적 특징

### 심리학적 기반
- ACT(수용전념치료) 이론 적용
- VAD(Valence-Arousal-Dominance) 감정 모델
- 대처 스타일 이론 (Endler & Parker, 1990)

### 기술 스택
- **이미지 생성**: Stable Diffusion 1.5 + ControlNet
- **개인화**: LoRA, DRaFT+ 강화학습
- **데이터베이스**: SQLite (SQLAlchemy ORM)
- **프론트엔드**: CLI (향후 웹/모바일 확장 가능)

### 확장성
- 모듈화된 아키텍처로 쉬운 기능 추가
- 폴백 메커니즘으로 안정적 동작
- JSON 기반 규칙 시스템으로 유연한 설정

## 연구적 가치

본 시스템은 즉시 사용 가능한 룰 기반 개인화를 제공하면서도, 데이터 축적 시 LoRA와 DRaFT+를 통해 개인화를 고도화할 수 있는 준비된(Data-Ready) 학습 파이프라인을 포함한다. 이러한 고급 모델들의 효과성 검증은 향후 중요한 연구 과제이다.

### 주요 연구 질문
1. 대처 스타일별 차별화된 시각화가 치료 효과에 미치는 영향
2. 방명록 시스템을 통한 인지적 탈융합(Defusion)의 효과성
3. 개인화 수준(Level 1-3)에 따른 사용자 만족도 및 치료 성과 차이

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포된다.

## 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 연락처

프로젝트 관련 문의사항은 이슈 트래커를 통해 남겨주세요.

---

**Note**: 이 시스템은 전문적인 의료 서비스를 대체하지 않는다. 심각한 우울 증상이 있는 경우 반드시 전문가의 도움을 받기를 권한다.