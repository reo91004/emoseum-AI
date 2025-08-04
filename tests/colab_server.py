!pip install flask pyngrok transformers torch torchvision

from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
from transformers import pipeline
import torch
import base64
import numpy as np
from io import BytesIO
from pyngrok import ngrok

# ngrok 인증키 설정
ngrok.set_auth_token("your_key_here")  # 여기에 ngrok 인증키를 입력하세요

app = Flask(__name__)

print("🚀 Emoseum Colab Server - Stable Diffusion & GoEmotions")
print("=" * 60)

# ===============================================================

# Stable Diffusion 모델 로드
print("📥 Stable Diffusion 모델 로드 중...")
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
sd_pipeline = sd_pipeline.to("cuda")
print("✅ Stable Diffusion 모델 로드 완료!")

# GoEmotions 모델 로드  
print("📥 GoEmotions 모델 로드 중...")
emotion_classifier = pipeline(
    "text-classification",
    model="joeddav/distilbert-base-uncased-go-emotions-student",
    device=0,  # GPU 사용
    top_k=None  # 모든 레이블의 점수를 반환
)
print("✅ GoEmotions 모델 로드 완료!")

print(f"🎯 Device set to use {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

# GoEmotions 감정 레이블과 VAD 매핑
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

EMOTION_VAD_MAPPING = {
    'admiration': (0.8, 0.5, 0.4), 'amusement': (0.8, 0.6, 0.7),
    'anger': (0.1, 0.8, 0.7), 'annoyance': (0.2, 0.6, 0.5),
    'approval': (0.7, 0.3, 0.6), 'caring': (0.8, 0.4, 0.6),
    'confusion': (0.3, 0.5, 0.2), 'curiosity': (0.6, 0.6, 0.5),
    'desire': (0.7, 0.7, 0.6), 'disappointment': (0.2, 0.3, 0.3),
    'disapproval': (0.2, 0.5, 0.6), 'disgust': (0.1, 0.5, 0.5),
    'embarrassment': (0.2, 0.6, 0.2), 'excitement': (0.9, 0.9, 0.7),
    'fear': (0.1, 0.8, 0.1), 'gratitude': (0.9, 0.4, 0.4),
    'grief': (0.1, 0.3, 0.1), 'joy': (0.9, 0.7, 0.8),
    'love': (0.9, 0.6, 0.6), 'nervousness': (0.3, 0.7, 0.2),
    'optimism': (0.8, 0.5, 0.7), 'pride': (0.8, 0.6, 0.8),
    'realization': (0.6, 0.5, 0.6), 'relief': (0.8, 0.3, 0.5),
    'remorse': (0.2, 0.4, 0.2), 'sadness': (0.1, 0.3, 0.2),
    'surprise': (0.6, 0.8, 0.4), 'neutral': (0.5, 0.5, 0.5)
}


def calculate_vad_scores(emotions, threshold=0.3):
    """감정 점수들로부터 VAD 점수 계산"""
    if not emotions:
        return [0.5, 0.5, 0.5]
    
    # 임계값 이상의 감정들만 선택
    filtered_emotions = [(em['label'], em['score']) for em in emotions if em['score'] >= threshold]
    
    if not filtered_emotions:
        # 임계값을 넘는 감정이 없으면 가장 높은 점수의 감정 사용
        filtered_emotions = [(emotions[0]['label'], emotions[0]['score'])]
    
    total_weight = sum(score for _, score in filtered_emotions)
    if total_weight == 0:
        return [0.5, 0.5, 0.5]
    
    weighted_vad = [0.0, 0.0, 0.0]
    
    for emotion, score in filtered_emotions:
        vad = EMOTION_VAD_MAPPING.get(emotion, (0.5, 0.5, 0.5))
        for i in range(3):
            weighted_vad[i] += vad[i] * score
    
    # 정규화
    vad_scores = [v / total_weight for v in weighted_vad]
    
    # 0-1 범위로 클리핑
    vad_scores = [max(0.0, min(1.0, v)) for v in vad_scores]
    
    return vad_scores

def calculate_intensity(emotions, threshold=0.3):
    """감정 강도 계산"""
    if not emotions:
        return "low"
    
    # 상위 3개 감정의 평균 점수
    top_emotions = emotions[:3]
    avg_score = np.mean([em['score'] for em in top_emotions])
    
    # Arousal 기반 강도 조정
    arousal_boost = 0.0
    for emotion in top_emotions:
        vad = EMOTION_VAD_MAPPING.get(emotion['label'], (0.5, 0.5, 0.5))
        arousal_boost += vad[1] * emotion['score']
    
    arousal_boost /= len(top_emotions)
    
    # 최종 강도 계산
    intensity_score = (avg_score + arousal_boost) / 2
    
    if intensity_score >= 0.7:
        return "high"
    elif intensity_score >= 0.4:
        return "medium"
    else:
        return "low"

@app.route('/generate', methods=['POST'])
def generate_image():
    """이미지 생성 엔드포인트"""
    global sd_pipeline
    
    if sd_pipeline is None:
        return jsonify({"success": False, "error": "Stable Diffusion model not loaded"})
    
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({"success": False, "error": "Prompt is required"})
        
        print(f"📝 이미지 생성 요청: {prompt[:50]}...")
        
        # 이미지 생성
        image = sd_pipeline(prompt).images[0]
        
        # Base64로 인코딩
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        print("✅ 이미지 생성 성공")
        
        return jsonify({
            "image": img_str, 
            "success": True,
            "service": "stable_diffusion",
            "generation_time": 30.0
        })
        
    except Exception as e:
        print(f"❌ 이미지 생성 실패: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    """감정 분석 엔드포인트"""
    global emotion_classifier
    
    if emotion_classifier is None:
        return jsonify({"success": False, "error": "GoEmotions model not loaded"})
    
    try:
        data = request.json
        text = data.get('text', '')
        threshold = data.get('threshold', 0.3)
        
        if not text:
            return jsonify({"success": False, "error": "Text is required"})
        
        print(f"📝 감정 분석 요청: {text[:50]}...")
        
        # 감정 분석 수행
        results = emotion_classifier(text)
        
        # 점수가 높은 감정들 선택
        emotions = []
        scores = {}
        
        for result in results[0]:  # pipeline은 리스트 안에 결과를 반환
            label = result['label']
            score = result['score']
            scores[label] = score
            
            if score >= threshold:
                emotions.append({'label': label, 'score': score})
        
        # 점수 기준으로 정렬
        emotions.sort(key=lambda x: x['score'], reverse=True)
        
        # 상위 5개 감정 선택
        top_emotions = emotions[:5]
        emotion_keywords = [em['label'] for em in top_emotions]
        
        # 감정이 없으면 neutral 추가
        if not emotion_keywords:
            emotion_keywords = ['neutral']
            top_emotions = [{'label': 'neutral', 'score': 1.0}]
        
        # VAD 점수 계산
        vad_scores = calculate_vad_scores(results[0], threshold)
        
        # 주요 감정 결정
        primary_emotion = emotion_keywords[0] if emotion_keywords else 'neutral'
        
        # 감정 강도 계산
        emotional_intensity = calculate_intensity(results[0], threshold)
        
        # 신뢰도 계산
        confidence = float(np.mean([em['score'] for em in top_emotions]))
        
        result_data = {
            "keywords": emotion_keywords,
            "vad_scores": vad_scores,
            "confidence": confidence,
            "primary_emotion": primary_emotion,
            "emotional_intensity": emotional_intensity,
            "all_scores": scores,
            "top_emotions": {em['label']: em['score'] for em in top_emotions}
        }
        
        print(f"✅ 감정 분석 성공: {emotion_keywords}")
        
        return jsonify({
            "success": True,
            "service": "goEmotions",
            **result_data
        })
        
    except Exception as e:
        print(f"❌ 감정 분석 실패: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크 엔드포인트"""
    global sd_pipeline, emotion_classifier
    
    return jsonify({
        "status": "healthy",
        "stable_diffusion_loaded": sd_pipeline is not None,
        "goEmotions_loaded": emotion_classifier is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    })

@app.route('/', methods=['GET'])
def home():
    """홈페이지"""
    return jsonify({
        "message": "Emoseum Colab Server - Stable Diffusion & GoEmotions",
        "endpoints": {
            "/generate": "POST - 이미지 생성 (prompt 필요)",
            "/analyze_emotion": "POST - 감정 분석 (text 필요)", 
            "/health": "GET - 헬스 체크"
        }
    })

# ===============================================================


# ngrok 터널 생성
print("\n🌐 ngrok 터널 생성 중...")
public_url = ngrok.connect(5000)
print(f"\n{'='*60}")
print(f"🎯 Colab Server가 실행됩니다!")
print(f"🌐 Public URL: {public_url}")
print(f"{'='*60}\n")

# Flask 앱 실행
print("🚀 Flask 서버 시작...")
app.run(host='0.0.0.0', port=5000, debug=False)