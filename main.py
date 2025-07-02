import os
import uvicorn
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging

# therapy_system.py에서 핵심 클래스들을 가져옵니다.
from therapy_system import HybridEmotionTherapySystem, EmotionEmbedding

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 필요한 폴더 생성
Path("user_adapters").mkdir(exist_ok=True)
Path("generated_images").mkdir(exist_ok=True)

# FastAPI 앱 초기화
app = FastAPI()

# 템플릿 및 정적 파일 설정
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount(
    "/generated_images",
    StaticFiles(directory="generated_images"),
    name="generated_images",
)


# --- AI 시스템 초기화 ---
# 경고: 이 과정은 매우 많은 메모리(VRAM)를 필요로 하며, 앱 시작 시 한 번만 실행되어야 합니다.
logger.info("AI 감정 치료 시스템을 초기화합니다. 시간이 걸릴 수 있습니다...")
# 실제 운영 시에는 try-except로 GPU 메모리 부족 등의 오류를 처리해야 합니다.
try:
    therapy_system = HybridEmotionTherapySystem()
    logger.info("시스템 초기화 완료.")
except Exception as e:
    logger.error(f"시스템 초기화 중 심각한 오류 발생: {e}")
    logger.error(
        "GPU 메모리가 부족하거나 필요한 모델을 다운로드할 수 없는 상태일 수 있습니다."
    )
    therapy_system = None  # 시스템 로드 실패 시 None으로 설정

# --- API 엔드포인트 ---


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """메인 페이지를 렌더링합니다."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate_image(request: Request):
    """사용자 입력에 따라 이미지를 생성합니다."""
    if not therapy_system:
        return JSONResponse(
            status_code=503, content={"error": "AI 시스템이 현재 사용 불가능합니다."}
        )

    form_data = await request.form()
    user_id = form_data.get("user_id")
    diary_text = form_data.get("diary_text")

    if not user_id or not diary_text:
        return JSONResponse(
            status_code=400,
            content={"error": "사용자 ID와 일기 내용을 모두 입력해야 합니다."},
        )

    logger.info(f"'{user_id}' 사용자를 위한 이미지 생성을 시작합니다.")

    try:
        generated_image, emotion, metadata = therapy_system.generate_therapeutic_image(
            user_id=user_id,
            diary_text=diary_text,
            base_prompt="digital art, therapeutic image",
        )

        # 생성된 이미지를 파일로 저장
        timestamp = metadata.get("timestamp", str(hash(diary_text)))
        image_filename = f"{user_id}_{timestamp}.png"
        image_path = Path("generated_images") / image_filename
        generated_image.save(image_path)

        logger.info(f"이미지 생성 완료: {image_path}")

        return JSONResponse(
            {
                "success": True,
                "image_url": f"/generated_images/{image_filename}",
                "emotion": {
                    "valence": f"{emotion.valence:.2f}",
                    "arousal": f"{emotion.arousal:.2f}",
                    "dominance": f"{emotion.dominance:.2f}",
                },
                # 피드백을 위해 필요한 정보 전달
                "feedback_data": {
                    "user_id": user_id,
                    "diary_text": diary_text,
                    "emotion_dict": emotion.__dict__,
                    "image_filename": image_filename,
                },
            }
        )

    except Exception as e:
        logger.error(f"이미지 생성 중 오류 발생: {e}")
        return JSONResponse(
            status_code=500, content={"error": "이미지 생성 중 오류가 발생했습니다."}
        )


def run_training_in_background(user_id, diary_text, feedback_score):
    """백그라운드에서 DRaFT 학습을 수행하는 함수"""
    logger.info(
        f"'{user_id}' 사용자를 위한 백그라운드 학습 시작 (점수: {feedback_score})."
    )
    try:
        therapy_system.train_with_user_feedback(
            user_id=user_id, diary_text=diary_text, feedback_score=float(feedback_score)
        )
        logger.info(f"'{user_id}' 사용자를 위한 백그라운드 학습 완료.")
    except Exception as e:
        logger.error(f"백그라운드 학습 중 오류 발생: {e}")


@app.post("/feedback")
async def handle_feedback(background_tasks: BackgroundTasks, request: Request):
    """사용자 피드백을 처리하고 모델 학습을 트리거합니다."""
    if not therapy_system:
        return JSONResponse(
            status_code=503, content={"error": "AI 시스템이 현재 사용 불가능합니다."}
        )

    form_data = await request.form()
    user_id = form_data.get("user_id")
    diary_text = form_data.get("diary_text")
    feedback_score = form_data.get("feedback_score")
    image_filename = form_data.get("image_filename")
    emotion_dict_str = form_data.get("emotion_dict")

    # 문자열로 받은 emotion_dict를 다시 객체로 변환
    import json

    emotion = EmotionEmbedding.from_vector(
        list(json.loads(emotion_dict_str.replace("'", '"')).values())
    )

    # PIL Image로 다시 로드
    from PIL import Image

    image_path = Path("generated_images") / image_filename
    if not image_path.exists():
        return JSONResponse(
            status_code=404, content={"error": "피드백할 이미지를 찾을 수 없습니다."}
        )

    generated_image = Image.open(image_path)

    # 1. 프로파일 업데이트 (빠르게 실행됨)
    therapy_system.update_user_feedback(
        user_id, generated_image, emotion, diary_text, float(feedback_score)
    )
    logger.info(f"'{user_id}' 사용자의 프로파일이 업데이트되었습니다.")

    # 2. 모델 학습 (시간이 오래 걸리므로 백그라운드에서 실행)
    background_tasks.add_task(
        run_training_in_background, user_id, diary_text, feedback_score
    )

    return JSONResponse(
        {
            "success": True,
            "message": "피드백이 성공적으로 제출되었으며, 개인화 학습이 시작되었습니다.",
        }
    )


# 서버 실행 (터미널에서 `uvicorn main:app --reload` 명령어로 실행)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
