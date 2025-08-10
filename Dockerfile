# 1. 베이스 이미지 설정 (원하는 파이썬 버전에 맞게 수정 가능)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 환경 변수 설정 (버퍼링 없이 로그를 바로 출력)
ENV PYTHONUNBUFFERED 1

# 4. requirements.txt 파일 먼저 복사
COPY requirements.txt .

# 5. 의존성 설치 (가장 중요한 부분!)
# torch를 먼저 설치한 후, 나머지 패키지를 설치하여 오류 해결
RUN pip install --no-cache-dir torch && \
    pip install --no-cache-dir -r requirements.txt

# 6. 나머지 프로젝트 소스 코드 전체 복사
COPY . .

# 7. 앱 실행 명령어 (자신의 실행 파일에 맞게 수정)
# 예: uvicorn main:app --host 0.0.0.0 --port $PORT
# 예: python main.py
CMD ["python", "run_api.py"]