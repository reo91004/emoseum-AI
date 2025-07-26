# config/settings.py

import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GPTSettings:
    """GPT API 설정 관리"""

    def __init__(self):
        # 환경변수에서 API 키 로드
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

        # GPT 모델 설정
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "150"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "1.0"))
        self.frequency_penalty = float(os.getenv("FREQUENCY_PENALTY", "0.0"))
        self.presence_penalty = float(os.getenv("PRESENCE_PENALTY", "0.0"))

        # 타임아웃 설정
        self.timeout_seconds = int(os.getenv("GPT_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("GPT_MAX_RETRIES", "3"))

        # 비용 관리 설정
        self.daily_token_limit = int(os.getenv("DAILY_TOKEN_LIMIT", "10000"))
        self.monthly_cost_limit = float(os.getenv("MONTHLY_COST_LIMIT", "50.0"))

        # 캐싱 설정
        self.enable_response_cache = (
            os.getenv("ENABLE_RESPONSE_CACHE", "true").lower() == "true"
        )
        self.cache_ttl_hours = int(os.getenv("CACHE_TTL_HOURS", "24"))

    def get_openai_config(self) -> Dict[str, Any]:
        """OpenAI API 설정 딕셔너리 반환"""
        return {
            "api_key": self.api_key,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "timeout": self.timeout_seconds,
        }

    def validate_settings(self) -> Dict[str, Any]:
        """설정 유효성 검사"""
        issues = []

        if not self.api_key:
            issues.append("OPENAI_API_KEY가 설정되지 않음")

        if self.max_tokens > 4096:
            issues.append(f"max_tokens가 너무 큼: {self.max_tokens}")

        if not 0 <= self.temperature <= 2:
            issues.append(f"temperature 범위 오류: {self.temperature}")

        if not 0 <= self.top_p <= 1:
            issues.append(f"top_p 범위 오류: {self.top_p}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
        }


class SystemSettings:
    """전체 시스템 설정"""

    def __init__(self):
        # 기본 경로 설정
        self.data_dir = Path(os.getenv("EMOSEUM_DATA_DIR", "data"))
        self.config_dir = Path(os.getenv("EMOSEUM_CONFIG_DIR", "config"))
        self.logs_dir = Path(os.getenv("EMOSEUM_LOGS_DIR", "logs"))

        # 디렉토리 생성
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # 이미지 생성 설정
        self.stable_diffusion_model = os.getenv(
            "SD_MODEL", "runwayml/stable-diffusion-v1-5"
        )
        self.image_generation_timeout = int(os.getenv("IMAGE_TIMEOUT", "120"))

        # 데이터베이스 설정
        self.users_db_path = self.data_dir / "users.db"
        self.gallery_db_path = self.data_dir / "gallery.db"

        # 로깅 설정
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.enable_file_logging = (
            os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
        )

        # 안전성 설정
        self.enable_safety_checks = (
            os.getenv("ENABLE_SAFETY_CHECKS", "true").lower() == "true"
        )
        self.max_prompt_length = int(os.getenv("MAX_PROMPT_LENGTH", "1000"))

        # GPT 설정 초기화
        self.gpt = GPTSettings()

    def get_db_config(self) -> Dict[str, str]:
        """데이터베이스 설정 반환"""
        return {
            "users_db": str(self.users_db_path),
            "gallery_db": str(self.gallery_db_path),
            "backup_dir": str(self.data_dir / "backups"),
        }

    def get_paths_config(self) -> Dict[str, Path]:
        """경로 설정 반환"""
        return {
            "data_dir": self.data_dir,
            "config_dir": self.config_dir,
            "logs_dir": self.logs_dir,
            "images_dir": self.data_dir / "gallery_images",
            "preferences_dir": self.data_dir / "preferences",
            "models_dir": self.data_dir / "models",
        }

    def load_prompt_templates(self) -> Optional[Dict[str, Any]]:
        """프롬프트 템플릿 로드"""
        templates_path = self.config_dir / "gpt_prompts.yaml"

        if not templates_path.exists():
            logger.warning(f"프롬프트 템플릿 파일을 찾을 수 없습니다: {templates_path}")
            return None

        try:
            import yaml

            with open(templates_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"프롬프트 템플릿 로드 실패: {e}")
            return None

    def load_safety_rules(self) -> Optional[Dict[str, Any]]:
        """안전성 규칙 로드"""
        safety_path = self.config_dir / "safety_rules.yaml"

        if not safety_path.exists():
            logger.warning(f"안전성 규칙 파일을 찾을 수 없습니다: {safety_path}")
            return None

        try:
            import yaml

            with open(safety_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"안전성 규칙 로드 실패: {e}")
            return None

    def create_env_template(self) -> str:
        """환경변수 템플릿 생성"""
        template = """# Emoseum GPT Integration Configuration

# OpenAI API Settings
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
MAX_TOKENS=150
TEMPERATURE=0.7
TOP_P=1.0
FREQUENCY_PENALTY=0.0
PRESENCE_PENALTY=0.0

# API Timeout and Retry Settings
GPT_TIMEOUT=30
GPT_MAX_RETRIES=3

# Cost Management
DAILY_TOKEN_LIMIT=10000
MONTHLY_COST_LIMIT=50.0

# Response Caching
ENABLE_RESPONSE_CACHE=true
CACHE_TTL_HOURS=24

# System Paths
EMOSEUM_DATA_DIR=data
EMOSEUM_CONFIG_DIR=config
EMOSEUM_LOGS_DIR=logs

# Stable Diffusion Settings
SD_MODEL=runwayml/stable-diffusion-v1-5
IMAGE_TIMEOUT=120

# Logging Settings
LOG_LEVEL=INFO
ENABLE_FILE_LOGGING=true

# Safety Settings
ENABLE_SAFETY_CHECKS=true
MAX_PROMPT_LENGTH=1000
"""
        return template

    def validate_environment(self) -> Dict[str, Any]:
        """환경 설정 전체 검사"""
        validation_result = {
            "overall_valid": True,
            "gpt_settings": self.gpt.validate_settings(),
            "paths_accessible": True,
            "templates_loaded": False,
            "safety_rules_loaded": False,
            "issues": [],
        }

        # 경로 접근성 확인
        try:
            for path_name, path in self.get_paths_config().items():
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            validation_result["paths_accessible"] = False
            validation_result["issues"].append(f"경로 생성 실패: {e}")

        # 템플릿 로드 확인
        templates = self.load_prompt_templates()
        validation_result["templates_loaded"] = templates is not None

        # 안전성 규칙 로드 확인
        safety_rules = self.load_safety_rules()
        validation_result["safety_rules_loaded"] = safety_rules is not None

        # GPT 설정 확인
        if not validation_result["gpt_settings"]["valid"]:
            validation_result["overall_valid"] = False
            validation_result["issues"].extend(
                validation_result["gpt_settings"]["issues"]
            )

        # 전체 유효성 판단
        if not validation_result["paths_accessible"]:
            validation_result["overall_valid"] = False

        return validation_result


# 전역 설정 인스턴스
settings = SystemSettings()

# 설정 검증 및 로깅
validation_result = settings.validate_environment()
if validation_result["overall_valid"]:
    logger.info("시스템 설정이 성공적으로 로드되었습니다.")
else:
    logger.warning(f"시스템 설정에 문제가 있습니다: {validation_result['issues']}")

# GPT 사용 가능 여부 확인
gpt_available = validation_result["gpt_settings"]["valid"]
if gpt_available:
    logger.info(f"GPT 서비스 사용 가능 - 모델: {settings.gpt.model_name}")
else:
    logger.warning("GPT 서비스를 사용할 수 없습니다. 환경변수를 확인해주세요.")
