# src/utils/error_handler.py

# ==============================================================================
# 이 파일은 GPT 관련 작업에서 발생하는 오류를 표준화된 방식으로 처리하기 위한 유틸리티이다.
# API 통신 오류, 콘텐츠 생성 실패, 안전성 위반 등 다양한 유형의 오류에 대해
# 일관된 형식의 에러 응답 객체를 생성하여, 시스템의 다른 부분에서 오류를
# 안정적으로 처리하고 사용자에게 적절한 안내를 제공할 수 있도록 돕는다.
# ==============================================================================

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class GPTErrorHandler:
    """명확한 에러 처리"""

    @staticmethod
    def handle_gpt_failure(
        error_type: str, error_message: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """GPT 실패 시 표준화된 에러 응답"""

        error_responses = {
            "api_unavailable": {
                "message": "GPT service is currently unavailable.",
                "action": "Please check your OpenAI API configuration.",
                "retry_recommended": True,
                "user_message": "Our AI service is temporarily unavailable. Please try again later.",
            },
            "generation_failed": {
                "message": "Content generation failed.",
                "action": "Please try with different input or contact support.",
                "retry_recommended": True,
                "user_message": "We couldn't generate content for your request. Please try again.",
            },
            "safety_violation": {
                "message": "Content violates safety guidelines.",
                "action": "Please modify your input and try again.",
                "retry_recommended": False,
                "user_message": "Your request couldn't be processed due to safety guidelines.",
            },
            "rate_limit_exceeded": {
                "message": "API rate limit exceeded.",
                "action": "Wait a few minutes before trying again.",
                "retry_recommended": True,
                "user_message": "Too many requests. Please wait a moment and try again.",
            },
            "token_limit_exceeded": {
                "message": "Content too long for processing.",
                "action": "Please shorten your input and try again.",
                "retry_recommended": True,
                "user_message": "Your content is too long. Please shorten it and try again.",
            },
            "authentication_failed": {
                "message": "API authentication failed.",
                "action": "Check your OpenAI API key configuration.",
                "retry_recommended": False,
                "user_message": "Authentication error. Please contact support.",
            },
        }

        response_template = error_responses.get(
            error_type, error_responses["generation_failed"]
        )

        logger.error(f"GPT 오류 처리: {error_type} - {error_message}")

        return {
            "success": False,
            "error_type": error_type,
            "error_message": error_message,
            "user_message": response_template["user_message"],
            "technical_message": response_template["message"],
            "recommended_action": response_template["action"],
            "retry_recommended": response_template["retry_recommended"],
            "context": context,
            "fallback_used": False,
            "timestamp": context.get("timestamp", "unknown"),
        }

    @staticmethod
    def handle_prompt_architect_failure(
        operation: str, error: str, diary_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """PromptArchitect 관련 오류 처리"""

        return GPTErrorHandler.handle_gpt_failure(
            error_type="generation_failed",
            error_message=f"PromptArchitect {operation} failed: {error}",
            context={
                "component": "prompt_architect",
                "operation": operation,
                "diary_length": len(diary_context.get("diary_text", "")),
                "emotion_keywords": diary_context.get("emotion_keywords", []),
                "coping_style": diary_context.get("coping_style", "unknown"),
            },
        )

    @staticmethod
    def handle_docent_failure(
        operation: str, error: str, user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Docent 관련 오류 처리"""

        return GPTErrorHandler.handle_gpt_failure(
            error_type="generation_failed",
            error_message=f"Docent {operation} failed: {error}",
            context={
                "component": "docent",
                "operation": operation,
                "user_id": user_context.get("user_id", "anonymous"),
                "gallery_item_id": user_context.get("gallery_item_id", "unknown"),
                "artwork_title": user_context.get("artwork_title", ""),
            },
        )

    @staticmethod
    def handle_transition_guidance_failure(
        error: str, guidance_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """전환 안내 생성 오류 처리"""

        return GPTErrorHandler.handle_gpt_failure(
            error_type="generation_failed",
            error_message=f"Transition guidance generation failed: {error}",
            context={
                "component": "transition_guidance",
                "artwork_title": guidance_context.get("artwork_title", ""),
                "emotion_keywords": guidance_context.get("emotion_keywords", []),
                "user_id": guidance_context.get("user_id", "anonymous"),
            },
        )

    @staticmethod
    def validate_gpt_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """GPT 응답 유효성 검증"""

        validation_result = {"is_valid": True, "issues": [], "content_quality": "good"}

        if not response.get("success", False):
            validation_result["is_valid"] = False
            validation_result["issues"].append("Response indicates failure")
            validation_result["content_quality"] = "failed"
            return validation_result

        content = response.get("content", "")
        if not content or len(content.strip()) < 10:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Content too short or empty")
            validation_result["content_quality"] = "poor"

        # 토큰 사용량 확인
        token_usage = response.get("token_usage", {})
        if token_usage.get("total_tokens", 0) > 4000:
            validation_result["issues"].append("High token usage")

        # 안전성 기본 체크
        unsafe_keywords = ["error", "failed", "cannot", "unable"]
        if any(keyword in content.lower() for keyword in unsafe_keywords):
            validation_result["issues"].append("Content may indicate processing issues")
            validation_result["content_quality"] = "questionable"

        if validation_result["issues"]:
            validation_result["content_quality"] = "needs_review"

        return validation_result
