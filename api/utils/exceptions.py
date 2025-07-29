# api/utils/exceptions.py

from typing import Any, Dict, Optional, List
from fastapi import HTTPException, status
from datetime import datetime


class EmoseumAPIException(HTTPException):
    """Emoseum API 기본 예외 클래스"""

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code or self.__class__.__name__.upper()
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """예외를 딕셔너리로 변환"""
        return {
            "success": False,
            "error_code": self.error_code,
            "error_message": self.detail,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "status_code": self.status_code,
        }


class AuthenticationError(EmoseumAPIException):
    """인증 실패 예외"""

    def __init__(
        self,
        detail: str = "Authentication failed",
        error_code: str = "AUTHENTICATION_FAILED",
        metadata: Dict[str, Any] = None,
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code=error_code,
            metadata=metadata,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(EmoseumAPIException):
    """권한 부족 예외"""

    def __init__(
        self,
        detail: str = "Insufficient permissions",
        required_permission: str = None,
        user_permissions: List[str] = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        if required_permission:
            metadata["required_permission"] = required_permission
        if user_permissions:
            metadata["user_permissions"] = user_permissions

        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="AUTHORIZATION_FAILED",
            metadata=metadata,
        )


class ValidationError(EmoseumAPIException):
    """데이터 검증 실패 예외"""

    def __init__(
        self,
        detail: str = "Validation failed",
        field_errors: Dict[str, List[str]] = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        if field_errors:
            metadata["field_errors"] = field_errors

        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="VALIDATION_ERROR",
            metadata=metadata,
        )


class ResourceNotFoundError(EmoseumAPIException):
    """리소스를 찾을 수 없음 예외"""

    def __init__(
        self,
        resource_type: str = "Resource",
        resource_id: str = None,
        detail: str = None,
        metadata: Dict[str, Any] = None,
    ):
        if not detail:
            detail = f"{resource_type} not found"
            if resource_id:
                detail += f": {resource_id}"

        if not metadata:
            metadata = {}

        metadata.update({"resource_type": resource_type, "resource_id": resource_id})

        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code="RESOURCE_NOT_FOUND",
            metadata=metadata,
        )


class ConflictError(EmoseumAPIException):
    """리소스 충돌 예외"""

    def __init__(
        self,
        detail: str = "Resource conflict",
        conflict_field: str = None,
        conflict_value: str = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        if conflict_field:
            metadata["conflict_field"] = conflict_field
        if conflict_value:
            metadata["conflict_value"] = conflict_value

        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="RESOURCE_CONFLICT",
            metadata=metadata,
        )


class RateLimitError(EmoseumAPIException):
    """Rate Limit 초과 예외"""

    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        limit: int = None,
        window_seconds: int = None,
        retry_after: int = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        metadata.update(
            {
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after": retry_after,
            }
        )

        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)

        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
            metadata=metadata,
            headers=headers,
        )


class ServiceUnavailableError(EmoseumAPIException):
    """서비스 이용 불가 예외"""

    def __init__(
        self,
        detail: str = "Service temporarily unavailable",
        service_name: str = None,
        estimated_recovery: str = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        metadata.update(
            {"service_name": service_name, "estimated_recovery": estimated_recovery}
        )

        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="SERVICE_UNAVAILABLE",
            metadata=metadata,
        )


class TherapySessionError(EmoseumAPIException):
    """치료 세션 관련 예외"""

    def __init__(
        self,
        detail: str = "Therapy session error",
        session_stage: str = None,
        journey_id: str = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        metadata.update({"session_stage": session_stage, "journey_id": journey_id})

        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="THERAPY_SESSION_ERROR",
            metadata=metadata,
        )


class PersonalizationError(EmoseumAPIException):
    """개인화 시스템 관련 예외"""

    def __init__(
        self,
        detail: str = "Personalization system error",
        personalization_level: int = None,
        required_data: List[str] = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        metadata.update(
            {
                "personalization_level": personalization_level,
                "required_data": required_data,
            }
        )

        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="PERSONALIZATION_ERROR",
            metadata=metadata,
        )


class ImageGenerationError(EmoseumAPIException):
    """이미지 생성 관련 예외"""

    def __init__(
        self,
        detail: str = "Image generation failed",
        backend: str = None,
        prompt_length: int = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        metadata.update({"backend": backend, "prompt_length": prompt_length})

        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="IMAGE_GENERATION_ERROR",
            metadata=metadata,
        )


class GPTServiceError(EmoseumAPIException):
    """GPT 서비스 관련 예외"""

    def __init__(
        self,
        detail: str = "GPT service error",
        service_type: str = None,
        token_usage: int = None,
        cost_usd: float = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        metadata.update(
            {
                "service_type": service_type,
                "token_usage": token_usage,
                "cost_usd": cost_usd,
            }
        )

        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="GPT_SERVICE_ERROR",
            metadata=metadata,
        )


class DatabaseError(EmoseumAPIException):
    """데이터베이스 관련 예외"""

    def __init__(
        self,
        detail: str = "Database operation failed",
        operation: str = None,
        table: str = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        metadata.update({"operation": operation, "table": table})

        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="DATABASE_ERROR",
            metadata=metadata,
        )


class EmergencyInterventionRequired(EmoseumAPIException):
    """응급 개입 필요 예외"""

    def __init__(
        self,
        detail: str = "Emergency intervention required",
        risk_level: str = "high",
        crisis_resources: List[Dict[str, str]] = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        metadata.update(
            {
                "risk_level": risk_level,
                "crisis_resources": crisis_resources
                or [
                    {
                        "name": "National Suicide Prevention Lifeline",
                        "phone": "988",
                        "available": "24/7",
                    }
                ],
                "immediate_action_required": True,
            }
        )

        super().__init__(
            status_code=status.HTTP_423_LOCKED,
            detail=detail,
            error_code="EMERGENCY_INTERVENTION_REQUIRED",
            metadata=metadata,
        )


class ConfigurationError(EmoseumAPIException):
    """설정 오류 예외"""

    def __init__(
        self,
        detail: str = "System configuration error",
        config_key: str = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        if config_key:
            metadata["config_key"] = config_key

        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="CONFIGURATION_ERROR",
            metadata=metadata,
        )


class QuotaExceededError(EmoseumAPIException):
    """할당량 초과 예외"""

    def __init__(
        self,
        detail: str = "Quota exceeded",
        quota_type: str = None,
        current_usage: int = None,
        quota_limit: int = None,
        reset_time: str = None,
        metadata: Dict[str, Any] = None,
    ):
        if not metadata:
            metadata = {}

        metadata.update(
            {
                "quota_type": quota_type,
                "current_usage": current_usage,
                "quota_limit": quota_limit,
                "reset_time": reset_time,
            }
        )

        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="QUOTA_EXCEEDED",
            metadata=metadata,
        )


# 편의 함수들
def raise_authentication_error(detail: str = None, **kwargs):
    """인증 오류 발생"""
    raise AuthenticationError(detail=detail or "Authentication required", **kwargs)


def raise_authorization_error(detail: str = None, **kwargs):
    """권한 오류 발생"""
    raise AuthorizationError(detail=detail or "Insufficient permissions", **kwargs)


def raise_not_found(resource_type: str, resource_id: str = None):
    """리소스를 찾을 수 없음 오류 발생"""
    raise ResourceNotFoundError(resource_type=resource_type, resource_id=resource_id)


def raise_validation_error(
    detail: str = None, field_errors: Dict[str, List[str]] = None
):
    """검증 오류 발생"""
    raise ValidationError(
        detail=detail or "Validation failed", field_errors=field_errors
    )


def raise_conflict_error(
    detail: str = None, conflict_field: str = None, conflict_value: str = None
):
    """충돌 오류 발생"""
    raise ConflictError(
        detail=detail or "Resource conflict",
        conflict_field=conflict_field,
        conflict_value=conflict_value,
    )


def raise_service_unavailable(service_name: str = None, estimated_recovery: str = None):
    """서비스 이용 불가 오류 발생"""
    raise ServiceUnavailableError(
        detail=f"{service_name or 'Service'} is temporarily unavailable",
        service_name=service_name,
        estimated_recovery=estimated_recovery,
    )


def raise_emergency_intervention(
    risk_level: str = "high", crisis_resources: List[Dict[str, str]] = None
):
    """응급 개입 필요 오류 발생"""
    raise EmergencyInterventionRequired(
        risk_level=risk_level, crisis_resources=crisis_resources
    )
