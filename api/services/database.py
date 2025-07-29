# api/services/database.py

from supabase import create_client, Client
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json
from api.config import settings

logger = logging.getLogger(__name__)


class SupabaseService:
    """Supabase 데이터베이스 서비스 (SQLite 완전 대체)"""

    def __init__(self):
        try:
            self.supabase: Client = create_client(
                settings.supabase_url, settings.supabase_anon_key
            )
            logger.info("✅ Supabase 클라이언트 초기화 완료")
        except Exception as e:
            logger.error(f"❌ Supabase 클라이언트 초기화 실패: {e}")
            raise

    # === 사용자 관리 ===
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """신규 사용자 생성"""
        try:
            # created_date를 현재 시간으로 설정
            if "created_date" not in user_data:
                user_data["created_date"] = datetime.utcnow().isoformat()

            result = self.supabase.table("users").insert(user_data).execute()

            if result.data:
                logger.info(f"✅ 사용자 생성 완료: {user_data.get('user_id')}")
                return result.data[0]
            else:
                raise Exception("No data returned from insert")

        except Exception as e:
            logger.error(f"❌ 사용자 생성 실패: {e}")
            raise

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """사용자 조회"""
        try:
            result = (
                self.supabase.table("users")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )

            if result.data:
                logger.debug(f"✅ 사용자 조회 완료: {user_id}")
                return result.data[0]
            else:
                logger.debug(f"🔍 사용자 없음: {user_id}")
                return None

        except Exception as e:
            logger.error(f"❌ 사용자 조회 실패: {e}")
            raise

    async def update_user(
        self, user_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """사용자 정보 업데이트"""
        try:
            # updated_date 자동 설정
            updates["updated_date"] = datetime.utcnow().isoformat()

            result = (
                self.supabase.table("users")
                .update(updates)
                .eq("user_id", user_id)
                .execute()
            )

            if result.data:
                logger.info(f"✅ 사용자 업데이트 완료: {user_id}")
                return result.data[0]
            else:
                raise Exception("No data returned from update")

        except Exception as e:
            logger.error(f"❌ 사용자 업데이트 실패: {e}")
            raise

    async def delete_user(self, user_id: str) -> bool:
        """사용자 삭제"""
        try:
            result = (
                self.supabase.table("users").delete().eq("user_id", user_id).execute()
            )
            logger.info(f"✅ 사용자 삭제 완료: {user_id}")
            return True

        except Exception as e:
            logger.error(f"❌ 사용자 삭제 실패: {e}")
            raise

    # === 심리검사 관리 ===
    async def save_assessment(
        self, user_id: str, assessment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """심리검사 결과 저장"""
        try:
            data = {
                "user_id": user_id,
                "phq9_score": assessment_data.get("phq9_score"),
                "cesd_score": assessment_data.get("cesd_score"),
                "meaq_score": assessment_data.get("meaq_score"),
                "ciss_score": assessment_data.get("ciss_score"),
                "coping_style": assessment_data.get("coping_style"),
                "severity_level": assessment_data.get("severity_level"),
                "interpretation": json.dumps(assessment_data.get("interpretation", {})),
                "recommendations": json.dumps(
                    assessment_data.get("recommendations", [])
                ),
                "created_date": datetime.utcnow().isoformat(),
            }

            result = self.supabase.table("psychometric_results").insert(data).execute()

            if result.data:
                logger.info(f"✅ 심리검사 결과 저장 완료: {user_id}")
                return result.data[0]
            else:
                raise Exception("No data returned from insert")

        except Exception as e:
            logger.error(f"❌ 심리검사 결과 저장 실패: {e}")
            raise

    async def get_assessment(self, user_id: str) -> Optional[Dict[str, Any]]:
        """심리검사 결과 조회 (최신)"""
        try:
            result = (
                self.supabase.table("psychometric_results")
                .select("*")
                .eq("user_id", user_id)
                .order("created_date", desc=True)
                .limit(1)
                .execute()
            )

            if result.data:
                assessment = result.data[0]
                # JSON 필드 파싱
                if assessment.get("interpretation"):
                    assessment["interpretation"] = json.loads(
                        assessment["interpretation"]
                    )
                if assessment.get("recommendations"):
                    assessment["recommendations"] = json.loads(
                        assessment["recommendations"]
                    )

                logger.debug(f"✅ 심리검사 결과 조회 완료: {user_id}")
                return assessment
            else:
                logger.debug(f"🔍 심리검사 결과 없음: {user_id}")
                return None

        except Exception as e:
            logger.error(f"❌ 심리검사 결과 조회 실패: {e}")
            raise

    async def get_assessment_history(
        self, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """심리검사 이력 조회"""
        try:
            result = (
                self.supabase.table("psychometric_results")
                .select("*")
                .eq("user_id", user_id)
                .order("created_date", desc=True)
                .limit(limit)
                .execute()
            )

            assessments = result.data or []

            # JSON 필드 파싱
            for assessment in assessments:
                if assessment.get("interpretation"):
                    assessment["interpretation"] = json.loads(
                        assessment["interpretation"]
                    )
                if assessment.get("recommendations"):
                    assessment["recommendations"] = json.loads(
                        assessment["recommendations"]
                    )

            logger.debug(
                f"✅ 심리검사 이력 조회 완료: {user_id} ({len(assessments)}개)"
            )
            return assessments

        except Exception as e:
            logger.error(f"❌ 심리검사 이력 조회 실패: {e}")
            raise

    # === 갤러리 아이템 관리 ===
    async def create_gallery_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """갤러리 아이템 생성"""
        try:
            # JSON 필드들 직렬화
            data = {
                "user_id": item_data["user_id"],
                "diary_text": item_data.get("diary_text"),
                "emotion_keywords": json.dumps(item_data.get("emotion_keywords", [])),
                "vad_scores": json.dumps(item_data.get("vad_scores", [])),
                "image_prompt": item_data.get("image_prompt"),
                "image_path": item_data.get("image_path"),
                "guestbook_title": item_data.get("guestbook_title"),
                "guestbook_content": item_data.get("guestbook_content"),
                "guestbook_tags": json.dumps(item_data.get("guestbook_tags", [])),
                "curator_message": item_data.get("curator_message"),
                "completion_status": item_data.get(
                    "completion_status", "moment_completed"
                ),
                "created_date": datetime.utcnow().isoformat(),
            }

            result = self.supabase.table("gallery_items").insert(data).execute()

            if result.data:
                gallery_item = result.data[0]
                # JSON 필드 파싱
                self._parse_gallery_item_json_fields(gallery_item)

                logger.info(f"✅ 갤러리 아이템 생성 완료: {gallery_item['id']}")
                return gallery_item
            else:
                raise Exception("No data returned from insert")

        except Exception as e:
            logger.error(f"❌ 갤러리 아이템 생성 실패: {e}")
            raise

    async def get_gallery_items(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """사용자의 갤러리 아이템 목록 조회"""
        try:
            result = (
                self.supabase.table("gallery_items")
                .select("*")
                .eq("user_id", user_id)
                .order("created_date", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )

            items = result.data or []

            # JSON 필드 파싱
            for item in items:
                self._parse_gallery_item_json_fields(item)

            logger.debug(f"✅ 갤러리 아이템 목록 조회 완료: {user_id} ({len(items)}개)")
            return items

        except Exception as e:
            logger.error(f"❌ 갤러리 아이템 목록 조회 실패: {e}")
            raise

    async def get_gallery_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """특정 갤러리 아이템 조회"""
        try:
            result = (
                self.supabase.table("gallery_items")
                .select("*")
                .eq("id", item_id)
                .execute()
            )

            if result.data:
                item = result.data[0]
                self._parse_gallery_item_json_fields(item)

                logger.debug(f"✅ 갤러리 아이템 조회 완료: {item_id}")
                return item
            else:
                logger.debug(f"🔍 갤러리 아이템 없음: {item_id}")
                return None

        except Exception as e:
            logger.error(f"❌ 갤러리 아이템 조회 실패: {e}")
            raise

    async def update_gallery_item(
        self, item_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """갤러리 아이템 업데이트"""
        try:
            # JSON 필드들 직렬화
            serialized_updates = {}
            for key, value in updates.items():
                if (
                    key in ["emotion_keywords", "vad_scores", "guestbook_tags"]
                    and value is not None
                ):
                    serialized_updates[key] = json.dumps(value)
                else:
                    serialized_updates[key] = value

            result = (
                self.supabase.table("gallery_items")
                .update(serialized_updates)
                .eq("id", item_id)
                .execute()
            )

            if result.data:
                item = result.data[0]
                self._parse_gallery_item_json_fields(item)

                logger.info(f"✅ 갤러리 아이템 업데이트 완료: {item_id}")
                return item
            else:
                raise Exception("No data returned from update")

        except Exception as e:
            logger.error(f"❌ 갤러리 아이템 업데이트 실패: {e}")
            raise

    async def delete_gallery_item(self, item_id: str) -> bool:
        """갤러리 아이템 삭제"""
        try:
            result = (
                self.supabase.table("gallery_items")
                .delete()
                .eq("id", item_id)
                .execute()
            )
            logger.info(f"✅ 갤러리 아이템 삭제 완료: {item_id}")
            return True

        except Exception as e:
            logger.error(f"❌ 갤러리 아이템 삭제 실패: {e}")
            raise

    # === 시각 선호도 관리 ===
    async def save_visual_preferences(
        self, user_id: str, preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """시각 선호도 저장/업데이트"""
        try:
            # 기존 선호도가 있는지 확인
            existing = (
                self.supabase.table("visual_preferences")
                .select("id")
                .eq("user_id", user_id)
                .execute()
            )

            data = {
                "user_id": user_id,
                "style_preferences": json.dumps(
                    preferences.get("style_preferences", {})
                ),
                "color_preferences": json.dumps(
                    preferences.get("color_preferences", {})
                ),
                "complexity_level": preferences.get("complexity_level", 5),
                "updated_date": datetime.utcnow().isoformat(),
            }

            if existing.data:
                # 업데이트
                result = (
                    self.supabase.table("visual_preferences")
                    .update(data)
                    .eq("user_id", user_id)
                    .execute()
                )
                logger.info(f"✅ 시각 선호도 업데이트 완료: {user_id}")
            else:
                # 신규 생성
                result = (
                    self.supabase.table("visual_preferences").insert(data).execute()
                )
                logger.info(f"✅ 시각 선호도 생성 완료: {user_id}")

            if result.data:
                pref = result.data[0]
                # JSON 필드 파싱
                if pref.get("style_preferences"):
                    pref["style_preferences"] = json.loads(pref["style_preferences"])
                if pref.get("color_preferences"):
                    pref["color_preferences"] = json.loads(pref["color_preferences"])

                return pref
            else:
                raise Exception("No data returned from operation")

        except Exception as e:
            logger.error(f"❌ 시각 선호도 저장 실패: {e}")
            raise

    async def get_visual_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """시각 선호도 조회"""
        try:
            result = (
                self.supabase.table("visual_preferences")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )

            if result.data:
                pref = result.data[0]
                # JSON 필드 파싱
                if pref.get("style_preferences"):
                    pref["style_preferences"] = json.loads(pref["style_preferences"])
                if pref.get("color_preferences"):
                    pref["color_preferences"] = json.loads(pref["color_preferences"])

                logger.debug(f"✅ 시각 선호도 조회 완료: {user_id}")
                return pref
            else:
                logger.debug(f"🔍 시각 선호도 없음: {user_id}")
                return None

        except Exception as e:
            logger.error(f"❌ 시각 선호도 조회 실패: {e}")
            raise

    # === 개인화 데이터 관리 ===
    async def save_personalization_data(
        self, user_id: str, interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """개인화 학습 데이터 저장"""
        try:
            data = {
                "user_id": user_id,
                "interaction_type": interaction_data.get("interaction_type"),
                "feedback_data": json.dumps(interaction_data.get("feedback_data", {})),
                "learning_weights": json.dumps(
                    interaction_data.get("learning_weights", {})
                ),
                "created_date": datetime.utcnow().isoformat(),
            }

            result = self.supabase.table("personalization_data").insert(data).execute()

            if result.data:
                logger.debug(f"✅ 개인화 데이터 저장 완료: {user_id}")
                return result.data[0]
            else:
                raise Exception("No data returned from insert")

        except Exception as e:
            logger.error(f"❌ 개인화 데이터 저장 실패: {e}")
            raise

    async def get_personalization_data(
        self, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """개인화 학습 데이터 조회"""
        try:
            result = (
                self.supabase.table("personalization_data")
                .select("*")
                .eq("user_id", user_id)
                .order("created_date", desc=True)
                .limit(limit)
                .execute()
            )

            data_list = result.data or []

            # JSON 필드 파싱
            for data in data_list:
                if data.get("feedback_data"):
                    data["feedback_data"] = json.loads(data["feedback_data"])
                if data.get("learning_weights"):
                    data["learning_weights"] = json.loads(data["learning_weights"])

            logger.debug(f"✅ 개인화 데이터 조회 완료: {user_id} ({len(data_list)}개)")
            return data_list

        except Exception as e:
            logger.error(f"❌ 개인화 데이터 조회 실패: {e}")
            raise

    # === 비용 추적 ===
    async def log_cost(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """API 비용 로깅"""
        try:
            data = {
                "user_id": cost_data.get("user_id"),
                "service_type": cost_data.get("service_type"),
                "tokens_used": cost_data.get("tokens_used"),
                "cost_usd": cost_data.get("cost_usd"),
                "api_call_metadata": json.dumps(cost_data.get("api_call_metadata", {})),
                "created_date": datetime.utcnow().isoformat(),
            }

            result = self.supabase.table("cost_tracking").insert(data).execute()

            if result.data:
                logger.debug(f"✅ 비용 로깅 완료: {cost_data.get('service_type')}")
                return result.data[0]
            else:
                raise Exception("No data returned from insert")

        except Exception as e:
            logger.error(f"❌ 비용 로깅 실패: {e}")
            raise

    async def get_cost_summary(
        self, user_id: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """비용 요약 조회"""
        try:
            # 날짜 필터링을 위한 계산
            from datetime import timedelta

            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

            query = (
                self.supabase.table("cost_tracking")
                .select("*")
                .gte("created_date", cutoff_date)
            )

            if user_id:
                query = query.eq("user_id", user_id)

            result = query.execute()

            costs = result.data or []

            # 비용 집계
            total_cost = sum(float(cost.get("cost_usd", 0)) for cost in costs)
            total_tokens = sum(int(cost.get("tokens_used", 0)) for cost in costs)

            service_breakdown = {}
            for cost in costs:
                service = cost.get("service_type", "unknown")
                if service not in service_breakdown:
                    service_breakdown[service] = {"cost": 0, "tokens": 0, "calls": 0}

                service_breakdown[service]["cost"] += float(cost.get("cost_usd", 0))
                service_breakdown[service]["tokens"] += int(cost.get("tokens_used", 0))
                service_breakdown[service]["calls"] += 1

            summary = {
                "period_days": days,
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "total_api_calls": len(costs),
                "service_breakdown": service_breakdown,
                "user_id": user_id,
            }

            logger.debug(f"✅ 비용 요약 조회 완료: ${total_cost:.4f} ({days}일간)")
            return summary

        except Exception as e:
            logger.error(f"❌ 비용 요약 조회 실패: {e}")
            raise

    # === 유틸리티 메서드 ===
    def _parse_gallery_item_json_fields(self, item: Dict[str, Any]) -> None:
        """갤러리 아이템의 JSON 필드들 파싱"""
        json_fields = ["emotion_keywords", "vad_scores", "guestbook_tags"]

        for field in json_fields:
            if item.get(field) and isinstance(item[field], str):
                try:
                    item[field] = json.loads(item[field])
                except json.JSONDecodeError:
                    logger.warning(f"⚠️  JSON 파싱 실패: {field}")
                    item[field] = []

    async def get_connection_status(self) -> Dict[str, Any]:
        """연결 상태 확인"""
        try:
            # 간단한 쿼리로 연결 테스트
            result = self.supabase.table("users").select("count").limit(1).execute()

            return {
                "status": "connected",
                "database": "supabase",
                "url": settings.supabase_url,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ 데이터베이스 연결 상태 확인 실패: {e}")
            return {
                "status": "disconnected",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }


# 전역 데이터베이스 인스턴스
db = SupabaseService()
