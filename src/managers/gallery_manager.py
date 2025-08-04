# src/managers/gallery_manager.py

# ==============================================================================
# 이 파일은 사용자의 감정 여정 데이터를 관리하는 역할을 한다.
# MongoDB를 사용하여 각 여정(일기, 생성된 이미지, 방명록, 도슨트 메시지 등)을
# `GalleryItem` 객체로 저장하고 조회한다. 또한, 생성된 이미지 파일을 파일 시스템에 저장하고 관리한다.
# `ACTTherapySystem`은 이 매니저를 통해 사용자의 미술관 데이터를 생성, 조회, 업데이트한다.
# ==============================================================================

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from PIL import Image
import shutil
from pymongo.database import Database
from bson import ObjectId

logger = logging.getLogger(__name__)


class GalleryItem:
    """미술관 전시 아이템"""

    def __init__(
        self,
        item_id: str,
        user_id: str,
        diary_text: str,
        emotion_keywords: List[str],
        vad_scores: Tuple[float, float, float],
        reflection_prompt: str,
        reflection_image_path: str,
        guestbook_title: str = "",
        guestbook_tags: List[str] = None,
        docent_message: Dict[str, Any] = None,
        message_reactions: List[str] = None,
        guided_question: str = "",
        created_date: str = "",
        coping_style: str = "balanced",
        # GPT 관련 새 필드들
        gpt_prompt_used: bool = True,
        gpt_prompt_tokens: int = 0,
        gpt_docent_used: bool = True,
        gpt_docent_tokens: int = 0,
        prompt_generation_time: float = 0.0,
        prompt_generation_method: str = "gpt",
        docent_generation_method: str = "gpt",
    ):

        self.item_id = item_id
        self.user_id = user_id
        self.diary_text = diary_text
        self.emotion_keywords = emotion_keywords
        self.vad_scores = vad_scores
        self.reflection_prompt = reflection_prompt
        self.reflection_image_path = reflection_image_path
        self.guestbook_title = guestbook_title
        self.guestbook_tags = guestbook_tags or []
        self.docent_message = docent_message or {}
        self.message_reactions = message_reactions or []
        self.guided_question = guided_question
        self.created_date = created_date or datetime.now().isoformat()
        self.coping_style = coping_style

        # GPT 관련 메타데이터
        self.gpt_prompt_used = gpt_prompt_used
        self.gpt_prompt_tokens = gpt_prompt_tokens
        self.gpt_docent_used = gpt_docent_used
        self.gpt_docent_tokens = gpt_docent_tokens
        self.prompt_generation_time = prompt_generation_time
        self.prompt_generation_method = prompt_generation_method
        self.docent_generation_method = docent_generation_method

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "item_id": self.item_id,
            "user_id": self.user_id,
            "diary_text": self.diary_text,
            "emotion_keywords": self.emotion_keywords,
            "vad_scores": self.vad_scores,
            "reflection_prompt": self.reflection_prompt,
            "reflection_image_path": self.reflection_image_path,
            "guestbook_title": self.guestbook_title,
            "guestbook_tags": self.guestbook_tags,
            "docent_message": self.docent_message,
            "message_reactions": self.message_reactions,
            "guided_question": self.guided_question,
            "created_date": self.created_date,
            "coping_style": self.coping_style,
            # GPT 메타데이터
            "gpt_prompt_used": self.gpt_prompt_used,
            "gpt_prompt_tokens": self.gpt_prompt_tokens,
            "gpt_docent_used": self.gpt_docent_used,
            "gpt_docent_tokens": self.gpt_docent_tokens,
            "prompt_generation_time": self.prompt_generation_time,
            "prompt_generation_method": self.prompt_generation_method,
            "docent_generation_method": self.docent_generation_method,
        }

    def get_completion_status(self) -> Dict[str, bool]:
        """각 단계별 완료 상태 반환"""
        return {
            "reflection": bool(self.reflection_image_path),
            "guestbook": bool(self.guestbook_title),
            "docent_message": bool(
                self.docent_message
                and isinstance(self.docent_message, dict)
                and self.docent_message
            ),
            "completed": bool(
                self.docent_message
                and isinstance(self.docent_message, dict)
                and self.docent_message
            ),
        }

    def get_next_step(self) -> str:
        """다음 해야 할 단계 반환"""
        status = self.get_completion_status()

        if not status["reflection"]:
            return "reflection"
        elif not status["guestbook"]:
            return "guestbook"
        elif not status["docent_message"]:
            return "docent_message"
        else:
            return "completed"

    def get_gpt_usage_summary(self) -> Dict[str, Any]:
        """GPT 사용량 요약"""
        return {
            "total_tokens": self.gpt_prompt_tokens + self.gpt_docent_tokens,
            "prompt_tokens": self.gpt_prompt_tokens,
            "docent_tokens": self.gpt_docent_tokens,
            "prompt_method": self.prompt_generation_method,
            "docent_method": self.docent_generation_method,
            "generation_time": self.prompt_generation_time,
            "fully_gpt_generated": self.gpt_prompt_used and self.gpt_docent_used,
        }


class GalleryManager:
    """미술관 데이터 관리자 - MongoDB 기반"""

    def __init__(self, mongodb_client, images_dir: str = "data/gallery_images"):
        self.db: Database = mongodb_client.sync_db
        self.gallery_items = self.db.gallery_items
        self.gallery_visits = self.db.gallery_visits
        self.message_reactions = self.db.message_reactions

        self.images_dir = Path(images_dir)

        # 디렉토리 생성
        self.images_dir.mkdir(parents=True, exist_ok=True)
        (self.images_dir / "reflection").mkdir(exist_ok=True)

        self._ensure_indexes()

    def _ensure_indexes(self):
        """MongoDB 인덱스 확인 및 생성"""
        try:
            # gallery_items 컬렉션 인덱스
            self.gallery_items.create_index("user_id")
            self.gallery_items.create_index("item_id", unique=True)  # UUID 고유 인덱스
            self.gallery_items.create_index("created_date")
            self.gallery_items.create_index([("user_id", 1), ("created_date", -1)])

            # gallery_visits 컬렉션 인덱스
            self.gallery_visits.create_index("user_id")
            self.gallery_visits.create_index("item_id")

            # message_reactions 컬렉션 인덱스
            self.message_reactions.create_index("user_id")
            self.message_reactions.create_index("item_id")

            logger.info("갤러리 MongoDB 인덱스가 확인되었습니다.")
        except Exception as e:
            logger.warning(f"갤러리 인덱스 생성 중 오류: {e}")

    def get_incomplete_journeys(self, user_id: str) -> List[GalleryItem]:
        """미완성 여정 목록 반환"""
        try:
            # docent_message가 비어있는 아이템들 조회 (미완성)
            incomplete_docs = self.gallery_items.find(
                {
                    "user_id": user_id,
                    "$or": [
                        {"docent_message": {"$exists": False}},
                        {"docent_message": {}},
                        {"docent_message": None},
                    ],
                }
            ).sort("created_date", -1)

            incomplete_items = []
            for doc in incomplete_docs:
                item = self._doc_to_gallery_item(doc)
                # 실제로 미완성인지 다시 한번 체크
                if not item.get_completion_status()["completed"]:
                    incomplete_items.append(item)

            return incomplete_items

        except Exception as e:
            logger.error(f"미완성 여정 조회 실패: {e}")
            return []

    def create_gallery_item(
        self,
        user_id: str,
        diary_text: str,
        emotion_keywords: List[str],
        vad_scores: Tuple[float, float, float],
        reflection_prompt: str,
        reflection_image: Image.Image,
        coping_style: str = "balanced",
        gpt_prompt_tokens: int = 0,
        prompt_generation_time: float = 0.0,
    ) -> str:
        """새 미술관 아이템 생성 (ACT 1-2단계 완료 후)"""

        # 반영 이미지 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reflection_filename = f"{user_id}_{timestamp}_reflection.png"
        reflection_path = self.images_dir / "reflection" / reflection_filename
        reflection_image.save(reflection_path)

        # MongoDB 문서 생성
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        item_id = f"{user_id}-{timestamp}"  # user-timestamp 형식
        item_doc = {
            "item_id": item_id,  # readable ID 추가
            "user_id": user_id,
            "diary_text": diary_text,
            "emotion_keywords": emotion_keywords,
            "vad_scores": list(vad_scores),
            "reflection_prompt": reflection_prompt,
            "reflection_image_path": str(reflection_path),
            "guestbook_title": "",
            "guestbook_tags": [],
            "docent_message": {},
            "message_reactions": [],
            "guided_question": "",
            "created_date": now.isoformat(),
            "coping_style": coping_style,
            # GPT 메타데이터
            "gpt_prompt_used": True,
            "gpt_prompt_tokens": gpt_prompt_tokens,
            "gpt_docent_used": True,
            "gpt_docent_tokens": 0,  # 아직 생성안됨
            "prompt_generation_time": prompt_generation_time,
            "prompt_generation_method": "gpt",
            "docent_generation_method": "gpt",
        }

        try:
            result = self.gallery_items.insert_one(item_doc)
            mongo_id = str(result.inserted_id)

            logger.info(f"새 미술관 아이템이 생성되었습니다: {item_id}")
            return item_id  # UUID 반환

        except Exception as e:
            logger.error(f"미술관 아이템 생성 실패: {e}")
            raise

    def complete_guestbook(
        self,
        item_id: str,
        guestbook_title: str,
        guestbook_tags: List[str],
        guided_question: str,
    ) -> bool:
        """방명록 작성 완료 (ACT 3단계 완료)"""

        try:
            result = self.gallery_items.update_one(
                {"item_id": item_id},
                {
                    "$set": {
                        "guestbook_title": guestbook_title,
                        "guestbook_tags": guestbook_tags,
                        "guided_question": guided_question,
                    }
                },
            )

            success = result.modified_count > 0
            if success:
                logger.info(f"방명록 작성이 완료되었습니다: 아이템 {item_id}")

            return success

        except Exception as e:
            logger.error(f"방명록 작성 완료 실패: {e}")
            return False

    def add_docent_message(self, item_id: str, docent_message: Dict[str, Any]) -> bool:
        """도슨트 메시지 추가 (ACT 4단계 완료)"""

        try:
            # GPT 메타데이터 추출
            metadata = docent_message.get("metadata", {})
            gpt_docent_tokens = metadata.get("token_usage", {}).get("total_tokens", 0)

            # 데이터베이스 업데이트
            result = self.gallery_items.update_one(
                {"item_id": item_id},
                {
                    "$set": {
                        "docent_message": docent_message,
                        "gpt_docent_tokens": gpt_docent_tokens,
                    }
                },
            )

            success = result.modified_count > 0
            if success:
                logger.info(f"도슨트 메시지가 추가되었습니다: 아이템 {item_id}")

            return success

        except Exception as e:
            logger.error(f"도슨트 메시지 추가 실패: {e}")
            return False

    def record_message_reaction(
        self, item_id: str, reaction_type: str, reaction_data: Dict[str, Any] = None
    ) -> bool:
        """메시지 반응 기록"""

        try:
            # 아이템 조회
            item_doc = self.gallery_items.find_one({"_id": ObjectId(item_id)})
            if not item_doc:
                logger.error(f"미술관 아이템을 찾을 수 없습니다: {item_id}")
                return False

            # 반응 기록 컬렉션에 저장
            reaction_doc = {
                "user_id": item_doc["user_id"],
                "item_id": item_id,
                "reaction_type": reaction_type,
                "reaction_data": reaction_data or {},
                "reaction_date": datetime.now().isoformat(),
            }

            self.message_reactions.insert_one(reaction_doc)

            # 아이템의 반응 목록 업데이트
            current_reactions = item_doc.get("message_reactions", [])
            current_reactions.append(reaction_type)

            self.gallery_items.update_one(
                {"_id": ObjectId(item_id)},
                {"$set": {"message_reactions": current_reactions}},
            )

            logger.info(
                f"메시지 반응이 기록되었습니다: {reaction_type} - 아이템 {item_id}"
            )
            return True

        except Exception as e:
            logger.error(f"메시지 반응 기록 실패: {e}")
            return False

    def get_gallery_item(self, item_id: str) -> Optional[GalleryItem]:
        """미술관 아이템 조회"""
        try:
            # Try to find by ObjectId first
            try:
                item_doc = self.gallery_items.find_one({"_id": ObjectId(item_id)})
                if item_doc:
                    return self._doc_to_gallery_item(item_doc)
            except:
                pass

            # If ObjectId fails, try to find by item_id field (UUID)
            item_doc = self.gallery_items.find_one({"item_id": item_id})

            if not item_doc:
                return None

            return self._doc_to_gallery_item(item_doc)

        except Exception as e:
            logger.error(f"미술관 아이템 조회 실패: {e}")
            return None

    def get_user_gallery(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[GalleryItem]:
        """사용자 미술관 조회"""
        try:
            query = {"user_id": user_id}

            # 날짜 필터링
            if date_from or date_to:
                date_filter = {}
                if date_from:
                    date_filter["$gte"] = date_from
                if date_to:
                    date_filter["$lte"] = date_to
                query["created_date"] = date_filter

            cursor = (
                self.gallery_items.find(query)
                .sort("created_date", -1)
                .skip(offset)
                .limit(limit)
            )

            return [self._doc_to_gallery_item(doc) for doc in cursor]

        except Exception as e:
            logger.error(f"사용자 미술관 조회 실패: {e}")
            return []

    def _doc_to_gallery_item(self, doc: Dict[str, Any]) -> GalleryItem:
        """MongoDB 문서를 GalleryItem 객체로 변환"""

        return GalleryItem(
            item_id=str(doc["_id"]),
            user_id=doc.get("user_id", ""),
            diary_text=doc.get("diary_text", ""),
            emotion_keywords=doc.get("emotion_keywords", []),
            vad_scores=tuple(doc.get("vad_scores", [0.0, 0.0, 0.0])),
            reflection_prompt=doc.get("reflection_prompt", ""),
            reflection_image_path=doc.get("reflection_image_path", ""),
            guestbook_title=doc.get("guestbook_title", ""),
            guestbook_tags=doc.get("guestbook_tags", []),
            docent_message=doc.get("docent_message", {}),
            message_reactions=doc.get("message_reactions", []),
            guided_question=doc.get("guided_question", ""),
            created_date=doc.get("created_date", ""),
            coping_style=doc.get("coping_style", "balanced"),
            # GPT 메타데이터
            gpt_prompt_used=doc.get("gpt_prompt_used", True),
            gpt_prompt_tokens=doc.get("gpt_prompt_tokens", 0),
            gpt_docent_used=doc.get("gpt_docent_used", True),
            gpt_docent_tokens=doc.get("gpt_docent_tokens", 0),
            prompt_generation_time=doc.get("prompt_generation_time", 0.0),
            prompt_generation_method=doc.get("prompt_generation_method", "gpt"),
            docent_generation_method=doc.get("docent_generation_method", "gpt"),
        )

    def get_gpt_usage_analytics(self, user_id: str) -> Dict[str, Any]:
        """사용자별 GPT 사용량 분석"""
        try:
            # 기본 통계 조회
            total_items = self.gallery_items.count_documents({"user_id": user_id})

            if total_items == 0:
                return {
                    "user_id": user_id,
                    "total_items": 0,
                    "gpt_adoption_rate": 0.0,
                    "error": "데이터 없음",
                }

            # Aggregation으로 GPT 사용 통계 계산
            pipeline = [
                {"$match": {"user_id": user_id}},
                {
                    "$group": {
                        "_id": None,
                        "total_items": {"$sum": 1},
                        "gpt_prompts": {"$sum": {"$cond": ["$gpt_prompt_used", 1, 0]}},
                        "gpt_docents": {"$sum": {"$cond": ["$gpt_docent_used", 1, 0]}},
                        "total_prompt_tokens": {"$sum": "$gpt_prompt_tokens"},
                        "total_docent_tokens": {"$sum": "$gpt_docent_tokens"},
                        "avg_generation_time": {"$avg": "$prompt_generation_time"},
                        "first_usage": {"$min": "$created_date"},
                        "latest_usage": {"$max": "$created_date"},
                    }
                },
            ]

            stats_result = list(self.gallery_items.aggregate(pipeline))
            if not stats_result:
                return {"user_id": user_id, "total_items": 0, "error": "집계 실패"}

            stats = stats_result[0]

            # 생성 방법별 분포
            prompt_method_pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {"_id": "$prompt_generation_method", "count": {"$sum": 1}}},
            ]
            prompt_methods = {
                doc["_id"]: doc["count"]
                for doc in self.gallery_items.aggregate(prompt_method_pipeline)
            }

            docent_method_pipeline = [
                {"$match": {"user_id": user_id, "docent_message": {"$ne": {}}}},
                {"$group": {"_id": "$docent_generation_method", "count": {"$sum": 1}}},
            ]
            docent_methods = {
                doc["_id"]: doc["count"]
                for doc in self.gallery_items.aggregate(docent_method_pipeline)
            }

            return {
                "user_id": user_id,
                "total_items": stats["total_items"],
                "gpt_usage": {
                    "prompt_generations": stats["gpt_prompts"],
                    "docent_generations": stats["gpt_docents"],
                    "full_gpt_journeys": min(
                        stats["gpt_prompts"], stats["gpt_docents"]
                    ),
                },
                "token_usage": {
                    "total_tokens": stats["total_prompt_tokens"]
                    + stats["total_docent_tokens"],
                    "prompt_tokens": stats["total_prompt_tokens"],
                    "docent_tokens": stats["total_docent_tokens"],
                    "avg_tokens_per_item": (
                        (stats["total_prompt_tokens"] + stats["total_docent_tokens"])
                        / stats["total_items"]
                    ),
                },
                "performance": {
                    "avg_generation_time": stats["avg_generation_time"] or 0.0,
                    "gpt_adoption_rate": stats["gpt_prompts"] / stats["total_items"],
                    "completion_rate": (
                        stats["gpt_docents"] / stats["gpt_prompts"]
                        if stats["gpt_prompts"] > 0
                        else 0.0
                    ),
                },
                "method_distribution": {
                    "prompt_methods": prompt_methods,
                    "docent_methods": docent_methods,
                },
                "timeline": {
                    "first_gpt_usage": stats["first_usage"],
                    "latest_gpt_usage": stats["latest_usage"],
                },
            }

        except Exception as e:
            logger.error(f"GPT 사용량 분석 실패: {e}")
            return {"user_id": user_id, "error": str(e)}

    def record_gallery_visit(
        self, user_id: str, item_id: str, visit_type: str, viewing_duration: float = 0.0
    ):
        """미술관 방문 기록"""
        try:
            visit_doc = {
                "user_id": user_id,
                "item_id": item_id,
                "visit_type": visit_type,
                "visit_date": datetime.now().isoformat(),
                "viewing_duration": viewing_duration,
            }

            self.gallery_visits.insert_one(visit_doc)

            logger.info(
                f"미술관 방문 기록: 사용자 {user_id}, 아이템 {item_id}, 유형 {visit_type}"
            )

        except Exception as e:
            logger.error(f"미술관 방문 기록 실패: {e}")

    def get_gallery_analytics(self, user_id: str) -> Dict[str, Any]:
        """미술관 분석 데이터 - GPT 통계 포함"""
        try:
            # 기본 통계
            total_items = self.gallery_items.count_documents({"user_id": user_id})

            if total_items == 0:
                return {"user_id": user_id, "total_items": 0, "error": "데이터 없음"}

            # 날짜 범위 조회
            date_pipeline = [
                {"$match": {"user_id": user_id}},
                {
                    "$group": {
                        "_id": None,
                        "first_date": {"$min": "$created_date"},
                        "last_date": {"$max": "$created_date"},
                    }
                },
            ]
            date_result = list(self.gallery_items.aggregate(date_pipeline))
            first_date = date_result[0]["first_date"] if date_result else None
            last_date = date_result[0]["last_date"] if date_result else None

            # 감정 분포 데이터
            vad_docs = list(
                self.gallery_items.find({"user_id": user_id}, {"vad_scores": 1})
            )
            vad_data = [doc["vad_scores"] for doc in vad_docs if "vad_scores" in doc]

            # 방명록 제목들
            title_docs = list(
                self.gallery_items.find(
                    {"user_id": user_id, "guestbook_title": {"$ne": ""}},
                    {"guestbook_title": 1},
                )
            )
            titles = [doc["guestbook_title"] for doc in title_docs]

            # 방문 패턴 분석
            visit_pipeline = [
                {"$match": {"user_id": user_id}},
                {
                    "$group": {
                        "_id": "$visit_type",
                        "count": {"$sum": 1},
                        "avg_duration": {"$avg": "$viewing_duration"},
                    }
                },
            ]
            visit_patterns = {}
            for doc in self.gallery_visits.aggregate(visit_pipeline):
                visit_patterns[doc["_id"]] = {
                    "count": doc["count"],
                    "avg_duration": doc["avg_duration"],
                }

            # GPT 사용량 분석 추가
            gpt_analytics = self.get_gpt_usage_analytics(user_id)

            analytics = {
                "user_id": user_id,
                "total_items": total_items,
                "date_range": {
                    "first_item": first_date,
                    "last_item": last_date,
                    "span_days": self._calculate_date_span(first_date, last_date),
                },
                "emotion_trends": self._analyze_emotion_trends(vad_data),
                "title_sentiments": self._analyze_title_sentiments(titles),
                "visit_patterns": visit_patterns,
                "completion_rate": self._calculate_completion_rate(user_id),
                "growth_insights": self._generate_growth_insights(vad_data),
                "gpt_analytics": gpt_analytics,
            }

            return analytics

        except Exception as e:
            logger.error(f"미술관 분석 데이터 조회 실패: {e}")
            return {"user_id": user_id, "error": str(e)}

    def get_message_reaction_analytics(self, user_id: str) -> Dict[str, Any]:
        """메시지 반응 분석 - GPT 메시지 성능 포함"""
        try:
            # 반응 유형별 집계
            reaction_pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {"_id": "$reaction_type", "count": {"$sum": 1}}},
            ]
            reaction_counts = {
                doc["_id"]: doc["count"]
                for doc in self.message_reactions.aggregate(reaction_pipeline)
            }

            # GPT 메시지별 반응 분석 (lookup 사용)
            gpt_reaction_pipeline = [
                {"$match": {"user_id": user_id}},
                {
                    "$lookup": {
                        "from": "gallery_items",
                        "localField": "item_id",
                        "foreignField": "_id",
                        "as": "gallery_item",
                    }
                },
                {"$unwind": "$gallery_item"},
                {
                    "$group": {
                        "_id": {
                            "method": "$gallery_item.curator_generation_method",
                            "reaction": "$reaction_type",
                        },
                        "count": {"$sum": 1},
                    }
                },
            ]

            gpt_reaction_data = {}
            for doc in self.message_reactions.aggregate(gpt_reaction_pipeline):
                method = doc["_id"]["method"]
                reaction = doc["_id"]["reaction"]
                count = doc["count"]

                if method not in gpt_reaction_data:
                    gpt_reaction_data[method] = {}
                gpt_reaction_data[method][reaction] = count

            # 시간별 반응 패턴
            daily_pipeline = [
                {"$match": {"user_id": user_id}},
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": {
                                    "$dateFromString": {"dateString": "$reaction_date"}
                                },
                            }
                        },
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"_id": -1}},
                {"$limit": 30},
            ]
            daily_reactions = {
                doc["_id"]: doc["count"]
                for doc in self.message_reactions.aggregate(daily_pipeline)
            }

            # 긍정적 반응률 계산
            total_reactions = sum(reaction_counts.values())
            positive_reactions = sum(
                reaction_counts.get(rt, 0) for rt in ["like", "save", "share"]
            )

            # GPT 메시지 성능 계산
            gpt_performance = self._calculate_gpt_message_performance(gpt_reaction_data)

            return {
                "total_reactions": total_reactions,
                "reaction_distribution": reaction_counts,
                "positive_reaction_rate": (
                    positive_reactions / total_reactions if total_reactions > 0 else 0
                ),
                "daily_reaction_pattern": daily_reactions,
                "engagement_level": self._calculate_engagement_level(
                    total_reactions, positive_reactions
                ),
                "gpt_message_performance": gpt_performance,
            }

        except Exception as e:
            logger.error(f"메시지 반응 분석 실패: {e}")
            return {"user_id": user_id, "error": str(e)}

    def _calculate_gpt_message_performance(
        self, gpt_reaction_data: Dict
    ) -> Dict[str, Any]:
        """GPT 메시지 성능 계산"""
        performance = {}

        for method, reactions in gpt_reaction_data.items():
            total = sum(reactions.values())
            positive = sum(reactions.get(rt, 0) for rt in ["like", "save", "share"])

            performance[method] = {
                "total_reactions": total,
                "positive_reactions": positive,
                "positive_rate": positive / total if total > 0 else 0,
                "reaction_distribution": reactions,
            }

        # GPT 전체 성능
        if "gpt" in performance:
            gpt_perf = performance["gpt"]
            performance["gpt_effectiveness"] = {
                "adoption_success": gpt_perf["positive_rate"] > 0.6,
                "user_satisfaction": (
                    "high"
                    if gpt_perf["positive_rate"] > 0.7
                    else "medium" if gpt_perf["positive_rate"] > 0.5 else "low"
                ),
                "recommendation": self._get_gpt_performance_recommendation(
                    gpt_perf["positive_rate"]
                ),
            }

        return performance

    def _get_gpt_performance_recommendation(self, positive_rate: float) -> str:
        """GPT 성능 기반 권장사항"""
        if positive_rate > 0.8:
            return "GPT 도슨트 메시지가 매우 효과적입니다. 현재 설정을 유지하세요."
        elif positive_rate > 0.6:
            return "GPT 메시지가 적절히 작동하고 있습니다. 개인화 수준을 높여보세요."
        elif positive_rate > 0.4:
            return (
                "GPT 메시지 품질 개선이 필요합니다. 프롬프트 엔지니어링을 검토하세요."
            )
        else:
            return "GPT 메시지 성능이 낮습니다. 시스템 설정과 안전성 검증을 점검하세요."

    def _calculate_engagement_level(
        self, total_reactions: int, positive_reactions: int
    ) -> str:
        """참여도 수준 계산"""
        if total_reactions == 0:
            return "새로운 사용자"

        positive_rate = positive_reactions / total_reactions

        if positive_rate >= 0.8 and total_reactions >= 10:
            return "매우 높음"
        elif positive_rate >= 0.6 and total_reactions >= 5:
            return "높음"
        elif positive_rate >= 0.4:
            return "보통"
        else:
            return "낮음"

    def _analyze_emotion_trends(self, vad_data: List[List[float]]) -> Dict[str, Any]:
        """감정 트렌드 분석"""
        if not vad_data:
            return {}

        valences = [d[0] for d in vad_data]
        arousals = [d[1] for d in vad_data]
        dominances = [d[2] for d in vad_data]

        return {
            "valence": {
                "avg": sum(valences) / len(valences),
                "trend": (
                    "improving"
                    if len(valences) > 1 and valences[-1] > valences[0]
                    else "stable"
                ),
            },
            "arousal": {
                "avg": sum(arousals) / len(arousals),
                "variability": max(arousals) - min(arousals) if arousals else 0,
            },
            "dominance": {
                "avg": sum(dominances) / len(dominances),
                "recent_change": (
                    dominances[-1] - dominances[0] if len(dominances) > 1 else 0
                ),
            },
        }

    def _analyze_title_sentiments(self, titles: List[str]) -> Dict[str, Any]:
        """방명록 제목 감정 분석"""
        if not titles:
            return {}

        positive_words = {
            "light",
            "bright",
            "hope",
            "peace",
            "joy",
            "calm",
            "beautiful",
        }
        negative_words = {"dark", "heavy", "storm", "sad", "grey", "empty", "cold"}

        positive_count = 0
        negative_count = 0

        for title in titles:
            title_lower = title.lower()
            if any(word in title_lower for word in positive_words):
                positive_count += 1
            elif any(word in title_lower for word in negative_words):
                negative_count += 1

        return {
            "total_titles": len(titles),
            "positive_titles": positive_count,
            "negative_titles": negative_count,
            "neutral_titles": len(titles) - positive_count - negative_count,
            "positivity_ratio": positive_count / len(titles) if titles else 0,
        }

    def _calculate_completion_rate(self, user_id: str) -> float:
        """완료율 계산 (docent_message 기준)"""
        try:
            total = self.gallery_items.count_documents({"user_id": user_id})
            completed = self.gallery_items.count_documents(
                {"user_id": user_id, "docent_message": {"$ne": {}}}
            )

            return completed / total if total > 0 else 0.0

        except Exception as e:
            logger.error(f"완료율 계산 실패: {e}")
            return 0.0

    def _generate_growth_insights(self, vad_data: List[List[float]]) -> List[str]:
        """성장 인사이트 생성"""
        insights = []

        if len(vad_data) < 2:
            return ["더 많은 데이터가 쌓이면 성장 패턴을 분석할 수 있습니다."]

        # 최근과 초기 비교
        recent_valence = sum(d[0] for d in vad_data[-3:]) / min(3, len(vad_data))
        initial_valence = sum(d[0] for d in vad_data[:3]) / min(3, len(vad_data))

        if recent_valence > initial_valence + 0.1:
            insights.append(
                "GPT 개인화를 통해 감정 상태가 전반적으로 개선되고 있습니다."
            )
        elif recent_valence < initial_valence - 0.1:
            insights.append("최근 감정적 어려움이 있는 것 같습니다.")
        else:
            insights.append("감정 상태가 안정적으로 유지되고 있습니다.")

        # 변동성 분석
        valence_var = sum((d[0] - recent_valence) ** 2 for d in vad_data) / len(
            vad_data
        )
        if valence_var < 0.1:
            insights.append("감정 변동성이 낮아 안정적입니다.")
        else:
            insights.append("감정 기복이 있지만 이는 자연스러운 현상입니다.")

        return insights

    def _calculate_date_span(self, first_date: str, last_date: str) -> int:
        """날짜 범위 계산"""
        if not first_date or not last_date:
            return 0

        try:
            first = datetime.fromisoformat(first_date.replace("Z", "+00:00"))
            last = datetime.fromisoformat(last_date.replace("Z", "+00:00"))
            return (last - first).days
        except:
            return 0

    def export_user_gallery(self, user_id: str, export_dir: str) -> Dict[str, Any]:
        """사용자 미술관 데이터 내보내기"""
        try:
            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)

            # 미술관 아이템들 조회
            items = self.get_user_gallery(user_id, limit=1000)

            # GPT 사용량 분석 추가
            gpt_analytics = self.get_gpt_usage_analytics(user_id)

            # JSON 데이터 생성
            export_data = {
                "user_id": user_id,
                "export_date": datetime.now().isoformat(),
                "total_items": len(items),
                "items": [item.to_dict() for item in items],
                "analytics": self.get_gallery_analytics(user_id),
                "message_analytics": self.get_message_reaction_analytics(user_id),
                "gpt_usage_analytics": gpt_analytics,
                "system_version": "mongodb_integrated",
            }

            # JSON 파일 저장
            json_path = export_path / f"{user_id}_gallery_export_mongodb.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            # 이미지 파일들 복사
            images_copied = 0
            for item in items:
                if (
                    item.reflection_image_path
                    and Path(item.reflection_image_path).exists()
                ):
                    try:
                        shutil.copy2(item.reflection_image_path, export_path)
                        images_copied += 1
                    except Exception as e:
                        logger.warning(
                            f"이미지 복사 실패: {item.reflection_image_path}, {e}"
                        )

            result = {
                "success": True,
                "export_path": str(export_path),
                "json_file": str(json_path),
                "items_exported": len(items),
                "images_copied": images_copied,
                "mongodb_data_included": True,
                "total_gpt_tokens": gpt_analytics.get("token_usage", {}).get(
                    "total_tokens", 0
                ),
            }

            logger.info(
                f"사용자 {user_id}의 미술관 데이터 내보내기 완료 (MongoDB): {export_path}"
            )
            return result

        except Exception as e:
            logger.error(f"미술관 데이터 내보내기 실패: {e}")
            return {"success": False, "error": str(e)}

    def cleanup_old_items(self, days_old: int = 365) -> int:
        """오래된 아이템 정리"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()

            # 삭제할 아이템들의 이미지 경로 조회
            old_items = list(
                self.gallery_items.find(
                    {"created_date": {"$lt": cutoff_date}}, {"reflection_image_path": 1}
                )
            )

            image_paths = [
                item.get("reflection_image_path")
                for item in old_items
                if item.get("reflection_image_path")
            ]

            # 데이터베이스에서 삭제
            delete_result = self.gallery_items.delete_many(
                {"created_date": {"$lt": cutoff_date}}
            )
            deleted_count = delete_result.deleted_count

            # 관련 컬렉션에서도 삭제
            self.gallery_visits.delete_many({"visit_date": {"$lt": cutoff_date}})
            self.message_reactions.delete_many({"reaction_date": {"$lt": cutoff_date}})

            # 이미지 파일 삭제
            for img_path in image_paths:
                try:
                    Path(img_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"이미지 삭제 실패: {img_path}, {e}")

            logger.info(f"오래된 미술관 아이템 {deleted_count}개가 정리되었습니다.")
            return deleted_count

        except Exception as e:
            logger.error(f"오래된 아이템 정리 실패: {e}")
            return 0

    def get_system_status(self) -> Dict[str, Any]:
        """갤러리 시스템 상태 확인"""
        try:
            # 전체 통계
            total_items = self.gallery_items.count_documents({})

            # GPT 사용률
            fully_gpt_items = self.gallery_items.count_documents(
                {"gpt_prompt_used": True, "gpt_docent_used": True}
            )

            return {
                "database_ready": True,
                "mongodb_migration_complete": True,
                "total_items": total_items,
                "gpt_adoption_rate": (
                    fully_gpt_items / total_items if total_items > 0 else 1.0
                ),
                "supports_gpt_metadata": True,
                "fallback_systems": False,
            }

        except Exception as e:
            logger.error(f"갤러리 시스템 상태 확인 실패: {e}")
            return {"database_ready": False, "error": str(e)}
