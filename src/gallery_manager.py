# src/gallery_manager.py

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from PIL import Image
import shutil

logger = logging.getLogger(__name__)


class GalleryItem:
    """미술관 전시 아이템"""

    def __init__(
        self,
        item_id: int,
        user_id: str,
        diary_text: str,
        emotion_keywords: List[str],
        vad_scores: Tuple[float, float, float],
        reflection_prompt: str,
        reflection_image_path: str,
        guestbook_title: str = "",
        guestbook_tags: List[str] = None,
        curator_message: Dict[str, Any] = None,
        message_reactions: List[str] = None,
        guided_question: str = "",
        created_date: str = "",
        coping_style: str = "balanced",
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
        self.curator_message = curator_message or {}
        self.message_reactions = message_reactions or []
        self.guided_question = guided_question
        self.created_date = created_date or datetime.now().isoformat()
        self.coping_style = coping_style

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
            "curator_message": self.curator_message,
            "message_reactions": self.message_reactions,
            "guided_question": self.guided_question,
            "created_date": self.created_date,
            "coping_style": self.coping_style,
        }

    def get_completion_status(self) -> Dict[str, bool]:
        """각 단계별 완료 상태 반환"""
        return {
            "reflection": bool(self.reflection_image_path),
            "guestbook": bool(self.guestbook_title),
            "curator_message": bool(
                self.curator_message
                and isinstance(self.curator_message, dict)
                and self.curator_message
            ),
            "completed": bool(
                self.curator_message
                and isinstance(self.curator_message, dict)
                and self.curator_message
            ),
        }

    def get_next_step(self) -> str:
        """다음 해야 할 단계 반환"""
        status = self.get_completion_status()

        if not status["reflection"]:
            return "reflection"
        elif not status["guestbook"]:
            return "guestbook"
        elif not status["curator_message"]:
            return "curator_message"
        else:
            return "completed"


class GalleryManager:
    """미술관 데이터 관리자"""

    def __init__(
        self, db_path: str = "data/gallery.db", images_dir: str = "data/gallery_images"
    ):
        self.db_path = Path(db_path)
        self.images_dir = Path(images_dir)

        # 디렉토리 생성
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # 이미지 저장 서브디렉토리
        (self.images_dir / "reflection").mkdir(exist_ok=True)

        self._init_database()

    def _init_database(self):
        """데이터베이스 초기화 및 마이그레이션"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 1. 기본 테이블 생성
        self._create_base_tables(cursor)

        # 2. 마이그레이션 실행
        self._run_migrations(cursor)

        conn.commit()
        conn.close()
        logger.info("미술관 데이터베이스가 초기화되었습니다.")

    def _create_base_tables(self, cursor):
        """기본 테이블 생성"""

        # 미술관 아이템 테이블
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS gallery_items (
                item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                diary_text TEXT NOT NULL,
                emotion_keywords TEXT,  -- JSON 배열
                vad_scores TEXT,        -- JSON [valence, arousal, dominance]
                reflection_prompt TEXT,
                reflection_image_path TEXT,
                guestbook_title TEXT,
                guestbook_tags TEXT,    -- JSON 배열
                curator_message TEXT,   -- JSON 큐레이터 메시지
                message_reactions TEXT, -- JSON 메시지 반응들
                guided_question TEXT,
                created_date TEXT,
                coping_style TEXT
            )
        """
        )

        # 인덱스 생성
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_gallery_user_date ON gallery_items(user_id, created_date)"
        )

        # 미술관 방문 기록
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS gallery_visits (
                visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id INTEGER,
                visit_type TEXT,  -- view_reflection, view_message, revisit
                visit_date TEXT,
                viewing_duration REAL,  -- 초 단위
                FOREIGN KEY (item_id) REFERENCES gallery_items (item_id)
            )
        """
        )

        # 메시지 반응 기록 테이블
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS message_reactions (
                reaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id INTEGER NOT NULL,
                reaction_type TEXT NOT NULL,  -- like, save, share, dismiss
                reaction_data TEXT,  -- JSON 추가 반응 데이터
                reaction_date TEXT,
                FOREIGN KEY (item_id) REFERENCES gallery_items (item_id)
            )
        """
        )

        # 사용자 미술관 통계 (기본 컬럼만)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS gallery_stats (
                user_id TEXT PRIMARY KEY,
                total_items INTEGER DEFAULT 0,
                first_item_date TEXT,
                last_item_date TEXT,
                total_visits INTEGER DEFAULT 0,
                avg_viewing_duration REAL DEFAULT 0.0,
                favorite_coping_style TEXT,
                updated_date TEXT
            )
        """
        )

    def _run_migrations(self, cursor):
        """데이터베이스 마이그레이션"""

        # 마이그레이션 추적 테이블
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS migrations (
                migration_id TEXT PRIMARY KEY,
                executed_date TEXT
            )
        """
        )

        # Migration 1: gallery_stats에 메시지 반응 컬럼 추가
        cursor.execute(
            "SELECT name FROM pragma_table_info('gallery_stats') WHERE name='total_message_reactions'"
        )
        if not cursor.fetchone():
            logger.info("Migration 1: gallery_stats에 메시지 반응 컬럼 추가")
            cursor.execute(
                "ALTER TABLE gallery_stats ADD COLUMN total_message_reactions INTEGER DEFAULT 0"
            )
            cursor.execute(
                "ALTER TABLE gallery_stats ADD COLUMN positive_reaction_rate REAL DEFAULT 0.0"
            )

            cursor.execute(
                "INSERT OR REPLACE INTO migrations (migration_id, executed_date) VALUES (?, ?)",
                ("add_message_reaction_columns", datetime.now().isoformat()),
            )

        # Migration 2: 기존 데이터의 누락된 컬럼들 기본값으로 채우기
        cursor.execute(
            "SELECT migration_id FROM migrations WHERE migration_id = ?",
            ("fill_missing_columns",),
        )
        if not cursor.fetchone():
            logger.info("Migration 2: 기존 데이터 정리")

            # curator_message와 message_reactions가 NULL인 경우 기본값 설정
            cursor.execute(
                """
                UPDATE gallery_items 
                SET curator_message = '{}', message_reactions = '[]' 
                WHERE curator_message IS NULL OR message_reactions IS NULL
                """
            )

            cursor.execute(
                "INSERT OR REPLACE INTO migrations (migration_id, executed_date) VALUES (?, ?)",
                ("fill_missing_columns", datetime.now().isoformat()),
            )

    def get_incomplete_journeys(self, user_id: str) -> List[GalleryItem]:
        """미완성 여정 목록 반환"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # curator_message가 비어있는 아이템들 조회 (미완성)
        cursor.execute(
            """
            SELECT * FROM gallery_items 
            WHERE user_id = ? AND (curator_message IS NULL OR curator_message = '' OR curator_message = '{}')
            ORDER BY created_date DESC
            """,
            (user_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        incomplete_items = []
        for row in rows:
            item = self._row_to_gallery_item(row)
            # 실제로 미완성인지 다시 한번 체크
            if not item.get_completion_status()["completed"]:
                incomplete_items.append(item)

        return incomplete_items

    def create_gallery_item(
        self,
        user_id: str,
        diary_text: str,
        emotion_keywords: List[str],
        vad_scores: Tuple[float, float, float],
        reflection_prompt: str,
        reflection_image: Image.Image,
        coping_style: str = "balanced",
    ) -> int:
        """새 미술관 아이템 생성 (ACT 1-2단계 완료 후)"""

        # 반영 이미지 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reflection_filename = f"{user_id}_{timestamp}_reflection.png"
        reflection_path = self.images_dir / "reflection" / reflection_filename
        reflection_image.save(reflection_path)

        # 데이터베이스에 저장
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO gallery_items 
            (user_id, diary_text, emotion_keywords, vad_scores, 
             reflection_prompt, reflection_image_path, curator_message, message_reactions, created_date, coping_style)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user_id,
                diary_text,
                json.dumps(emotion_keywords),
                json.dumps(vad_scores),
                reflection_prompt,
                str(reflection_path),
                "{}",  # 빈 curator_message
                "[]",  # 빈 message_reactions
                datetime.now().isoformat(),
                coping_style,
            ),
        )

        item_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # 통계 업데이트
        self._update_user_stats(user_id)

        logger.info(f"새 미술관 아이템이 생성되었습니다: {item_id}")
        return item_id

    def complete_guestbook(
        self,
        item_id: int,
        guestbook_title: str,
        guestbook_tags: List[str],
        guided_question: str,
    ) -> bool:
        """방명록 작성 완료 (ACT 3단계 완료)"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE gallery_items 
            SET guestbook_title = ?, guestbook_tags = ?, guided_question = ?
            WHERE item_id = ?
        """,
            (guestbook_title, json.dumps(guestbook_tags), guided_question, item_id),
        )

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if success:
            logger.info(f"방명록 작성이 완료되었습니다: 아이템 {item_id}")

        return success

    def add_curator_message(
        self, item_id: int, curator_message: Dict[str, Any]
    ) -> bool:
        """큐레이터 메시지 추가 (ACT 4단계 완료)"""

        # 기존 아이템 정보 조회
        item = self.get_gallery_item(item_id)
        if not item:
            logger.error(f"미술관 아이템을 찾을 수 없습니다: {item_id}")
            return False

        # 데이터베이스 업데이트
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE gallery_items 
            SET curator_message = ?
            WHERE item_id = ?
        """,
            (json.dumps(curator_message), item_id),
        )

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if success:
            logger.info(f"큐레이터 메시지가 추가되었습니다: 아이템 {item_id}")

        return success

    def record_message_reaction(
        self, item_id: int, reaction_type: str, reaction_data: Dict[str, Any] = None
    ) -> bool:
        """메시지 반응 기록"""

        # 기존 아이템 조회
        item = self.get_gallery_item(item_id)
        if not item:
            logger.error(f"미술관 아이템을 찾을 수 없습니다: {item_id}")
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 반응 기록 테이블에 저장
        cursor.execute(
            """
            INSERT INTO message_reactions 
            (user_id, item_id, reaction_type, reaction_data, reaction_date)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                item.user_id,
                item_id,
                reaction_type,
                json.dumps(reaction_data) if reaction_data else "{}",
                datetime.now().isoformat(),
            ),
        )

        # 아이템의 반응 목록 업데이트
        current_reactions = item.message_reactions.copy()
        current_reactions.append(reaction_type)

        cursor.execute(
            """
            UPDATE gallery_items 
            SET message_reactions = ?
            WHERE item_id = ?
        """,
            (json.dumps(current_reactions), item_id),
        )

        conn.commit()
        conn.close()

        # 통계 업데이트
        self._update_user_stats(item.user_id)

        logger.info(f"메시지 반응이 기록되었습니다: {reaction_type} - 아이템 {item_id}")
        return True

    def get_gallery_item(self, item_id: int) -> Optional[GalleryItem]:
        """미술관 아이템 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM gallery_items WHERE item_id = ?", (item_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_gallery_item(row)

    def get_user_gallery(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[GalleryItem]:
        """사용자 미술관 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM gallery_items WHERE user_id = ?"
        params = [user_id]

        # 날짜 필터링
        if date_from:
            query += " AND created_date >= ?"
            params.append(date_from)

        if date_to:
            query += " AND created_date <= ?"
            params.append(date_to)

        query += " ORDER BY created_date DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_gallery_item(row) for row in rows]

    def _row_to_gallery_item(self, row) -> GalleryItem:
        """DB 행을 GalleryItem 객체로 변환"""
        return GalleryItem(
            item_id=row[0],
            user_id=row[1],
            diary_text=row[2],
            emotion_keywords=json.loads(row[3]) if row[3] else [],
            vad_scores=tuple(json.loads(row[4])) if row[4] else (0.0, 0.0, 0.0),
            reflection_prompt=row[5] or "",
            reflection_image_path=row[6] or "",
            guestbook_title=row[7] or "",
            guestbook_tags=json.loads(row[8]) if row[8] else [],
            curator_message=json.loads(row[9]) if row[9] and row[9] != "{}" else {},
            message_reactions=(
                json.loads(row[10]) if row[10] and row[10] != "[]" else []
            ),
            guided_question=row[11] or "",
            created_date=row[12] or "",
            coping_style=row[13] or "balanced",
        )

    def record_gallery_visit(
        self, user_id: str, item_id: int, visit_type: str, viewing_duration: float = 0.0
    ):
        """미술관 방문 기록"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO gallery_visits 
            (user_id, item_id, visit_type, visit_date, viewing_duration)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                user_id,
                item_id,
                visit_type,
                datetime.now().isoformat(),
                viewing_duration,
            ),
        )

        conn.commit()
        conn.close()

        logger.info(
            f"미술관 방문 기록: 사용자 {user_id}, 아이템 {item_id}, 유형 {visit_type}"
        )

    def get_gallery_analytics(self, user_id: str) -> Dict[str, Any]:
        """미술관 분석 데이터"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 기본 통계
        cursor.execute(
            """
            SELECT COUNT(*), MIN(created_date), MAX(created_date)
            FROM gallery_items WHERE user_id = ?
        """,
            (user_id,),
        )
        total_items, first_date, last_date = cursor.fetchone()

        # 감정 분포 분석
        cursor.execute(
            """
            SELECT vad_scores FROM gallery_items WHERE user_id = ?
        """,
            (user_id,),
        )
        vad_data = [json.loads(row[0]) for row in cursor.fetchall() if row[0]]

        # 방명록 제목 감정 분석
        cursor.execute(
            """
            SELECT guestbook_title FROM gallery_items 
            WHERE user_id = ? AND guestbook_title != ''
        """,
            (user_id,),
        )
        titles = [row[0] for row in cursor.fetchall()]

        # 방문 패턴 분석
        cursor.execute(
            """
            SELECT visit_type, COUNT(*), AVG(viewing_duration)
            FROM gallery_visits WHERE user_id = ?
            GROUP BY visit_type
        """,
            (user_id,),
        )
        visit_patterns = {
            row[0]: {"count": row[1], "avg_duration": row[2]}
            for row in cursor.fetchall()
        }

        conn.close()

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
        }

        return analytics

    def get_message_reaction_analytics(self, user_id: str) -> Dict[str, Any]:
        """메시지 반응 분석"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 반응 유형별 집계
        cursor.execute(
            """
            SELECT reaction_type, COUNT(*) 
            FROM message_reactions 
            WHERE user_id = ?
            GROUP BY reaction_type
        """,
            (user_id,),
        )
        reaction_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # 시간별 반응 패턴
        cursor.execute(
            """
            SELECT DATE(reaction_date) as date, COUNT(*) 
            FROM message_reactions 
            WHERE user_id = ?
            GROUP BY DATE(reaction_date)
            ORDER BY date DESC
            LIMIT 30
        """,
            (user_id,),
        )
        daily_reactions = {row[0]: row[1] for row in cursor.fetchall()}

        # 긍정적 반응률
        total_reactions = sum(reaction_counts.values())
        positive_reactions = sum(
            reaction_counts.get(rt, 0) for rt in ["like", "save", "share"]
        )

        conn.close()

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
        }

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
        """완료율 계산 (curator_message 기준)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN curator_message != '' AND curator_message != '{}' THEN 1 ELSE 0 END) as completed
            FROM gallery_items WHERE user_id = ?
        """,
            (user_id,),
        )

        total, completed = cursor.fetchone()
        conn.close()

        return completed / total if total > 0 else 0.0

    def _generate_growth_insights(self, vad_data: List[List[float]]) -> List[str]:
        """성장 인사이트 생성"""
        insights = []

        if len(vad_data) < 2:
            return ["더 많은 데이터가 쌓이면 성장 패턴을 분석할 수 있습니다."]

        # 최근과 초기 비교
        recent_valence = sum(d[0] for d in vad_data[-3:]) / min(3, len(vad_data))
        initial_valence = sum(d[0] for d in vad_data[:3]) / min(3, len(vad_data))

        if recent_valence > initial_valence + 0.1:
            insights.append("감정 상태가 전반적으로 개선되고 있습니다.")
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

    def _update_user_stats(self, user_id: str):
        """사용자 통계 업데이트"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 현재 통계 계산
        cursor.execute(
            """
            SELECT COUNT(*), MIN(created_date), MAX(created_date)
            FROM gallery_items WHERE user_id = ?
        """,
            (user_id,),
        )
        total_items, first_date, last_date = cursor.fetchone()

        # 방문 통계
        cursor.execute(
            """
            SELECT COUNT(*), AVG(viewing_duration)
            FROM gallery_visits WHERE user_id = ?
        """,
            (user_id,),
        )
        visit_result = cursor.fetchone()
        total_visits = visit_result[0] if visit_result[0] else 0
        avg_duration = visit_result[1] if visit_result[1] else 0.0

        # 메시지 반응 통계
        cursor.execute(
            """
            SELECT COUNT(*), 
                   SUM(CASE WHEN reaction_type IN ('like', 'save', 'share') THEN 1 ELSE 0 END)
            FROM message_reactions WHERE user_id = ?
        """,
            (user_id,),
        )
        reaction_result = cursor.fetchone()
        total_reactions = reaction_result[0] if reaction_result[0] else 0
        positive_reactions = reaction_result[1] if reaction_result[1] else 0

        positive_rate = (
            positive_reactions / total_reactions if total_reactions > 0 else 0.0
        )

        # 선호 대처 스타일
        cursor.execute(
            """
            SELECT coping_style, COUNT(*) as cnt
            FROM gallery_items WHERE user_id = ?
            GROUP BY coping_style ORDER BY cnt DESC LIMIT 1
        """,
            (user_id,),
        )

        fav_style_result = cursor.fetchone()
        fav_style = fav_style_result[0] if fav_style_result else "balanced"

        # 통계 테이블 업데이트 (컬럼 존재 여부 체크)
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO gallery_stats
                (user_id, total_items, first_item_date, last_item_date, 
                 total_visits, avg_viewing_duration, favorite_coping_style, 
                 total_message_reactions, positive_reaction_rate, updated_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    total_items,
                    first_date,
                    last_date,
                    total_visits,
                    avg_duration,
                    fav_style,
                    total_reactions,
                    positive_rate,
                    datetime.now().isoformat(),
                ),
            )
        except sqlite3.OperationalError as e:
            # 컬럼이 없는 경우 기본 컬럼만 업데이트
            logger.warning(f"일부 통계 컬럼 업데이트 실패: {e}")
            cursor.execute(
                """
                INSERT OR REPLACE INTO gallery_stats
                (user_id, total_items, first_item_date, last_item_date, 
                 total_visits, avg_viewing_duration, favorite_coping_style, updated_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    total_items,
                    first_date,
                    last_date,
                    total_visits,
                    avg_duration,
                    fav_style,
                    datetime.now().isoformat(),
                ),
            )

        conn.commit()
        conn.close()

    def export_user_gallery(self, user_id: str, export_dir: str) -> Dict[str, Any]:
        """사용자 미술관 데이터 내보내기"""
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        # 미술관 아이템들 조회
        items = self.get_user_gallery(user_id, limit=1000)

        # JSON 데이터 생성
        export_data = {
            "user_id": user_id,
            "export_date": datetime.now().isoformat(),
            "total_items": len(items),
            "items": [item.to_dict() for item in items],
            "analytics": self.get_gallery_analytics(user_id),
            "message_analytics": self.get_message_reaction_analytics(user_id),
        }

        # JSON 파일 저장
        json_path = export_path / f"{user_id}_gallery_export.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        # 이미지 파일들 복사
        images_copied = 0
        for item in items:
            if item.reflection_image_path and Path(item.reflection_image_path).exists():
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
        }

        logger.info(f"사용자 {user_id}의 미술관 데이터 내보내기 완료: {export_path}")
        return result

    def cleanup_old_items(self, days_old: int = 365) -> int:
        """오래된 아이템 정리"""
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 삭제할 아이템들의 이미지 경로 조회
        cursor.execute(
            """
            SELECT reflection_image_path
            FROM gallery_items WHERE created_date < ?
        """,
            (cutoff_date,),
        )

        image_paths = [row[0] for row in cursor.fetchall() if row[0]]

        # 데이터베이스에서 삭제
        cursor.execute(
            "DELETE FROM gallery_items WHERE created_date < ?", (cutoff_date,)
        )
        deleted_count = cursor.rowcount

        cursor.execute(
            "DELETE FROM gallery_visits WHERE visit_date < ?", (cutoff_date,)
        )

        cursor.execute(
            "DELETE FROM message_reactions WHERE reaction_date < ?", (cutoff_date,)
        )

        conn.commit()
        conn.close()

        # 이미지 파일 삭제
        for img_path in image_paths:
            try:
                Path(img_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"이미지 삭제 실패: {img_path}, {e}")

        logger.info(f"오래된 미술관 아이템 {deleted_count}개가 정리되었습니다.")
        return deleted_count
