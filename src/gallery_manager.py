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
        hope_prompt: str = "",
        hope_image_path: str = "",
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
        self.hope_prompt = hope_prompt
        self.hope_image_path = hope_image_path
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
            "hope_prompt": self.hope_prompt,
            "hope_image_path": self.hope_image_path,
            "guided_question": self.guided_question,
            "created_date": self.created_date,
            "coping_style": self.coping_style,
        }


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
        (self.images_dir / "hope").mkdir(exist_ok=True)

        self._init_database()

    def _init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

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
                hope_prompt TEXT,
                hope_image_path TEXT,
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
                visit_type TEXT,  -- view_reflection, view_hope, revisit
                visit_date TEXT,
                viewing_duration REAL,  -- 초 단위
                FOREIGN KEY (item_id) REFERENCES gallery_items (item_id)
            )
        """
        )

        # 사용자 미술관 통계
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

        conn.commit()
        conn.close()
        logger.info("미술관 데이터베이스가 초기화되었습니다.")

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
             reflection_prompt, reflection_image_path, created_date, coping_style)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user_id,
                diary_text,
                json.dumps(emotion_keywords),
                json.dumps(vad_scores),
                reflection_prompt,
                str(reflection_path),
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

    def add_hope_image(
        self, item_id: int, hope_prompt: str, hope_image: Image.Image
    ) -> bool:
        """희망 이미지 추가 (ACT 4단계 완료)"""

        # 기존 아이템 정보 조회
        item = self.get_gallery_item(item_id)
        if not item:
            logger.error(f"미술관 아이템을 찾을 수 없습니다: {item_id}")
            return False

        # 희망 이미지 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hope_filename = f"{item.user_id}_{timestamp}_hope.png"
        hope_path = self.images_dir / "hope" / hope_filename
        hope_image.save(hope_path)

        # 데이터베이스 업데이트
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE gallery_items 
            SET hope_prompt = ?, hope_image_path = ?
            WHERE item_id = ?
        """,
            (hope_prompt, str(hope_path), item_id),
        )

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if success:
            logger.info(f"희망 이미지가 추가되었습니다: 아이템 {item_id}")

        return success

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
            hope_prompt=row[9] or "",
            hope_image_path=row[10] or "",
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

        # 간단한 감정 분석 (실제 구현에서는 더 정교한 분석 도구 사용 가능)
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
        """완료율 계산"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN hope_image_path != '' THEN 1 ELSE 0 END) as completed
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

        # 통계 테이블 업데이트
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
        }

        # JSON 파일 저장
        json_path = export_path / f"{user_id}_gallery_export.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        # 이미지 파일들 복사
        images_copied = 0
        for item in items:
            for img_path in [item.reflection_image_path, item.hope_image_path]:
                if img_path and Path(img_path).exists():
                    try:
                        shutil.copy2(img_path, export_path)
                        images_copied += 1
                    except Exception as e:
                        logger.warning(f"이미지 복사 실패: {img_path}, {e}")

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
            SELECT reflection_image_path, hope_image_path
            FROM gallery_items WHERE created_date < ?
        """,
            (cutoff_date,),
        )

        image_paths = []
        for row in cursor.fetchall():
            if row[0]:
                image_paths.append(row[0])
            if row[1]:
                image_paths.append(row[1])

        # 데이터베이스에서 삭제
        cursor.execute(
            "DELETE FROM gallery_items WHERE created_date < ?", (cutoff_date,)
        )
        deleted_count = cursor.rowcount

        cursor.execute(
            "DELETE FROM gallery_visits WHERE visit_date < ?", (cutoff_date,)
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
