# src/rule_manager.py

import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CopingStyleRules:
    """대처 스타일별 감정 표현 규칙 관리"""

    def __init__(self, rules_file: Optional[str] = None):
        self.rules_file = Path(rules_file) if rules_file else None

        # 기본 감정 순화 규칙 (회피 성향용)
        self.softening_rules = {
            # 강도 높은 부정적 감정 -> 부드러운 표현
            "overwhelming": "encompassing",
            "crushing": "weighty",
            "devastating": "profound",
            "terrifying": "unsettling",
            "hopeless": "challenging",
            "desperate": "yearning",
            "agonizing": "difficult",
            "unbearable": "intense",
            "suffocating": "confining",
            "nightmare": "troubling dream",
            "torment": "inner struggle",
            "anguish": "deep concern",
            "despair": "low moment",
            "rage": "strong displeasure",
            "fury": "intense frustration",
            "hatred": "strong dislike",
            "disgust": "discomfort",
            "revulsion": "unease",
            "panic": "high anxiety",
            "terror": "worry",
            "dread": "apprehension",
            # 관계/사회적 부정 감정
            "abandoned": "temporarily alone",
            "rejected": "not chosen",
            "betrayed": "disappointed by trust",
            "isolated": "seeking connection",
            "alienated": "feeling different",
            "worthless": "questioning value",
            "inadequate": "room for growth",
            "failure": "learning experience",
            "shame": "self-consciousness",
            "humiliation": "embarrassment",
            # 신체/에너지 관련
            "exhausted": "needing rest",
            "drained": "low energy",
            "weakened": "needing strength",
            "broken": "needing healing",
            "shattered": "fragmented",
            "torn": "conflicted",
            "bleeding": "hurting",
            "wounded": "tender",
        }

        # 강화 규칙 (직면 성향용)
        self.intensifying_rules = {
            "sad": "deeply sorrowful",
            "angry": "burning with anger",
            "worried": "acutely anxious",
            "tired": "profoundly exhausted",
            "lonely": "starkly isolated",
            "confused": "utterly bewildered",
            "hurt": "deeply wounded",
            "frustrated": "intensely aggravated",
            "disappointed": "crushingly let down",
            "stressed": "severely pressured",
            "overwhelmed": "completely inundated",
            "lost": "thoroughly disoriented",
            "empty": "utterly void",
            "heavy": "tremendously burdened",
            "dark": "profoundly shadowed",
            "cold": "bitingly frigid",
            "harsh": "brutally severe",
            "rough": "jaggedly difficult",
        }

        # 균형 조절 규칙 (균형 성향용)
        self.balancing_rules = {
            "overwhelming": "significant",
            "crushing": "heavy",
            "devastating": "impactful",
            "terrifying": "concerning",
            "hopeless": "difficult",
            "desperate": "urgent",
            "agonizing": "painful",
            "unbearable": "challenging",
            "suffocating": "constraining",
            "nightmare": "difficult situation",
            # 반대로 너무 약한 표현도 적절히 조절
            "slightly sad": "melancholic",
            "a bit worried": "concerned",
            "somewhat tired": "fatigued",
            "kind of lonely": "solitary",
            "pretty confused": "uncertain",
        }

        # 문맥별 대체 표현
        self.contextual_alternatives = {
            "emotional_pain": {
                "avoidant": ["gentle difficulty", "quiet struggle", "inner tenderness"],
                "confrontational": ["raw pain", "sharp anguish", "cutting sorrow"],
                "balanced": [
                    "emotional challenge",
                    "inner conflict",
                    "heartfelt difficulty",
                ],
            },
            "relationship_issues": {
                "avoidant": [
                    "connection challenges",
                    "social distance",
                    "interpersonal space",
                ],
                "confrontational": [
                    "relationship crisis",
                    "social rupture",
                    "interpersonal conflict",
                ],
                "balanced": [
                    "relationship difficulty",
                    "social tension",
                    "interpersonal challenge",
                ],
            },
            "self_worth": {
                "avoidant": [
                    "self-questioning",
                    "inner uncertainty",
                    "personal reflection",
                ],
                "confrontational": [
                    "self-doubt crisis",
                    "worth examination",
                    "identity struggle",
                ],
                "balanced": [
                    "self-evaluation",
                    "worth consideration",
                    "personal assessment",
                ],
            },
            "future_anxiety": {
                "avoidant": [
                    "tomorrow's mystery",
                    "future wondering",
                    "upcoming uncertainty",
                ],
                "confrontational": [
                    "future dread",
                    "tomorrow's terror",
                    "upcoming catastrophe",
                ],
                "balanced": [
                    "future concern",
                    "tomorrow's challenge",
                    "upcoming difficulty",
                ],
            },
        }

        # 감정 카테고리별 키워드 매핑
        self.emotion_categories = {
            "sadness": [
                "sad",
                "depressed",
                "melancholy",
                "sorrowful",
                "grief",
                "mourning",
            ],
            "anger": ["angry", "furious", "rage", "irritated", "annoyed", "frustrated"],
            "fear": ["afraid", "scared", "terrified", "anxious", "worried", "panic"],
            "disgust": ["disgusted", "revolted", "repulsed", "sickened", "appalled"],
            "surprise": ["shocked", "astonished", "amazed", "startled", "stunned"],
            "joy": ["happy", "joyful", "elated", "ecstatic", "cheerful", "delighted"],
            "trust": ["trusting", "accepting", "admiring", "grateful", "appreciative"],
            "anticipation": ["excited", "eager", "hopeful", "optimistic", "expectant"],
        }

        # 규칙 파일이 있으면 로드
        if self.rules_file and self.rules_file.exists():
            self.load_rules()

    def apply_coping_style_rules(
        self, text: str, coping_style: str, emotion_keywords: List[str] = None
    ) -> str:
        """대처 스타일에 따른 감정 표현 규칙 적용"""

        if coping_style == "avoidant":
            return self._apply_softening_rules(text, emotion_keywords)
        elif coping_style == "confrontational":
            return self._apply_intensifying_rules(text, emotion_keywords)
        elif coping_style == "balanced":
            return self._apply_balancing_rules(text, emotion_keywords)
        else:
            return text

    def _apply_softening_rules(
        self, text: str, emotion_keywords: List[str] = None
    ) -> str:
        """순화 규칙 적용 (회피 성향)"""
        processed_text = text.lower()

        # 직접적인 키워드 대체
        for harsh, soft in self.softening_rules.items():
            if harsh in processed_text:
                processed_text = processed_text.replace(harsh, soft)

        # 감정 카테고리별 문맥적 대체
        if emotion_keywords:
            for keyword in emotion_keywords:
                category = self._get_emotion_category(keyword)
                if category and category in ["sadness", "anger", "fear", "disgust"]:
                    alternatives = self.contextual_alternatives.get(
                        "emotional_pain", {}
                    )
                    if "avoidant" in alternatives:
                        # 부정적 감정을 더 부드러운 은유로 대체
                        metaphor = alternatives["avoidant"][0]  # 첫 번째 대안 사용
                        processed_text = processed_text.replace(
                            keyword.lower(), metaphor
                        )

        return processed_text

    def _apply_intensifying_rules(
        self, text: str, emotion_keywords: List[str] = None
    ) -> str:
        """강화 규칙 적용 (직면 성향)"""
        processed_text = text.lower()

        # 감정 강도 증가
        for mild, intense in self.intensifying_rules.items():
            if mild in processed_text:
                processed_text = processed_text.replace(mild, intense)

        # 감정 키워드가 있으면 더 구체적이고 생생한 표현으로
        if emotion_keywords:
            for keyword in emotion_keywords:
                if keyword.lower() in processed_text:
                    intensified = self._intensify_keyword(keyword)
                    processed_text = processed_text.replace(
                        keyword.lower(), intensified
                    )

        return processed_text

    def _apply_balancing_rules(
        self, text: str, emotion_keywords: List[str] = None
    ) -> str:
        """균형 조절 규칙 적용 (균형 성향)"""
        processed_text = text.lower()

        # 극단적 표현을 중성적으로 조절
        for extreme, balanced in self.balancing_rules.items():
            if extreme in processed_text:
                processed_text = processed_text.replace(extreme, balanced)

        return processed_text

    def _get_emotion_category(self, keyword: str) -> Optional[str]:
        """감정 키워드의 카테고리 찾기"""
        keyword_lower = keyword.lower()

        for category, words in self.emotion_categories.items():
            if keyword_lower in words:
                return category

        return None

    def _intensify_keyword(self, keyword: str) -> str:
        """키워드 강화"""
        # 이미 강화 규칙에 있으면 사용
        if keyword.lower() in self.intensifying_rules:
            return self.intensifying_rules[keyword.lower()]

        # 아니면 수식어 추가
        intensifiers = ["deeply", "profoundly", "intensely", "acutely", "severely"]
        category = self._get_emotion_category(keyword)

        if category in ["sadness", "fear"]:
            return f"deeply {keyword}"
        elif category in ["anger", "disgust"]:
            return f"intensely {keyword}"
        else:
            return f"profoundly {keyword}"

    def get_contextual_prompt_modifiers(
        self, coping_style: str, emotion_category: str
    ) -> List[str]:
        """대처 스타일과 감정 카테고리에 따른 프롬프트 수식어 반환"""

        base_modifiers = {
            "avoidant": {
                "sadness": [
                    "gentle melancholy",
                    "soft contemplation",
                    "quiet reflection",
                ],
                "anger": ["muted tension", "restrained energy", "contained intensity"],
                "fear": ["cautious uncertainty", "gentle concern", "soft apprehension"],
                "general": ["protective atmosphere", "safe distance", "gentle buffer"],
            },
            "confrontational": {
                "sadness": ["raw sorrow", "unfiltered grief", "stark melancholy"],
                "anger": ["blazing fury", "intense rage", "burning frustration"],
                "fear": ["sharp anxiety", "acute terror", "cutting worry"],
                "general": ["direct confrontation", "unmasked truth", "bare reality"],
            },
            "balanced": {
                "sadness": [
                    "thoughtful sorrow",
                    "considered grief",
                    "balanced melancholy",
                ],
                "anger": [
                    "measured frustration",
                    "controlled intensity",
                    "focused energy",
                ],
                "fear": ["reasonable concern", "balanced worry", "grounded anxiety"],
                "general": [
                    "balanced perspective",
                    "measured approach",
                    "considered view",
                ],
            },
        }

        return base_modifiers.get(coping_style, {}).get(
            emotion_category, base_modifiers.get(coping_style, {}).get("general", [])
        )

    def create_emotion_bridge(
        self, original_emotion: str, target_emotion: str, coping_style: str
    ) -> str:
        """감정 전환 브릿지 생성 (ACT 3단계 -> 4단계)"""

        bridges = {
            "avoidant": {
                "negative_to_hope": [
                    "while gently allowing space for {hope}",
                    "softly opening to possibilities of {hope}",
                    "tenderly making room for {hope}",
                    "quietly welcoming hints of {hope}",
                ],
                "neutral_to_hope": [
                    "gracefully inviting {hope}",
                    "peacefully embracing {hope}",
                    "calmly receiving {hope}",
                ],
            },
            "confrontational": {
                "negative_to_hope": [
                    "boldly transforming {original} into {hope}",
                    "courageously replacing {original} with {hope}",
                    "directly shifting from {original} to {hope}",
                    "powerfully converting {original} into {hope}",
                ],
                "neutral_to_hope": [
                    "actively cultivating {hope}",
                    "intentionally building {hope}",
                    "deliberately fostering {hope}",
                ],
            },
            "balanced": {
                "negative_to_hope": [
                    "thoughtfully transitioning from {original} toward {hope}",
                    "carefully nurturing {hope} alongside {original}",
                    "wisely allowing {hope} to emerge from {original}",
                    "mindfully transforming {original} into {hope}",
                ],
                "neutral_to_hope": [
                    "naturally developing {hope}",
                    "organically growing {hope}",
                    "harmoniously cultivating {hope}",
                ],
            },
        }

        # 원본 감정이 부정적인지 판단
        is_negative = any(
            neg in original_emotion.lower()
            for neg in ["sad", "angry", "fear", "dark", "heavy", "difficult"]
        )

        bridge_type = "negative_to_hope" if is_negative else "neutral_to_hope"
        bridge_templates = bridges.get(coping_style, bridges["balanced"])[bridge_type]

        # 랜덤하게 하나 선택
        import random

        template = random.choice(bridge_templates)

        return template.format(original=original_emotion, hope=target_emotion)

    def validate_rules(self) -> Dict[str, List[str]]:
        """규칙 유효성 검사"""
        issues = {
            "circular_references": [],
            "missing_categories": [],
            "inconsistent_intensity": [],
        }

        # 순환 참조 검사
        for original, replacement in self.softening_rules.items():
            if (
                replacement in self.softening_rules
                and self.softening_rules[replacement] == original
            ):
                issues["circular_references"].append(f"{original} <-> {replacement}")

        # 카테고리 일관성 검사
        for category, keywords in self.emotion_categories.items():
            for keyword in keywords:
                if (
                    keyword not in self.softening_rules
                    and keyword not in self.intensifying_rules
                    and keyword not in self.balancing_rules
                ):
                    issues["missing_categories"].append(f"{category}: {keyword}")

        return {k: v for k, v in issues.items() if v}  # 빈 리스트 제거

    def save_rules(self, file_path: Optional[str] = None):
        """규칙을 JSON 파일로 저장"""
        save_path = Path(file_path) if file_path else self.rules_file
        if not save_path:
            logger.warning("저장할 파일 경로가 지정되지 않았습니다.")
            return

        rules_data = {
            "softening_rules": self.softening_rules,
            "intensifying_rules": self.intensifying_rules,
            "balancing_rules": self.balancing_rules,
            "contextual_alternatives": self.contextual_alternatives,
            "emotion_categories": self.emotion_categories,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)

        logger.info(f"감정 표현 규칙이 저장되었습니다: {save_path}")

    def load_rules(self, file_path: Optional[str] = None):
        """JSON 파일에서 규칙 로드"""
        load_path = Path(file_path) if file_path else self.rules_file
        if not load_path or not load_path.exists():
            logger.warning(f"규칙 파일을 찾을 수 없습니다: {load_path}")
            return

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                rules_data = json.load(f)

            self.softening_rules.update(rules_data.get("softening_rules", {}))
            self.intensifying_rules.update(rules_data.get("intensifying_rules", {}))
            self.balancing_rules.update(rules_data.get("balancing_rules", {}))
            self.contextual_alternatives.update(
                rules_data.get("contextual_alternatives", {})
            )
            self.emotion_categories.update(rules_data.get("emotion_categories", {}))

            logger.info(f"감정 표현 규칙이 로드되었습니다: {load_path}")

        except Exception as e:
            logger.error(f"규칙 로드 실패: {e}")

    def add_custom_rule(self, rule_type: str, original: str, replacement: str):
        """사용자 정의 규칙 추가"""

        if rule_type == "softening":
            self.softening_rules[original] = replacement
        elif rule_type == "intensifying":
            self.intensifying_rules[original] = replacement
        elif rule_type == "balancing":
            self.balancing_rules[original] = replacement
        else:
            logger.warning(f"알 수 없는 규칙 유형: {rule_type}")
            return

        logger.info(
            f"사용자 정의 {rule_type} 규칙이 추가되었습니다: {original} -> {replacement}"
        )

    def get_rule_statistics(self) -> Dict[str, Any]:
        """규칙 통계 반환"""
        return {
            "total_softening_rules": len(self.softening_rules),
            "total_intensifying_rules": len(self.intensifying_rules),
            "total_balancing_rules": len(self.balancing_rules),
            "emotion_categories": len(self.emotion_categories),
            "contextual_alternatives": len(self.contextual_alternatives),
            "validation_issues": len(self.validate_rules()),
        }
