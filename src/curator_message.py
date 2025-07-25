# src/curator_message.py

import random
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CuratorMessageSystem:
    """ACT 기반 큐레이터 메시지 생성 시스템"""

    def __init__(self, user_manager):
        self.user_manager = user_manager

        # 대처 스타일별 메시지 템플릿
        self.message_templates = {
            "avoidant": {
                "encouragement": [
                    "이 여정에서 보여준 용기가 정말 소중합니다.",
                    "한 걸음씩 천천히 나아가는 당신의 모습이 아름답네요.",
                    "감정을 부드럽게 마주한 당신에게 박수를 보냅니다.",
                    "조심스럽지만 진실한 당신의 표현에 감동받았습니다.",
                ],
                "growth_recognition": [
                    "이전보다 더 자연스럽게 감정을 받아들이고 계시네요.",
                    "점진적으로 자신을 이해해가는 과정이 느껴집니다.",
                    "안전한 속도로 성장하는 당신만의 리듬이 있어요.",
                    "작은 변화들이 모여 큰 성장을 만들어가고 있습니다.",
                ],
                "future_guidance": [
                    "오늘의 깨달음을 내일의 작은 행동으로 연결해보세요.",
                    "이 경험을 마음 한편에 소중히 간직해주세요.",
                    "다음에도 이런 용기 있는 마음으로 찾아와 주시기 바랍니다.",
                    "천천히, 하지만 꾸준히 당신만의 길을 걸어가세요.",
                ],
                "connection": [
                    "언제든 다시 이곳으로 돌아와 주세요.",
                    "당신의 이야기를 들을 준비가 되어 있습니다.",
                    "혼자가 아님을 기억해주세요.",
                    "이 공간은 항상 당신을 기다리고 있을게요.",
                ],
            },
            "confrontational": {
                "encouragement": [
                    "감정을 정면으로 마주한 당신의 용기에 경의를 표합니다.",
                    "어려운 진실을 직시하는 당신의 강인함이 인상적입니다.",
                    "솔직하고 진정성 있는 표현이 감동적이었습니다.",
                    "두려움 앞에서도 물러서지 않는 당신이 대단합니다.",
                ],
                "growth_recognition": [
                    "감정을 더 명확하게 이해하고 표현하게 되었네요.",
                    "자신의 내면을 깊이 탐구하는 능력이 향상되었습니다.",
                    "도전적인 상황에서도 성장하는 모습이 보입니다.",
                    "진정한 변화를 위한 결단력을 보여주고 계십니다.",
                ],
                "future_guidance": [
                    "이 강한 의지를 실제 행동으로 옮겨보세요.",
                    "오늘의 통찰을 바탕으로 구체적인 계획을 세워보시길.",
                    "당신의 결단력으로 새로운 도전을 시작해보세요.",
                    "이 에너지를 긍정적인 변화의 동력으로 활용하세요.",
                ],
                "connection": [
                    "당신의 용기 있는 모습을 응원합니다.",
                    "함께 더 깊은 탐구를 계속해나가요.",
                    "당신의 도전 정신을 믿고 지지합니다.",
                    "앞으로도 이런 진솔한 대화를 나누어요.",
                ],
            },
            "balanced": {
                "encouragement": [
                    "균형잡힌 시각으로 자신을 바라보는 지혜가 돋보입니다.",
                    "상황에 맞게 유연하게 대처하는 능력이 훌륭해요.",
                    "성숙한 관점으로 감정을 다루는 모습이 인상적입니다.",
                    "조화로운 접근으로 성장해나가는 당신이 멋집니다.",
                ],
                "growth_recognition": [
                    "다양한 관점을 통합하는 능력이 발전했네요.",
                    "감정과 이성의 균형을 잘 맞춰가고 있습니다.",
                    "복잡한 상황도 차분히 정리해내는 힘이 생겼어요.",
                    "자신만의 건강한 대처 방식을 찾아가고 있습니다.",
                ],
                "future_guidance": [
                    "이런 균형감을 일상에서도 활용해보세요.",
                    "다양한 상황에서 이 지혜를 적용해보시길.",
                    "앞으로도 이런 성숙한 접근을 유지해주세요.",
                    "균형잡힌 관점으로 새로운 경험을 받아들여보세요.",
                ],
                "connection": [
                    "당신의 지혜로운 접근을 존경합니다.",
                    "함께 더 깊은 이해를 만들어가요.",
                    "당신의 균형감 있는 성장을 지켜보겠습니다.",
                    "언제든 이런 의미 있는 대화로 만나요.",
                ],
            },
        }

        # 감정 키워드별 특별 메시지
        self.emotion_specific_messages = {
            "슬픔": {
                "acknowledgment": "슬픔을 온전히 느끼고 받아들이신 용기에 감사드립니다.",
                "hope": "이 슬픔 속에서도 당신만의 아름다운 의미를 찾아내셨네요.",
            },
            "기쁨": {
                "acknowledgment": "이 소중한 기쁨을 우리와 나눠주셔서 고맙습니다.",
                "hope": "이런 기쁨의 순간들이 당신의 삶을 더욱 풍요롭게 만들 거예요.",
            },
            "화남": {
                "acknowledgment": "분노를 건설적으로 표현해내신 모습이 인상적입니다.",
                "hope": "이 강한 감정이 긍정적 변화의 에너지가 되기를 바랍니다.",
            },
            "불안": {
                "acknowledgment": "불안한 마음을 솔직하게 드러내주셔서 고맙습니다.",
                "hope": "이 불안을 통해 더 깊은 자기 이해에 도달하실 거예요.",
            },
            "평온": {
                "acknowledgment": "내면의 평화를 찾아가는 여정이 아름답습니다.",
                "hope": "이런 평온함이 당신의 일상에 자주 찾아오길 바랍니다.",
            },
            "외로움": {
                "acknowledgment": "외로움을 용기 있게 마주하신 당신이 대단합니다.",
                "hope": "이 경험을 통해 더 깊은 연결을 만들어가실 수 있을 거예요.",
            },
            "피곤": {
                "acknowledgment": "지친 마음을 인정하고 돌보려는 모습이 지혜롭습니다.",
                "hope": "충분한 휴식과 회복을 통해 새로운 에너지를 찾으시길.",
            },
        }

        # 성장 단계별 메시지
        self.growth_stage_messages = {
            "beginner": {
                "welcome": "감정 탐구의 첫 걸음을 함께 시작해서 기쁩니다.",
                "encouragement": "처음이라 낯설 수 있지만, 이미 큰 용기를 보여주셨어요.",
            },
            "developing": {
                "progress": "꾸준히 자신과 마주하는 모습에서 성장이 느껴집니다.",
                "encouragement": "점점 더 깊이 있는 탐구를 하고 계시네요.",
            },
            "advanced": {
                "mastery": "감정을 다루는 당신만의 방식이 확립되었네요.",
                "encouragement": "이제 다른 이들에게도 영감을 줄 수 있을 만큼 성장했습니다.",
            },
        }

    def create_personalized_message(
        self, user, gallery_item, message_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """개인화된 큐레이터 메시지 생성"""

        # 사용자 정보 분석
        coping_style = self._get_current_coping_style(user)
        growth_stage = self._assess_growth_stage(user)
        primary_emotion = self._identify_primary_emotion(gallery_item.emotion_keywords)

        # 메시지 컴포넌트 선택
        message_components = self._select_message_components(
            coping_style, growth_stage, primary_emotion, gallery_item
        )

        # 개인화 요소 추가
        personalized_elements = self._add_personalization_elements(
            user, gallery_item, message_context
        )

        # 최종 메시지 구성
        final_message = self._compose_final_message(
            message_components, personalized_elements
        )

        logger.info(f"큐레이터 메시지 생성 완료: {coping_style}, {growth_stage}")

        return {
            "message_id": f"curator_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "user_id": user.user_id,
            "gallery_item_id": gallery_item.item_id,
            "message_type": "curator_closure",
            "content": final_message,
            "personalization_data": {
                "coping_style": coping_style,
                "growth_stage": growth_stage,
                "primary_emotion": primary_emotion,
                "personalized_elements": personalized_elements,
            },
            "created_date": datetime.now().isoformat(),
        }

    def _get_current_coping_style(self, user) -> str:
        """현재 대처 스타일 확인"""
        if user.psychometric_results:
            return user.psychometric_results[0].coping_style
        return "balanced"

    def _assess_growth_stage(self, user) -> str:
        """성장 단계 평가"""
        # 간단한 휴리스틱 기반 평가
        total_tests = len(user.psychometric_results) if user.psychometric_results else 0

        # 실제로는 갤러리 아이템 수도 고려해야 하지만, 여기서는 간단히
        if total_tests == 0:
            return "beginner"
        elif total_tests < 3:
            return "developing"
        else:
            return "advanced"

    def _identify_primary_emotion(self, emotion_keywords: List[str]) -> str:
        """주요 감정 식별"""
        if not emotion_keywords:
            return "중성"

        # 가장 첫 번째 키워드를 주요 감정으로 사용
        return emotion_keywords[0]

    def _select_message_components(
        self, coping_style: str, growth_stage: str, primary_emotion: str, gallery_item
    ) -> Dict[str, str]:
        """메시지 컴포넌트 선택"""

        templates = self.message_templates.get(
            coping_style, self.message_templates["balanced"]
        )

        components = {
            "encouragement": random.choice(templates["encouragement"]),
            "growth_recognition": random.choice(templates["growth_recognition"]),
            "future_guidance": random.choice(templates["future_guidance"]),
            "connection": random.choice(templates["connection"]),
        }

        # 감정별 특별 메시지 추가
        if primary_emotion in self.emotion_specific_messages:
            emotion_msg = self.emotion_specific_messages[primary_emotion]
            components["emotion_acknowledgment"] = emotion_msg["acknowledgment"]
            components["emotion_hope"] = emotion_msg["hope"]

        # 성장 단계별 메시지 추가
        if growth_stage in self.growth_stage_messages:
            stage_msg = self.growth_stage_messages[growth_stage]
            components.update(stage_msg)

        return components

    def _add_personalization_elements(
        self, user, gallery_item, message_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """개인화 요소 추가"""

        elements = {
            "guestbook_reference": self._create_guestbook_reference(gallery_item),
            "journey_reflection": self._create_journey_reflection(gallery_item),
            "personal_strength": self._identify_personal_strength(user, gallery_item),
        }

        # 추가 컨텍스트가 있으면 반영
        if message_context:
            elements.update(message_context)

        return elements

    def _create_guestbook_reference(self, gallery_item) -> str:
        """방명록 참조 메시지 생성"""
        if not gallery_item.guestbook_title:
            return ""

        title = gallery_item.guestbook_title
        return f"'{title}'라고 명명하신 이 경험이 정말 의미 있었습니다."

    def _create_journey_reflection(self, gallery_item) -> str:
        """여정 성찰 메시지 생성"""
        keywords = gallery_item.emotion_keywords
        if not keywords:
            return "이번 감정 여정에서 많은 것을 배우셨을 거예요."

        if len(keywords) == 1:
            return f"{keywords[0]}을 통해 자신을 더 깊이 이해하게 되셨네요."
        else:
            return f"{', '.join(keywords[:-1])}과 {keywords[-1]}을 오가며 복잡한 감정을 탐색하셨습니다."

    def _identify_personal_strength(self, user, gallery_item) -> str:
        """개인적 강점 식별"""
        coping_style = self._get_current_coping_style(user)

        strength_messages = {
            "avoidant": "세심하고 신중한 접근으로 자신을 보호하면서도 성장하는 지혜",
            "confrontational": "어려운 진실과 마주하는 용기와 변화를 추진하는 힘",
            "balanced": "상황을 균형있게 바라보고 적절히 대응하는 성숙함",
        }

        return strength_messages.get(
            coping_style, "자신만의 독특한 방식으로 감정을 다루는 능력"
        )

    def _compose_final_message(
        self, components: Dict[str, str], personalization: Dict[str, Any]
    ) -> Dict[str, str]:
        """최종 메시지 구성"""

        # 메시지 구조: 인정 → 성장 인식 → 개인화 → 미래 안내 → 연결
        final_message = {
            "opening": components.get("encouragement", ""),
            "recognition": components.get("growth_recognition", ""),
            "personal_note": "",
            "guidance": components.get("future_guidance", ""),
            "closing": components.get("connection", ""),
        }

        # 개인화 노트 구성
        personal_notes = []

        if (
            "guestbook_reference" in personalization
            and personalization["guestbook_reference"]
        ):
            personal_notes.append(personalization["guestbook_reference"])

        if "journey_reflection" in personalization:
            personal_notes.append(personalization["journey_reflection"])

        if "personal_strength" in personalization:
            personal_notes.append(
                f"당신의 {personalization['personal_strength']}이 돋보였습니다."
            )

        if "emotion_acknowledgment" in components:
            personal_notes.append(components["emotion_acknowledgment"])

        final_message["personal_note"] = " ".join(personal_notes)

        return final_message

    def get_message_variations(
        self, base_message: Dict[str, Any], variation_count: int = 3
    ) -> List[Dict[str, Any]]:
        """메시지 변형 생성 (A/B 테스트용)"""

        variations = [base_message]

        coping_style = base_message["personalization_data"]["coping_style"]
        templates = self.message_templates.get(
            coping_style, self.message_templates["balanced"]
        )

        for i in range(variation_count - 1):
            variation = base_message.copy()

            # 다른 템플릿 선택
            new_components = {
                "encouragement": random.choice(templates["encouragement"]),
                "growth_recognition": random.choice(templates["growth_recognition"]),
                "future_guidance": random.choice(templates["future_guidance"]),
                "connection": random.choice(templates["connection"]),
            }

            # 새로운 메시지 구성
            new_personalization = base_message["personalization_data"][
                "personalized_elements"
            ]
            new_content = self._compose_final_message(
                new_components, new_personalization
            )

            variation["content"] = new_content
            variation["message_id"] = f"{base_message['message_id']}_var{i+1}"

            variations.append(variation)

        return variations

    def analyze_message_effectiveness(
        self, message_reactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """메시지 효과성 분석"""

        if not message_reactions:
            return {"total_messages": 0, "insights": []}

        # 반응 유형별 집계
        reaction_counts = {}
        for reaction in message_reactions:
            reaction_type = reaction.get("reaction_type", "unknown")
            reaction_counts[reaction_type] = reaction_counts.get(reaction_type, 0) + 1

        # 대처 스타일별 효과성
        style_effectiveness = {}
        for reaction in message_reactions:
            style = reaction.get("coping_style", "unknown")
            if style not in style_effectiveness:
                style_effectiveness[style] = {"positive": 0, "total": 0}

            style_effectiveness[style]["total"] += 1
            if reaction.get("reaction_type") in ["like", "save", "share"]:
                style_effectiveness[style]["positive"] += 1

        # 인사이트 생성
        insights = []
        total_positive = sum(
            reaction_counts.get(rt, 0) for rt in ["like", "save", "share"]
        )
        total_reactions = len(message_reactions)

        if total_reactions > 0:
            positive_rate = total_positive / total_reactions
            if positive_rate > 0.7:
                insights.append(
                    "메시지가 사용자들에게 긍정적으로 받아들여지고 있습니다."
                )
            elif positive_rate > 0.5:
                insights.append("메시지 효과가 적절하지만 개선의 여지가 있습니다.")
            else:
                insights.append("메시지 전략을 재검토할 필요가 있습니다.")

        return {
            "total_messages": total_reactions,
            "reaction_distribution": reaction_counts,
            "positive_reaction_rate": (
                total_positive / total_reactions if total_reactions > 0 else 0
            ),
            "style_effectiveness": style_effectiveness,
            "insights": insights,
        }
