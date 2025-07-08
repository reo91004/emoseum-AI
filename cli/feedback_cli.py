#!/usr/bin/env python3
"""
부담 없는 피드백 수집 CLI - 우울증 치료제 특화
"""

import argparse
import time
from datetime import datetime
from typing import Dict, List, Any

from core.therapy_system import EmotionalImageTherapySystem
from models.smart_feedback_system import SmartFeedbackCollector
from config import logger


class GentleFeedbackCLI:
    """부담 없는 피드백 수집 CLI"""
    
    def __init__(self):
        self.system = EmotionalImageTherapySystem()
        self.feedback_collector = SmartFeedbackCollector()
        
    def collect_gentle_feedback(self, user_id: str, emotion_id: int):
        """부담 없는 방식으로 피드백 수집"""
        
        print("\n" + "="*50)
        print("🌸 이미지에 대한 소감을 들려주세요")
        print("="*50)
        
        # 사용자 프로파일과 감정 기록 가져오기
        user_profile = self.system.get_user_profile(user_id)
        
        # 해당 감정 기록 찾기
        target_record = None
        for record in user_profile.emotion_history:
            if record.get("id") == emotion_id:
                target_record = record
                break
        
        if not target_record:
            print("❌ 해당 이미지를 찾을 수 없습니다.")
            return None
        
        # 피드백 인터페이스 생성
        interface = self.feedback_collector.create_gentle_feedback_interface(
            image_metadata=target_record.get("image_metadata", {}),
            user_emotion=target_record["emotion"],
            user_history=user_profile.emotion_history
        )
        
        # 격려 메시지 출력
        print(f"💬 {interface['encouragement']}")
        print()
        
        # 상호작용 시작 시간
        start_time = time.time()
        
        # 기본 점수 수집 (필수)
        feedback_data = {}
        
        while True:
            try:
                print(f"📊 {interface['primary_question']}")
                score_input = input("➤ 점수 입력 (1-5): ").strip()
                
                if score_input.lower() in ['q', 'quit', '종료']:
                    print("😊 괜찮습니다. 다음에 다시 시도해보세요.")
                    return None
                
                score = float(score_input)
                if 1 <= score <= 5:
                    feedback_data["score"] = score
                    break
                else:
                    print("⚠️ 1부터 5까지의 숫자를 입력해주세요.")
                    
            except ValueError:
                print("⚠️ 숫자를 입력해주세요. (종료하려면 'q' 입력)")
        
        print(f"✅ 점수 {score}점으로 기록됩니다.")
        print()
        
        # 추가 질문들 (선택사항)
        additional_responses = {}
        
        for i, question in enumerate(interface.get("additional_questions", [])):
            if self._should_ask_question(question, feedback_data, additional_responses):
                response = self._ask_optional_question(question, i+1)
                if response:
                    additional_responses[f"additional_{i}"] = response
        
        # 빠른 태그 선택 (있는 경우)
        selected_tags = self._collect_quick_tags(interface)
        if selected_tags:
            feedback_data["selected_tags"] = selected_tags
        
        # 상호작용 시간 계산
        interaction_time = time.time() - start_time
        
        # 메타데이터 구성
        interaction_metadata = {
            "viewing_time": max(interaction_time, 3.0),  # 최소 3초
            "response_time": interaction_time,
            "hesitation_count": 0,  # CLI에서는 측정하기 어려움
            "question_skip_count": sum(1 for v in additional_responses.values() if v == "skip")
        }
        
        # 피드백 데이터 통합
        all_responses = {**feedback_data, **additional_responses}
        
        # 시스템에 피드백 전송
        try:
            result = self.system.process_feedback(
                user_id=user_id,
                emotion_id=emotion_id,
                feedback_score=score,
                feedback_type="gentle_rating",
                comments=self._format_additional_feedback(additional_responses, selected_tags),
                interaction_metadata=interaction_metadata
            )
            
            if result["success"]:
                self._display_feedback_results(result)
                return result
            else:
                print(f"❌ 피드백 처리 실패: {result.get('error', '알 수 없는 오류')}")
                return None
                
        except Exception as e:
            print(f"❌ 시스템 오류: {e}")
            return None
    
    def _should_ask_question(
        self, 
        question: Dict[str, Any], 
        feedback_data: Dict[str, Any],
        previous_responses: Dict[str, Any]
    ) -> bool:
        """질문을 할지 결정 (사용자 부담 최소화)"""
        
        # 기본적으로 선택사항은 50% 확률로만
        if question.get("optional", True):
            # 이미 많은 질문에 답했으면 스킵
            if len(previous_responses) >= 2:
                return False
            
            # 낮은 점수면 부담 줄이기
            if feedback_data.get("score", 3) <= 2.5:
                return False
            
            # 일정 확률로만 질문
            import random
            return random.random() < 0.6
        
        return True
    
    def _ask_optional_question(self, question: Dict[str, Any], question_num: int) -> str:
        """선택적 질문 수행"""
        
        print(f"🤔 추가 질문 {question_num} (선택사항)")
        print(f"   {question['question']}")
        
        options = question.get("options", [])
        if options:
            for i, option in enumerate(options, 1):
                print(f"   {i}. {option}")
            
            print("   0. 건너뛰기")
            print()
            
            while True:
                try:
                    choice = input("➤ 선택 (번호 입력): ").strip()
                    
                    if choice == "0" or choice.lower() in ['skip', '건너뛰기', '']:
                        print("⏭️ 건너뛰었습니다.\n")
                        return "skip"
                    
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(options):
                        selected = options[choice_idx]
                        print(f"✅ '{selected}' 선택됨\n")
                        return selected
                    else:
                        print(f"⚠️ 1부터 {len(options)} 사이의 번호를 입력해주세요.")
                        
                except ValueError:
                    print("⚠️ 숫자를 입력해주세요.")
        else:
            # 자유 응답
            response = input("➤ 답변 (Enter로 건너뛰기): ").strip()
            if response:
                print(f"✅ 응답 기록됨\n")
                return response
            else:
                print("⏭️ 건너뛰었습니다.\n")
                return "skip"
    
    def _collect_quick_tags(self, interface: Dict[str, Any]) -> List[str]:
        """빠른 태그 수집"""
        
        # 태그 질문 찾기
        tag_question = None
        for question in interface.get("additional_questions", []):
            if question.get("type") == "quick_tags":
                tag_question = question
                break
        
        if not tag_question:
            return []
        
        print("🏷️ 마지막으로, 이미지에서 느끼는 것이 있다면 선택해주세요 (선택사항)")
        print(f"   {tag_question['question']}")
        
        options = tag_question.get("options", [])
        max_selections = tag_question.get("max_selections", 2)
        
        for i, option in enumerate(options, 1):
            print(f"   {i}. {option}")
        
        print("   0. 건너뛰기")
        print(f"\n💡 최대 {max_selections}개까지 선택 가능 (번호를 쉼표로 구분)")
        
        while True:
            try:
                choices = input("➤ 선택 (예: 1,3): ").strip()
                
                if choices == "0" or choices.lower() in ['skip', '건너뛰기', '']:
                    print("⏭️ 건너뛰었습니다.\n")
                    return []
                
                if not choices:
                    return []
                
                selected_tags = []
                for choice in choices.split(','):
                    choice = choice.strip()
                    choice_idx = int(choice) - 1
                    
                    if 0 <= choice_idx < len(options):
                        tag = options[choice_idx]
                        if tag not in selected_tags and tag != "없음":
                            selected_tags.append(tag)
                
                if len(selected_tags) > max_selections:
                    print(f"⚠️ 최대 {max_selections}개까지만 선택 가능합니다.")
                    continue
                
                if selected_tags:
                    print(f"✅ 선택된 태그: {', '.join(selected_tags)}\n")
                else:
                    print("⏭️ 선택된 태그가 없습니다.\n")
                
                return selected_tags
                
            except (ValueError, IndexError):
                print("⚠️ 올바른 번호를 입력해주세요. (예: 1,3)")
    
    def _format_additional_feedback(
        self, 
        additional_responses: Dict[str, Any],
        tags: List[str]
    ) -> str:
        """추가 피드백을 코멘트 형태로 포맷"""
        
        comments = []
        
        # 추가 응답들
        for key, value in additional_responses.items():
            if value and value != "skip":
                comments.append(f"[{key}] {value}")
        
        # 태그들
        if tags:
            comments.append(f"[tags] {', '.join(tags)}")
        
        return " | ".join(comments) if comments else None
    
    def _display_feedback_results(self, result: Dict[str, Any]):
        """피드백 결과 표시"""
        
        print("🎉 피드백이 성공적으로 기록되었습니다!")
        print()
        
        # 기본 정보
        print(f"📊 총 상호작용: {result.get('total_interactions', 0)}회")
        print(f"📝 총 피드백: {result.get('total_feedbacks', 0)}회")
        
        # 향상된 피드백 정보
        enhanced = result.get("enhanced_feedback")
        if enhanced:
            print("\n🧠 AI가 분석한 내용:")
            
            # 감정 일치도
            emotion_align = enhanced.get("emotion_alignment")
            if emotion_align:
                interpretation = emotion_align.get("interpretation", "")
                if interpretation:
                    print(f"   💭 {interpretation}")
            
            # 시각적 선호도
            visual_prefs = enhanced.get("visual_preferences", {})
            if visual_prefs:
                for pref_type, value in visual_prefs.items():
                    if "positive" in pref_type:
                        element = pref_type.replace("_positive", "")
                        print(f"   ✨ {element} 요소를 선호하시는 것 같아요")
            
            # 추론된 선호도
            inferred = enhanced.get("inferred_preferences", {})
            if inferred:
                therapeutic = inferred.get("therapeutic_preference")
                if therapeutic:
                    print(f"   🌱 치료적 효과: {therapeutic.replace('_', ' ')}")
        
        # 학습 결과
        if result.get("training_performed"):
            print("\n🤖 개인화 학습이 수행되었습니다.")
            training = result.get("training_result", {})
            mode = training.get("mode", "")
            if "adaptive" in mode or "lora" in mode:
                print(f"   📈 학습 모드: {mode}")
        
        # 치료 인사이트
        insights = result.get("therapeutic_insights", {})
        if insights and "emotional_state" in insights:
            mood = insights["emotional_state"].get("current_mood", "")
            if mood:
                print(f"\n😊 현재 기분: {mood}")
        
        print("\n💚 소중한 피드백 감사합니다!")


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="부담 없는 피드백 수집")
    parser.add_argument("--user-id", required=True, help="사용자 ID")
    parser.add_argument("--emotion-id", type=int, required=True, help="감정 기록 ID")
    
    args = parser.parse_args()
    
    cli = GentleFeedbackCLI()
    
    print("🌸 Emoseum 부담 없는 피드백 시스템")
    print("치료에 도움이 되도록 편안하게 의견을 들려주세요.")
    
    result = cli.collect_gentle_feedback(args.user_id, args.emotion_id)
    
    if result:
        print("\n✨ 피드백 수집이 완료되었습니다!")
    else:
        print("\n😊 언제든 다시 시도해보세요.")


if __name__ == "__main__":
    main()