# src/services/diary_exploration_service.py

import json
import logging
from typing import Dict, List, Any, Optional
import yaml
import os

logger = logging.getLogger(__name__)


class DiaryExplorationService:
    """일기 심화 탐색을 위한 질문 생성 서비스"""
    
    def __init__(self, gpt_service=None):
        # 외부에서 GPTService를 주입받거나 None
        self.gpt_service = gpt_service
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """gpt_prompts.yaml에서 일기 심화 탐색 프롬포트 로드"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'gpt_prompts.yaml')
            
            with open(config_path, 'r', encoding='utf-8') as f:
                all_prompts = yaml.safe_load(f)
                
            return all_prompts.get('diary_exploration', {})
        
        except Exception as e:
            logger.error(f"프롬포트 로드 실패: {e}")
            return {}
    
    def generate_exploration_questions(
        self, 
        diary_text: str, 
        emotion_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        일기 내용을 바탕으로 심화 탐색 질문 생성
        
        Args:
            diary_text: 사용자의 일기 내용
            emotion_keywords: 감정 분석 결과 키워드들
            
        Returns:
            생성된 질문들과 메타데이터를 포함한 딕셔너리
        """
        try:
            # 안전성 검사
            if self._contains_unsafe_content(diary_text):
                return self._create_safety_response()
            
            # 시스템 메시지
            system_message = self.prompts.get('system_message', '')
            
            # 사용자 메시지 템플릿 구성
            user_message_template = self.prompts.get('user_message_template', '')
            user_message = user_message_template.format(
                diary_text=diary_text,
                emotion_keywords=emotion_keywords
            )
            
            # GPT API 호출 (gpt_service가 None이면 오류 발생)
            if not self.gpt_service:
                raise ValueError("GPT 서비스가 초기화되지 않았습니다")
                
            response = self.gpt_service.get_completion(
                system_message=system_message,
                user_message=user_message,
                temperature=0.7,
                max_tokens=800
            )
            
            # JSON 응답 파싱
            if response and response.get('success'):
                logger.info(f"GPT 원본 응답: {response['content']}")  # 디버깅용 로그 추가
                result = self._parse_exploration_response(response['content'])
                result['success'] = True
                return result
            else:
                logger.error(f"GPT 응답 실패: {response}")
                return self._create_fallback_response(emotion_keywords)
                
        except Exception as e:
            logger.error(f"질문 생성 실패: {e}")
            return self._create_fallback_response(emotion_keywords)
    
    def generate_follow_up_question(
        self, 
        diary_text: str, 
        previous_question: str,
        user_response: str,
        emotion_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        이전 답변을 바탕으로 후속 질문 생성
        
        Args:
            diary_text: 원본 일기 내용
            previous_question: 이전 질문
            user_response: 사용자의 답변
            emotion_keywords: 감정 키워드들
            
        Returns:
            생성된 후속 질문과 메타데이터
        """
        try:
            # 안전성 검사
            if self._contains_unsafe_content(user_response):
                return self._create_safety_response()
            
            # 시스템 메시지 (기존과 동일)
            system_message = self.prompts.get('system_message', '')
            
            # 후속 질문용 사용자 메시지 템플릿 구성
            follow_up_template = self.prompts.get('follow_up_message_template', '')
            user_message = follow_up_template.format(
                diary_text=diary_text,
                previous_question=previous_question,
                user_response=user_response,
                emotion_keywords=emotion_keywords
            )
            
            # GPT API 호출
            if not self.gpt_service:
                raise ValueError("GPT 서비스가 초기화되지 않았습니다")
                
            response = self.gpt_service.get_completion(
                system_message=system_message,
                user_message=user_message,
                temperature=0.7,
                max_tokens=600
            )
            
            # JSON 응답 파싱
            if response and response.get('success'):
                logger.info(f"GPT 후속질문 원본 응답: {response['content']}")  # 디버깅용 로그 추가
                result = self._parse_exploration_response(response['content'])
                result['success'] = True
                result['is_follow_up'] = True
                return result
            else:
                logger.error(f"GPT 응답 실패: {response}")
                return self._create_follow_up_fallback_response(emotion_keywords)
                
        except Exception as e:
            logger.error(f"후속 질문 생성 실패: {e}")
            return self._create_follow_up_fallback_response(emotion_keywords)
    
    def _parse_exploration_response(self, response_text: str) -> Dict[str, Any]:
        """GPT 응답 텍스트를 파싱하여 구조화된 데이터로 변환"""
        try:
            logger.info(f"파싱 시도: {response_text[:200]}...")  # 첫 200자만 로그
            
            # JSON 파싱 시도
            result = json.loads(response_text)
            logger.info(f"JSON 파싱 성공: {result}")
            
            # 단일 질문 형식인지 확인
            if 'question' in result:
                logger.info("단일 질문 형식 감지")
                # 새로운 단일 질문 형식
                required_fields = ['question', 'exploration_theme', 'encouragement']
                if not all(key in result for key in required_fields):
                    logger.warning(f"필수 필드 누락: {[f for f in required_fields if f not in result]}")
                    raise ValueError("필수 필드가 누락됨")
                
                # 단일 질문을 리스트로 변환하여 기존 형식과 호환성 유지
                question_data = {
                    "question": result["question"],
                    "category": result.get("category", "general"),
                    "explanation": result.get("explanation", "")
                }
                result["questions"] = [question_data]
                logger.info(f"단일 질문을 리스트로 변환: {result['questions']}")
                
            else:
                logger.info("다중 질문 형식 감지")
                # 기존 다중 질문 형식 (fallback 지원)
                if not all(key in result for key in ['questions', 'exploration_theme', 'encouragement']):
                    raise ValueError("필수 필드가 누락됨")
                
                questions = result.get('questions', [])
                if len(questions) == 0:
                    raise ValueError("질문이 없음")
            
            logger.info(f"파싱 완료: {len(result.get('questions', []))}개 질문")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            # 텍스트에서 질문만 추출하는 fallback 로직
            return self._extract_questions_from_text(response_text)
        
        except Exception as e:
            logger.error(f"응답 파싱 실패: {e}")
            raise
    
    def _extract_questions_from_text(self, text: str) -> Dict[str, Any]:
        """JSON 파싱 실패 시 텍스트에서 질문 추출"""
        questions = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.endswith('?') or '질문' in line:
                if len(line) > 10:  # 너무 짧은 질문 제외
                    questions.append({
                        "question": line,
                        "category": "general",
                        "explanation": "감정을 더 깊이 탐색하는 데 도움이 됩니다."
                    })
        
        # 단일 질문 시스템: 최대 1개 질문만 유지
        questions = questions[:1]
        if len(questions) == 0:
            questions.extend(self._get_default_questions()[:1])
        
        return {
            "questions": questions,
            "exploration_theme": "Deep Emotional Exploration",
            "encouragement": "Take your time exploring your emotions. There are no right or wrong answers."
        }
    
    def _contains_unsafe_content(self, diary_text: str) -> bool:
        """일기 내용에 안전하지 않은 키워드가 포함되어 있는지 확인"""
        # gpt_prompts.yaml의 unsafe_keywords 사용
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'gpt_prompts.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                all_prompts = yaml.safe_load(f)
            
            unsafe_keywords = all_prompts.get('unsafe_keywords', {})
            text_lower = diary_text.lower()
            
            for category, keywords in unsafe_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        logger.warning(f"안전하지 않은 키워드 감지: {keyword} in {category}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"안전성 검사 실패: {e}")
            return False  # 검사 실패 시 안전하다고 간주
    
    def _create_safety_response(self) -> Dict[str, Any]:
        """안전하지 않은 내용 감지 시 반환할 응답"""
        return {
            "success": False,
            "questions": [],
            "exploration_theme": "Safety First",
            "encouragement": "If you're going through a difficult time, please consider seeking help from a mental health professional.",
            "safety_warning": True,
            "professional_help_suggested": True
        }
    
    def _create_fallback_response(self, emotion_keywords: List[str]) -> Dict[str, Any]:
        """GPT 실패 시 대체 응답 생성"""
        default_questions = self._get_default_questions()
        
        # 단일 질문 시스템: 1개 질문만 생성
        customized_questions = []
        question = default_questions[0] if default_questions else {
            "question": "Can you describe in more detail the specific situation that led to this emotion?",
            "category": "emotion_cause",
            "explanation": "Helps clarify the cause of emotions by making the situation more specific."
        }
        
        if emotion_keywords:
            # 첫 번째 감정으로 질문 개인화
            primary_emotion = emotion_keywords[0]
            question_text = question["question"].replace("이 감정", f"{primary_emotion}")
            question["question"] = question_text
        customized_questions.append(question)
        
        return {
            "success": True,
            "questions": customized_questions,
            "exploration_theme": "Emotional Exploration",
            "encouragement": "Take some time to explore your emotions.",
            "fallback_used": True
        }
    
    def _create_follow_up_fallback_response(self, emotion_keywords: List[str]) -> Dict[str, Any]:
        """후속 질문 생성 실패 시 대체 응답"""
        follow_up_questions = self._get_follow_up_questions()
        
        # 랜덤하게 후속 질문 선택 (더 다양한 경험 제공)
        import random
        question_data = random.choice(follow_up_questions) if follow_up_questions else {
            "question": "What new thoughts or feelings are coming up for you as you reflect on this?",
            "category": "emotion_detail",
            "explanation": "Explores emerging emotions and thoughts from reflection."
        }
        
        return {
            "success": True,
            "questions": [question_data],
            "exploration_theme": "Continued Exploration",
            "encouragement": "Thank you for sharing. Let's continue exploring together.",
            "fallback_used": True,
            "is_follow_up": True
        }
    
    def _get_default_questions(self) -> List[Dict[str, str]]:
        """YAML에서 기본 질문 세트 동적 로드"""
        questions = []
        try:
            # YAML에서 질문 카테고리별로 예시 질문들 수집
            question_categories = self.prompts.get('question_categories', {})
            
            for category, category_data in question_categories.items():
                examples = category_data.get('examples', [])
                purpose = category_data.get('purpose', '')
                
                for example in examples:
                    questions.append({
                        "question": example,
                        "category": category,
                        "explanation": purpose
                    })
            
            # 질문이 없으면 fallback
            if not questions:
                return self._get_hardcoded_fallback_questions()
                
            return questions
            
        except Exception as e:
            logger.error(f"YAML에서 질문 로드 실패: {e}")
            return self._get_hardcoded_fallback_questions()
    
    def _get_hardcoded_fallback_questions(self) -> List[Dict[str, str]]:
        """YAML 로드 실패 시 하드코딩된 fallback 질문들"""
        return [
            {
                "question": "Can you describe in more detail the specific situation or moment that led to this emotion?",
                "category": "emotion_cause", 
                "explanation": "Helps clarify the cause of emotions by making the situation more specific."
            },
            {
                "question": "Where in your body do you feel this emotion most strongly?",
                "category": "sensation",
                "explanation": "Allows you to recognize emotions more concretely through physical sensations."
            },
            {
                "question": "If someone you love was in the same situation, what advice would you want to give them?",
                "category": "perspective",
                "explanation": "Provides a new perspective by looking from another person's point of view."
            }
        ]
    
    def _get_follow_up_questions(self) -> List[Dict[str, str]]:
        """YAML에서 후속 질문 세트 동적 로드"""
        try:
            # YAML에서 후속 질문들 로드
            follow_up_questions = self.prompts.get('follow_up_questions', [])
            
            if not follow_up_questions:
                return self._get_hardcoded_follow_up_questions()
            
            # YAML 형식을 내부 형식으로 변환
            questions = []
            for q_data in follow_up_questions:
                if isinstance(q_data, dict):
                    questions.append({
                        "question": q_data.get('question', ''),
                        "category": q_data.get('category', 'general'),
                        "explanation": q_data.get('explanation', '')
                    })
            
            return questions if questions else self._get_hardcoded_follow_up_questions()
            
        except Exception as e:
            logger.error(f"YAML에서 후속 질문 로드 실패: {e}")
            return self._get_hardcoded_follow_up_questions()
    
    def _get_hardcoded_follow_up_questions(self) -> List[Dict[str, str]]:
        """YAML 로드 실패 시 하드코딩된 fallback 후속 질문들"""
        return [
            {
                "question": "How has sharing that made you feel?",
                "category": "sensation",
                "explanation": "Acknowledges the user's sharing and explores current feelings."
            },
            {
                "question": "What emotions are coming up for you as you reflect on what you just shared?",
                "category": "emotion_detail",
                "explanation": "Helps identify new emotions that may have emerged from reflection."
            },
            {
                "question": "Is there anything else about this experience that feels important to explore?",
                "category": "perspective",
                "explanation": "Allows the user to guide the exploration in their preferred direction."
            }
        ]
    
    def get_question_categories_info(self) -> Dict[str, Any]:
        """질문 카테고리 정보 반환"""
        return self.prompts.get('question_categories', {})
    
    def get_safety_guidelines(self) -> List[str]:
        """안전 가이드라인 반환"""
        return self.prompts.get('safety_guidelines', [])


# 전역 서비스 인스턴스
_diary_exploration_service: Optional[DiaryExplorationService] = None


def get_diary_exploration_service(gpt_service=None) -> DiaryExplorationService:
    """전역 일기 심화 탐색 서비스 인스턴스 반환"""
    global _diary_exploration_service
    
    if _diary_exploration_service is None:
        _diary_exploration_service = DiaryExplorationService(gpt_service=gpt_service)
    
    return _diary_exploration_service