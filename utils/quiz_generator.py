# utils/quiz_generator.py

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
import random
import json

class QuizGenerator:
    """Generate quizzes from documents using LLM."""
    
    def __init__(self, llm=None, model_name="gpt-3.5-turbo", api_key=None):
        """Initialize the QuizGenerator with an LLM."""
        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                temperature=0.7
            )
        
        # Template for generating multiple choice questions
        self.mcq_template = PromptTemplate(
            input_variables=["text", "num_questions", "difficulty"],
            template="""
            Based on the following text, create {num_questions} multiple choice questions at {difficulty} difficulty level.
            For each question, provide exactly 4 options with only 1 correct answer.
            
            FORMAT YOUR RESPONSE AS A JSON ARRAY where each question is a dictionary with the keys:
            - "question": The question text
            - "options": Array of 4 possible answers as strings
            - "correct_answer": The index (0-3) of the correct answer
            - "explanation": Brief explanation of why the answer is correct
            
            TEXT:
            {text}
            
            QUESTIONS:
            """
        )
        
        # Template for generating open-ended questions
        self.open_template = PromptTemplate(
            input_variables=["text", "num_questions", "difficulty"],
            template="""
            Based on the following text, create {num_questions} open-ended questions at {difficulty} difficulty level.
            
            FORMAT YOUR RESPONSE AS A JSON ARRAY where each question is a dictionary with the keys:
            - "question": The question text
            - "answer": A comprehensive answer to the question
            - "keywords": Array of 3-5 keywords that should appear in a good answer
            
            TEXT:
            {text}
            
            QUESTIONS:
            """
        )
        
        # Template for generating true/false questions
        self.tf_template = PromptTemplate(
            input_variables=["text", "num_questions", "difficulty"],
            template="""
            Based on the following text, create {num_questions} true/false statements at {difficulty} difficulty level.
            Aim for a mix of true and false statements.
            
            FORMAT YOUR RESPONSE AS A JSON ARRAY where each question is a dictionary with the keys:
            - "statement": The statement to evaluate as true or false
            - "correct_answer": Boolean true or false
            - "explanation": Brief explanation of why the statement is true or false
            
            TEXT:
            {text}
            
            STATEMENTS:
            """
        )
    
    def _generate_questions(self, documents, question_type, num_questions=5, difficulty="medium"):
        """Generate questions of specified type from documents."""
        # Concatenate document text with source info
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Limit text length to avoid token limits
        max_chars = 6000
        if len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars]
        
        # Select appropriate template based on question type
        if question_type == "multiple_choice":
            template = self.mcq_template
        elif question_type == "open_ended":
            template = self.open_template
        elif question_type == "true_false":
            template = self.tf_template
        else:
            raise ValueError(f"Unknown question type: {question_type}")
            
        # Create and run the chain
        chain = LLMChain(llm=self.llm, prompt=template)
        result = chain.run(text=combined_text, num_questions=num_questions, difficulty=difficulty)
        
        try:
            # Parse JSON result
            questions = json.loads(result)
            return questions
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            return {"error": "Failed to generate valid questions. Please try again."}
    
    def generate_quiz(self, documents, quiz_config):
        """Generate a complete quiz based on the configuration."""
        question_types = quiz_config.get("question_types", ["multiple_choice"])
        num_questions = quiz_config.get("num_questions", 5)
        difficulty = quiz_config.get("difficulty", "medium")
        
        all_questions = []
        
        # Distribute questions among selected types
        questions_per_type = max(1, num_questions // len(question_types))
        remaining = num_questions - (questions_per_type * len(question_types))
        
        for q_type in question_types:
            # Calculate how many questions of this type
            q_count = questions_per_type
            if remaining > 0:
                q_count += 1
                remaining -= 1
                
            # Generate questions
            type_questions = self._generate_questions(
                documents,
                q_type,
                num_questions=q_count,
                difficulty=difficulty
            )
            
            # Add question type to each question
            for q in type_questions:
                q["type"] = q_type
            
            all_questions.extend(type_questions)
            
        # Shuffle questions
        random.shuffle(all_questions)
        
        return {
            "questions": all_questions,
            "topic": quiz_config.get("topic", "Knowledge Quiz"),
            "difficulty": difficulty,
            "total_questions": len(all_questions)
        }
    
    def grade_open_answer(self, question, user_answer):
        """Grade an open-ended question using the LLM."""
        template = PromptTemplate(
            input_variables=["question", "correct_answer", "user_answer", "keywords"],
            template="""
            Evaluate the user's answer to the following question:
            
            QUESTION: {question}
            
            REFERENCE ANSWER: {correct_answer}
            
            IMPORTANT KEYWORDS: {keywords}
            
            USER'S ANSWER: {user_answer}
            
            Score the answer on a scale of 0-100, where:
            - 0-20: Completely incorrect or unrelated
            - 21-40: Has some relevant information but misses key concepts
            - 41-60: Partially correct but incomplete
            - 61-80: Mostly correct with minor inaccuracies
            - 81-100: Fully correct and comprehensive
            
            FORMAT YOUR RESPONSE AS A JSON with the keys:
            - "score": Numerical score (0-100)
            - "feedback": Brief feedback explaining the score
            - "missing_keywords": List of keywords that were missing from the answer
            
            EVALUATION:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=template)
        result = chain.run(
            question=question["question"],
            correct_answer=question["answer"],
            user_answer=user_answer,
            keywords=question["keywords"]
        )
        
        try:
            # Parse JSON result
            evaluation = json.loads(result)
            return evaluation
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            return {
                "score": 0,
                "feedback": "Failed to evaluate answer. Please try again.",
                "missing_keywords": []
            }
