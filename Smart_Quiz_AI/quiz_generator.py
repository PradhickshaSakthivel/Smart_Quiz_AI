import json
import os
import re
import streamlit as st
from groq import Groq


class QuizGenerator:
    def __init__(self):
        self.api_key = (
            st.secrets.get("GROQ_API_KEY")
            or st.session_state.get("GROQ_API_KEY")
        )

        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None

        self.demo_mode = not bool(self.client)

    
    def generate_from_topic(self, topic: str, difficulty: str, num_questions: int, question_type: str = "Multiple Choice"):
        """Generate questions from a topic with specified type"""
        if self.demo_mode or not self.client:
            return self._create_demo_questions(topic, num_questions, question_type)
        
        # Create prompt based on question type
        if question_type == "Multiple Choice":
            prompt = self._create_mcq_prompt(topic, difficulty, num_questions)
        elif question_type == "True/False":
            prompt = self._create_tf_prompt(topic, difficulty, num_questions)
        elif question_type == "Short Answer":
            prompt = self._create_sa_prompt(topic, difficulty, num_questions)
        else:
            # Default to MCQ
            prompt = self._create_mcq_prompt(topic, difficulty, num_questions)
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            st.write(f"📝 LLM Response received for {question_type}")
            
            # Extract JSON from response
            json_match =( re.search(r'\[.*\]', response_text, re.DOTALL)
            if isinstance(response_text,str)
              
            else None
            )

            if json_match:
                json_str = json_match.group(0)
                # Clean JSON string
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                questions = json.loads(json_str)
                
                if isinstance(questions, list) and len(questions) > 0:
                    st.write(f"✅ Parsed {len(questions)} {question_type} questions from LLM")
                    return questions[:num_questions]
            
            # If JSON parsing fails, return demo questions
            st.warning("LLM didn't return valid JSON, using demo questions")
            return self._create_demo_questions(topic, num_questions, question_type)
            
        except Exception as e:
            st.error(f"❌ Error generating {question_type} questions: {str(e)}")
            return self._create_demo_questions(topic, num_questions, question_type)
    
    def generate_from_file_content(self, file_content: str, file_name: str, difficulty: str, num_questions: int, question_type: str = "Multiple Choice"):
        """Generate questions from file content with specified type"""
        if self.demo_mode or not self.client:
            return self._create_file_demo_questions(file_name, num_questions, question_type)
        
        # Limit content length
        content_preview = file_content[:2000] if len(file_content) > 2000 else file_content
        
        # Create prompt based on question type
        if question_type == "Multiple Choice":
            prompt = self._create_mcq_file_prompt(file_name, content_preview, difficulty, num_questions)
        elif question_type == "True/False":
            prompt = self._create_tf_file_prompt(file_name, content_preview, difficulty, num_questions)
        elif question_type == "Short Answer":
            prompt = self._create_sa_file_prompt(file_name, content_preview, difficulty, num_questions)
        else:
            prompt = self._create_mcq_file_prompt(file_name, content_preview, difficulty, num_questions)
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            st.write(f"📝 LLM Response received for file-based {question_type}")
            
            # Extract JSON from response
            json_match = (re.search(r'\[.*\]', response_text, re.DOTALL)
            if isinstance(response_text,str)
            else None
            )
            if json_match:
                json_str = json_match.group(0)
                # Clean JSON string
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                questions = json.loads(json_str)
                
                if isinstance(questions, list) and len(questions) > 0:
                    st.write(f"✅ Parsed {len(questions)} {question_type} questions from file")
                    return questions[:num_questions]
            
            # If JSON parsing fails, return demo questions
            st.warning("LLM didn't return valid JSON for file, using demo questions")
            return self._create_file_demo_questions(file_name, num_questions, question_type)
            
        except Exception as e:
            st.error(f"❌ Error generating file {question_type} questions: {str(e)}")
            return self._create_file_demo_questions(file_name, num_questions, question_type)
    
    def _create_mcq_prompt(self, topic, difficulty, num_questions):
        """Create prompt for Multiple Choice Questions"""
        return f"""
You are a quiz question generator. Generate {num_questions} {difficulty} level MULTIPLE CHOICE questions about "{topic}".

IMPORTANT RULES:
1. Return ONLY valid JSON array, no other text
2. Each question must have exactly 4 options
3. Correct answer must be "A", "B", "C", or "D"
4. Include clear explanation

JSON FORMAT:
[
  {{
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "A",
    "explanation": "Explanation here"
  }}
]

Generate {num_questions} multiple choice questions now:
"""
    
    def _create_tf_prompt(self, topic, difficulty, num_questions):
        """Create prompt for True/False Questions"""
        return f"""
You are a quiz question generator. Generate {num_questions} {difficulty} level TRUE/FALSE questions about "{topic}".

IMPORTANT RULES:
1. Return ONLY valid JSON array, no other text
2. Each question must have options: ["True", "False"]
3. Correct answer must be "True" or "False"
4. Statements should be factual claims about {topic}
5. Include clear explanation

JSON FORMAT:
[
  {{
    "question": "Statement about {topic}",
    "options": ["True", "False"],
    "correct_answer": "True",
    "explanation": "Explanation why this is true/false"
  }}
]

Generate {num_questions} true/false questions now. Make some true and some false:
"""
    
    def _create_sa_prompt(self, topic, difficulty, num_questions):
        """Create prompt for Short Answer Questions"""
        return f"""
You are a quiz question generator. Generate {num_questions} {difficulty} level SHORT ANSWER questions about "{topic}".

IMPORTANT RULES:
1. Return ONLY valid JSON array, no other text
2. Questions should require 1-2 sentence answers
3. For short answer, use empty options array: []
4. Provide expected correct answer
5. Include explanation/context

JSON FORMAT:
[
  {{
    "question": "Question requiring short written answer?",
    "options": [],
    "correct_answer": "Expected answer here (1-2 sentences)",
    "explanation": "Additional context or information"
  }}
]

Generate {num_questions} short answer questions now:
"""
    
    def _create_mcq_file_prompt(self, file_name, content, difficulty, num_questions):
        """Create MCQ prompt for file content"""
        return f"""
Based on this document content, generate {num_questions} {difficulty} level MULTIPLE CHOICE questions.

DOCUMENT: {file_name}
CONTENT:
{content}

IMPORTANT RULES:
1. Return ONLY valid JSON array, no other text
2. Questions must be based ONLY on the provided content
3. Each question must have exactly 4 options
4. Correct answer must be "A", "B", "C", or "D"
5. Include explanation referencing the document

JSON FORMAT:
[
  {{
    "question": "Question based on document",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "A",
    "explanation": "Explanation with document reference"
  }}
]

Generate {num_questions} multiple choice questions now:
"""
    
    def _create_tf_file_prompt(self, file_name, content, difficulty, num_questions):
        """Create True/False prompt for file content"""
        return f"""
Based on this document content, generate {num_questions} {difficulty} level TRUE/FALSE questions.

DOCUMENT: {file_name}
CONTENT:
{content}

IMPORTANT RULES:
1. Return ONLY valid JSON array, no other text
2. Statements must be based ONLY on the provided content
3. Each question must have options: ["True", "False"]
4. Correct answer must be "True" or "False"
5. Include explanation referencing the document

JSON FORMAT:
[
  {{
    "question": "Statement about document content",
    "options": ["True", "False"],
    "correct_answer": "True",
    "explanation": "Explanation with document reference"
  }}
]

Generate {num_questions} true/false questions now:
"""
    
    def _create_sa_file_prompt(self, file_name, content, difficulty, num_questions):
        """Create Short Answer prompt for file content"""
        return f"""
Based on this document content, generate {num_questions} {difficulty} level SHORT ANSWER questions.

DOCUMENT: {file_name}
CONTENT:
{content}

IMPORTANT RULES:
1. Return ONLY valid JSON array, no other text
2. Questions must require brief written answers (1-2 sentences)
3. Questions must be based ONLY on the provided content
4. For short answer, use empty options array: []
5. Provide expected correct answer
6. Include explanation referencing the document

JSON FORMAT:
[
  {{
    "question": "Question requiring short written answer",
    "options": [],
    "correct_answer": "Expected answer (1-2 sentences)",
    "explanation": "Additional context with document reference"
  }}
]

Generate {num_questions} short answer questions now:
"""
    
    def _create_demo_questions(self, topic, num_questions, question_type):
        """Create demo questions for topics"""
        questions = []
        
        for i in range(num_questions):
            if question_type == "Multiple Choice":
                questions.append({
                    "question": f"Demo MCQ {i+1}: What is an important aspect of {topic}?",
                    "options": [
                        f"Core concept A related to {topic}",
                        "Unrelated concept B",
                        "Unrelated concept C",
                        "Unrelated concept D"
                    ],
                    "correct_answer": "A",
                    "explanation": f"This is a demo multiple-choice question about {topic}. Option A represents a key concept."
                })
            
            elif question_type == "True/False":
                is_true = i % 2 == 0
                questions.append({
                    "question": f"Demo TF {i+1}: {topic} is an important field in modern technology.",
                    "options": ["True", "False"],
                    "correct_answer": "True" if is_true else "False",
                    "explanation": f"This is a demo true/false question. The statement is {'true' if is_true else 'false'} based on general knowledge."
                })
            
            elif question_type == "Short Answer":
                questions.append({
                    "question": f"Demo SA {i+1}: Briefly explain one application of {topic}.",
                    "options": [],
                    "correct_answer": f"One application of {topic} is in developing intelligent systems that can solve complex problems.",
                    "explanation": f"This is a demo short answer question about {topic}. Applications vary based on the specific field."
                })
        
        return questions
    
    def _create_file_demo_questions(self, file_name, num_questions, question_type):
        """Create demo questions for files"""
        questions = []
        
        for i in range(num_questions):
            if question_type == "Multiple Choice":
                questions.append({
                    "question": f"Demo MCQ {i+1}: Based on '{file_name}', what is the document primarily about?",
                    "options": [
                        "Main topic of the document",
                        "Unrelated topic 1",
                        "Unrelated topic 2",
                        "Unrelated topic 3"
                    ],
                    "correct_answer": "A",
                    "explanation": f"Based on document analysis, '{file_name}' primarily discusses its main topic."
                })
            
            elif question_type == "True/False":
                is_true = i % 2 == 0
                questions.append({
                    "question": f"Demo TF {i+1}: The document '{file_name}' contains factual information.",
                    "options": ["True", "False"],
                    "correct_answer": "True" if is_true else "False",
                    "explanation": f"This statement is {'true' if is_true else 'false'} based on typical document analysis."
                })
            
            elif question_type == "Short Answer":
                questions.append({
                    "question": f"Demo SA {i+1}: What key information does '{file_name}' provide?",
                    "options": [],
                    "correct_answer": f"The document '{file_name}' provides key information about its subject matter, including important concepts and details.",
                    "explanation": f"Based on document structure, '{file_name}' contains informative content about its topic."
                })
        
        return questions


# Singleton instance
_generator_instance = None

def get_question_generator():
    """Get or create the question generator instance"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = QuizGenerator()
    return _generator_instance