"""
Simple LLM Interface using OpenAI API directly.
No complex frameworks or dependencies.
"""

import logging
from typing import Optional
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

class SimpleLLMInterface:
    """Simple interface to OpenAI's API."""
    
    def __init__(self, api_key: Optional[str], model: str = "gpt-3.5-turbo"):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized LLM interface with model: {model}")
    
    def generate_response(self, user_message: str, context: str = "") -> str:
        """Generate a response to the user message with optional context."""
        try:
            # Prepare the prompt
            if context:
                system_message = f"""You are a helpful AI assistant for academic support. 
                Use the following context from the user's documents to help answer their question:

                CONTEXT:
                {context}

                Please provide helpful, accurate responses based on this context and your knowledge.
                If the context doesn't contain relevant information, say so and provide general help."""
            else:
                system_message = """You are a helpful AI assistant for academic support. 
                Help students with their questions about studying, assignments, and academic topics."""
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def generate_study_plan(self, subject: str, topics: list, deadline: str) -> str:
        """Generate a study plan."""
        topics_text = ", ".join(topics) if topics else "general topics"
        
        prompt = f"""Create a study plan for the subject: {subject}
        Topics to cover: {topics_text}
        Deadline: {deadline}
        
        Please provide:
        1. A weekly breakdown
        2. Specific study goals for each week
        3. Recommended study methods
        4. Time allocation suggestions
        """
        
        return self.generate_response(prompt)
    
    def help_with_assignment(self, assignment_description: str, requirements: str = "") -> str:
        """Help with assignment planning."""
        prompt = f"""Help me with this assignment:
        
        Assignment: {assignment_description}
        
        Requirements: {requirements if requirements else "No specific requirements provided"}
        
        Please provide:
        1. An outline or structure
        2. Key points to address
        3. Suggested approach
        4. Resources or research directions
        """
        
        return self.generate_response(prompt)
    
    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """Summarize a text."""
        if len(text) <= max_length:
            return text
        
        prompt = f"""Please provide a concise summary of the following text (aim for about {max_length} characters):

        {text[:3000]}  # Limit input to avoid token limits
        """
        
        return self.generate_response(prompt)
    
    def explain_concept(self, concept: str, context: str = "") -> str:
        """Explain a concept."""
        prompt = f"Please explain the concept: {concept}"
        if context:
            prompt += f"\n\nContext from user's notes:\n{context}"
        
        return self.generate_response(prompt)
