# test.py (fixed version)
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define your output parser structure using Pydantic
class ResponseModel(BaseModel):
    answer: str = Field(description="The detailed answer to the question")
    confidence: str = Field(description="Confidence level of the answer", enum=["High", "Medium", "Low"])

# Initialize components
def setup_chain():
    # Create parser FIRST
    parser = PydanticOutputParser(pydantic_object=ResponseModel)
    
    # Define prompt template WITH parser reference
    prompt_template = ChatPromptTemplate.from_template(
        "You're a helpful AI assistant. {format_instructions} Respond to: {question}"
    ).partial(format_instructions=parser.get_format_instructions())
    
    # Initialize Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )
    
    # Create chain
    chain = prompt_template | model | parser
    return chain

# Main execution
if __name__ == "__main__":
    try:
        chain = setup_chain()
        
        # Example question
        question = "Explain quantum computing in simple terms"
        
        # Get response (no parser reference needed here)
        response = chain.invoke({"question": question})
        
        # Print structured output
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        
    except Exception as e:
        print(f"Error: {str(e)}")