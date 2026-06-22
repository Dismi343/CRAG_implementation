### Retrieval Evaluator
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")

# Data model
class RetrievalEvaluator(BaseModel):
    """Classify retrieved documents based on how relevant it is to the user's question."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
# LLM with function call
def retrival_evaluator_llm(docs, question):
    retrieval_evaluator_llm = ChatOpenAI(
        model_name="openai/gpt-oss-120b:free", #openai/gpt-oss-120b:free
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0
    )

    structured_llm_evaluator = retrieval_evaluator_llm.with_structured_output(
            RetrievalEvaluator, 
            method="json_mode"
        )
    # Prompt
    system = "You are an evaluator checking if a document is relevant to a question. Respond in JSON with key 'binary_score' as 'yes' or 'no'."


    
    retrieval_evaluator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \\n\\n {docs} \\n\\n User question: {question} \n\n JSON:"),
        ]
    )
    retrieval_grader = retrieval_evaluator_prompt | structured_llm_evaluator

    # Improve formatting: join docs into a single string
    docs_string = "\n\n".join([doc.page_content for doc in docs])
    try:
        results = retrieval_grader.invoke({"docs": docs_string, "question": question})
        print("results from the retriver: ",results)
        return results
    except Exception as e:
        print(f"Error in LLM evaluation: {e}. Defaulting to 'no'.")
        # Fallback to 'no' so the system triggers web search instead of crashing
        return RetrievalEvaluator(binary_score="no")