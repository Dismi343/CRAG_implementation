from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from nodes.vector_store import get_retriever
from nodes.retrieval_evalv import retrival_evaluator_llm
load_dotenv()


OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")

### Question Re-writer
def question_rewriter(question):
    # LLM
    question_rewriter_llm = ChatOpenAI(
            model_name="openai/gpt-oss-120b:free", 
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            temperature=0
    )
    # Prompt
    system = """You are a question re-writer that converts an input question to a better version that is optimized \\n 
        for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \\n\\n {question} \\n Formulate an improved question.",
            ),
        ]
    )
    question_rewriter = re_write_prompt | question_rewriter_llm | StrOutputParser()
    return question_rewriter.invoke({"question": question})