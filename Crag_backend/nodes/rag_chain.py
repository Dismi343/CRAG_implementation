### Generate
#from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from nodes.vector_store import get_retriever
from nodes.retrieval_evalv import retrival_evaluator_llm
from nodes.question_rewriter import question_rewriter
from nodes.web_search import tavily_web_search
load_dotenv()

OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")

def rag_generate(docs, question):
    # Prompt
    #rag_prompt = hub.pull("rlm/rag-prompt")
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {question} 
            Context: {context} 
            Answer:"""
    rag_prompt = ChatPromptTemplate.from_template(template)
    # LLM
    rag_llm = ChatOpenAI(
        model_name="openai/gpt-oss-120b:free", 
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0
    )


    # retriever = get_retriever()
    # docs = retriever.invoke(question)


    # Post-processing
    def format_docs(docs):
        return "\\n\\n".join(doc.page_content for doc in docs)
    # Chain
    rag_chain = rag_prompt | rag_llm | StrOutputParser()

    #Evaluation of the used documents "yes or no" if they are relevant to the question
    # eval_output=retrival_evaluator_llm(docs, question)

    # if(eval_output.binary_score == "no"):
    #     question = question_rewriter(question)
    #     docs = retriever.invoke(question)
    #     websearch = tavily_web_search(question)

    generation = rag_chain.invoke({"context": format_docs(docs), "question": question})

    print(rag_prompt.messages[0].prompt.template)
    print("====================================================================")
    print("\nQuestion: %s" % question)
    print("----")
    print("Documents:\\n")
    print('\\n\\n'.join(['- %s' % x.page_content for x in docs]))
    print("----")
    print("Final answer: %s" % generation)
    # print("====================================================================")
    # print("Evaluation of the used documents: %s" % eval_output.binary_score)
    # print("====================================================================")
    # print("Web search results: %s" % websearch)
    return generation