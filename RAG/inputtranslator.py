import os
import sys

sys.path.append(os.getcwd())

from src.infrastructure.llm.openai_llm import OpenAILLMProvider

def inputtranslator(question=None):
    llm_provider = OpenAILLMProvider()
    translation_dictionary = {
        "ryczalt": "ryczałt",
    }
    prompt = """
    Please use the following translation dictionary:
    {translation_dictionary}
    
    You are a legal assistant. Translate the following text to Polish as a lawyer would and make it sound like a lawyer would say it:
    {question}
    Please return only the translated text.
    """
    return llm_provider.get_llm().invoke(prompt.format(question=question, translation_dictionary=translation_dictionary)).content
