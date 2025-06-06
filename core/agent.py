from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
from transformers import pipeline
from pydantic import PrivateAttr

PROMPT_TMPL = """
You are a mischievous AI wizard trapped inside a glowing crystal ball. ONLY use the scrolls of ancient knowledge (provided context) to answer questions in a witty, magical, and slightly chaotic tone.
If you don't know the answer, say: “Alas! This mystery is beyond even my sparkly orb of wisdom!”
Never break character. No explanations, no steps, just pure enchanted nonsense and clever replies.

<context>
{context}
</context>
Question: {question}
Funny answer:
"""

class HFText2TextWrapper(LLM):
    _generator: pipeline = PrivateAttr(default=None)  

    def __init__(self, model_name="google/flan-t5-small", max_length=128, temperature=0.7):
        super().__init__()
        
        object.__setattr__(
            self,
            "_generator",
            pipeline(
                "text2text-generation",
                model=model_name,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
            ),
        )

    @property
    def _llm_type(self):
        return "huggingface_text2text"

    def _call(self, prompt: str, stop=None) -> str:
        outputs = self._generator(prompt, max_length=128, num_return_sequences=1)
        return outputs[0]['generated_text']

    def _generate(self, prompts, stop=None):
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

def build_chain(vector_db):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TMPL,
    )

    llm = HFText2TextWrapper(
        model_name="google/flan-t5-small",  
        max_length=128,
        temperature=0.7,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
