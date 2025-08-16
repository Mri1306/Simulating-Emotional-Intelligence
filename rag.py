import os
import json
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import BitsAndBytesConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document


hf_token = "your_hf_token"
model_id = "your_model_id"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
vector_db_path = "./vector2_db"
json_path = "test_dataset.json"


with open(json_path, "r") as f:
    data = json.load(f)

def parse_metadata(input_str):
    """Improved metadata parser with fallback values"""
    metadata = {
        "emotion": "neutral",
        "tone": "neutral",
        "response_time": "medium"
    }
    
    
    if "Emotion:" in input_str:
        metadata["emotion"] = input_str.split("Emotion:")[1].split("\n")[0].strip().lower()
    
    
    if "Tone:" in input_str:
        metadata["tone"] = input_str.split("Tone:")[1].split("\n")[0].strip().lower()
    
    
    if "Response Time:" in input_str:
        rt = input_str.split("Response Time:")[1].split("\n")[0].strip().lower()
        metadata["response_time"] = rt.split("(")[0].strip()
    
    return metadata


seen_content = set()
documents = []
for sample in data:
    meta = parse_metadata(sample["input"])
    content = sample["output"].strip()
    
    
    if content not in seen_content:
        documents.append(Document(
            page_content=content,
            metadata=meta
        ))
        seen_content.add(content)


embeddings = HuggingFaceEmbeddings(model_name=embedding_model)


if not os.path.exists(vector_db_path):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    docs = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(vector_db_path)
else:
    vectorstore = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

def filtered_retrieval(query, emotion, tone, response_time, k=5):
    """Enhanced retrieval with deduplication"""
    results = vectorstore.similarity_search(query, k=k*2)  
    unique_contents = set()
    filtered = []
    
    for doc in results:
        content_hash = hash(doc.page_content[:300])  
        if (content_hash not in unique_contents and
            doc.metadata["emotion"] == emotion.lower() and
            doc.metadata["tone"] == tone.lower()):
            filtered.append(doc)
            unique_contents.add(content_hash)
            if len(filtered) == k:
                break
                
    return filtered[:k] if filtered else results[:k]


tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    device_map="auto",
    quantization_config=quant_config
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    temperature=0.7,
    repetition_penalty=1.1,
    truncation=True,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)


template = """[INST]
<<SYS>>
Analyze this context and respond with {tone} tone considering {emotion}:
{context}
<</SYS>>

{question} [/INST]"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "emotion", "tone", "question"]
)

def clean_response(text):
    """Remove residual XML tags and trim output"""
    return text.replace("</s>", "").replace("<s>", "").strip()

def ask_rag(question, emotion, tone, response_time):
    """Final production-ready implementation"""
    docs = filtered_retrieval(question, emotion, tone, response_time)
    context = "\n".join({doc.page_content for doc in docs[:3]})  
    
    response = pipe(
        prompt.format(
            context=context,
            emotion=emotion,
            tone=tone,
            question=question
        ),
        max_new_tokens=2048,
        temperature=0.7,
        repetition_penalty=1.1
    )[0]['generated_text']
    
    return clean_response(response.split("[/INST]")[-1])


if __name__ == "__main__":
    print("Filtered RAG Ready!")
    result = ask_rag(
        question="How do you deal with uncertainty?",
        emotion="fear & anxiety",
        tone="calm",
        response_time="fast"
    )
    print("\nFinal Response:")
    print(result)




     
