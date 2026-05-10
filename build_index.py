"""
build_index.py — Pré-computa o índice FAISS (execute uma única vez)
===================================================================
    python build_index.py
"""
import json
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
FAQ_FILE  = BASE_DIR / "data" / "corpus_faqs.json"
FAISS_DIR = BASE_DIR / "faiss_index"


def main():
    logger.info("Lendo corpus...")
    with open(FAQ_FILE, encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data["corpus"]:
        texto = (
            f"Pergunta: {item['pergunta']}\n"
            f"Resposta: {item['resposta']}\n"
            f"Palavras-chave: {', '.join(item['palavras_chave'])}"
        )
        docs.append(Document(
            page_content=texto,
            metadata={"id": item["id"], "cluster": item["cluster"]},
        ))

    logger.info(f"{len(docs)} documentos. Aplicando chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(docs)
    logger.info(f"{len(chunks)} chunks gerados.")

    logger.info("Carregando modelo de embeddings (pode demorar na 1ª vez)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

    logger.info("Construindo e salvando índice FAISS...")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(FAISS_DIR))
    logger.info(f"✅ Índice FAISS salvo em: {FAISS_DIR}")


if __name__ == "__main__":
    main()
