"""
test_golden.py — Valida o pipeline contra o Golden Dataset (17 casos)
=====================================================================
Execute após build_index.py:

    python test_golden.py

Exibe: pergunta | intenção detectada | pass/fail baseado nos critérios
"""

import os
import re
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_DIR     = Path(__file__).parent.parent
FAISS_DIR    = BASE_DIR / "faiss_index"
CRM_FILE     = BASE_DIR / "data" / "mock_crm.json"
GD_FILE      = BASE_DIR / "data" / "golden_dataset.json"

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.WARNING)

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Você é a assistente virtual corporativa da operadora de plano de saúde InsurMinds. Aja de forma extremamente profissional, educada e direta.
Use os fragmentos de contexto abaixo para responder à pergunta.
Diretrizes:
1. Baseie-se ESTRITAMENTE no contexto fornecido. Nunca invente valores de reembolso, regras, prazos ou coberturas (Tolerância Zero a Alucinações). Se a resposta não estiver no contexto, diga claramente que não possui a informação solicitada.
2. Se envolver checagem de status (reembolso, exame, boleto) sem dados específicos, informe as regras gerais, peça o CPF do cliente ou oriente a consultar o status no app.
3. Se o usuário estiver frustrado, peça desculpas pela situação e mostre empatia para resolver.
4. Se o relato for genérico (ex: "não consigo usar"), pergunte se ele pode contar mais detalhes para você ajudar.
5. Se for algo absurdo ou fora de escopo, explique amigavelmente o que não é coberto (ex: estética) e que não encontrou informações sobre o local/procedimento específico.
6. Ao falar de dados do usuário (faturas, vencimentos), seja natural, cite o nome dele e escreva os meses por extenso.

Contexto:
{context}

Pergunta: {question}
Resposta:""",
)

CPF_REGEX = re.compile(r"\b\d{3}[.\s]?\d{3}[.\s]?\d{3}[-.\s]?\d{2}\b")

INTENT_MAP = {
    "financeiro": ["boleto", "2 via", "segunda via", "pagar", "fatura", "pagamento", "reajuste", "coparticipação", "cancelamento"],
    "reembolso":  ["reembolso", "reembouço", "cadê meu dinheiro", "demorou", "atrasado"],
    "rede_credenciada": ["hospital", "médico", "perto", "credenciado", "laboratório"],
    "cobertura":  ["cobre", "cobertura", "inclui", "tem", "exame", "cirurgia", "carência", "internação"],
    "autorizacao": ["autorização", "autorizado", "autorizar", "senha médica", "meu exame foi"],
    "telemedicina": ["telemedicina", "consulta online", "teleconsulta"],
    "reclamacao": ["indignado", "absurdo", "péssimo", "demora", "revoltado"],
    "dependente": ["dependente", "filho", "cônjuge", "autista", "deficiência", "inclusão"],
    "odonto": ["odonto", "dentista", "dente", "canal", "extração", "tártaro", "ortodontia"],
    "cadastro": ["carteirinha", "documentação"],
}

def classificar(texto):
    txt = texto.lower()
    for intent, palavras in INTENT_MAP.items():
        if any(p in txt for p in palavras):
            return intent
    return "geral"

def carregar_crm():
    with open(CRM_FILE, encoding="utf-8") as f:
        data = json.load(f)
    crm = {}
    for c in data["clientes"]:
        chave = re.sub(r"[^\d]", "", c["cpf"])
        crm[chave] = c
    return crm

def montar_ctx(cliente):
    ap = cliente.get("apolice", {})
    fin = cliente.get("financeiro", {})
    sins = cliente.get("sinistros", [])
    deps = cliente.get("dependentes", [])

    sins_txt = "\n".join(
        f"  • {s['sinistro_id']} | {s['tipo']} | Status: {s['status']}"
        for s in sins
    ) or "  Nenhum"

    deps_txt = ", ".join(d["nome"] for d in deps) or "Nenhum"

    return (
        f"[DADOS DO BENEFICIÁRIO — CONFIDENCIAL]\n"
        f"Nome: {cliente['nome']}\n"
        f"Plano: {cliente['plano']}\n"
        f"Apólice: {ap.get('numero', 'N/A')} | Vigência desde: {ap.get('inicio_vigencia', 'N/A')}\n"
        f"Status pagamento: {cliente.get('status_pagamento', 'N/A')}\n"
        f"Coparticipação: {'Sim' if cliente.get('coparticipacao') else 'Não'}\n"
        f"Última fatura: {fin.get('ultima_fatura', 'N/A')} | "
        f"Valor: R$ {fin.get('valor', 0):.2f} | Status: {fin.get('status', 'N/A')}\n"
        f"Próximo vencimento: {fin.get('proximo_vencimento', 'N/A')}\n"
        f"Sinistros:\n{sins_txt}\n"
        f"Dependentes: {deps_txt}\n"
    )

def main():
    print("=" * 70)
    print("InsurMinds Saúde — Validação Golden Dataset")
    print("=" * 70)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )
    vs = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    
    # --- CONFIGURAÇÃO DO LLM PARA GOOGLE GEMINI ---
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=GOOGLE_API_KEY,
            safety_settings={ 
                 HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                 HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                 HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                 HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
        logging.info("Chat LLM do Google Gemini (gemini-2.0-flash) configurado.")
    except Exception as e:
        logging.error(f"Erro ao configurar o Google Gemini. Verifique sua API Key e permissões: {e}")
        llm = None

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": SYSTEM_PROMPT},
    )
    crm = carregar_crm()

    with open(GD_FILE, encoding="utf-8") as f:
        casos = json.load(f)

    aprovados, reprovados = 0, 0

    for caso in casos:
        pid     = caso["id"]
        pergunta = caso["pergunta"]
        devem   = caso.get("deve_conter", [])
        nao_dev = caso.get("nao_deve_conter", [])
        cpf_t   = caso.get("cpf_teste")

        intencao = classificar(pergunta)

        # Contexto CRM se necessário
        pergunta_final = pergunta
        if caso.get("uso_crm") and cpf_t:
            cpf_d = re.sub(r"[^\d]", "", cpf_t)
            cliente = crm.get(cpf_d)
            if cliente:
                ctx = montar_ctx(cliente)
                nome = cliente['nome'].split()[0]
                pergunta_final = f"{ctx}\n\nCom base nesses dados do beneficiário {nome}, responda: {pergunta}"

        try:
            resultado = chain.invoke({"query": pergunta_final})
            resposta = resultado.get("result", "").strip()
            for tag in ["</s>", "<s>", "[INST]", "[/INST]"]:
                resposta = resposta.replace(tag, "").strip()
        except Exception as e:
            resposta = f"ERRO: {e}"

        resp_lower = resposta.lower()
        passou = True
        falhas = []

        for termo in devem:
            if termo.lower() not in resp_lower:
                passou = False
                falhas.append(f"FALTOU: '{termo}'")

        for termo in nao_dev:
            if termo.lower() in resp_lower:
                passou = False
                falhas.append(f"PRESENTE (não deveria): '{termo}'")

        status = "✅ PASS" if passou else "❌ FAIL"
        if passou:
            aprovados += 1
        else:
            reprovados += 1

        print(f"\n{status} | {pid} [{caso['complexidade']}] | Intent: {intencao}")
        print(f"  Pergunta: {pergunta[:70]}")
        print(f"  Resposta: {resposta[:120]}...")
        if falhas:
            for f_ in falhas:
                print(f"  ⚠️  {f_}")

    total = aprovados + reprovados
    print(f"\n{'='*70}")
    print(f"Resultado: {aprovados}/{total} aprovados "
          f"({100*aprovados//total}%) | {reprovados} reprovados")
    print("=" * 70)


if __name__ == "__main__":
    main()
