"""
InsurMinds Saúde — Chatbot Telegram v2.0
=========================================
Arquitetura de 3 camadas (conforme documento lean):
  Camada 1 — Intent Classification (roteamento por intenção)
  Camada 2 — RAG Retrieval (FAISS + corpus de plano de saúde)
  Camada 3 — Guardrails (LGPD, anti-alucinação, compliance)

Personas suportadas:
  - Beneficiário Idoso   → linguagem simples, perguntas curtas
  - RH Empresarial       → perguntas técnicas, múltiplos beneficiários
  - Usuário Premium      → alta exigência, resposta rápida e precisa
  - Jovem Digital        → mensagens rápidas, linguagem informal, erros ortográficos
"""

import os
import re
import asyncio
import json
import logging
import threading
from pathlib import Path
from dotenv import load_dotenv

from flask import Flask
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_API_TOKEN   = os.getenv("HF_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

BASE_DIR  = Path(__file__).parent
FAQ_FILE  = BASE_DIR / "data" / "corpus_faqs.json"
CRM_FILE  = BASE_DIR / "data" / "mock_crm.json"
FAISS_DIR = BASE_DIR / "faiss_index"

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

CPF_REGEX = re.compile(r"\b\d{3}[.\s]?\d{3}[.\s]?\d{3}[-.\s]?\d{2}\b")

# ─────────────────────────────────────────────
# Servidor Web (Flask) para Deploy no Render
# ─────────────────────────────────────────────
app_web = Flask(__name__)

@app_web.route("/")
def index():
    return "✅ InsurMinds Saúde Bot está rodando online!"

def run_web():
    port = int(os.getenv("PORT", 8080))
    app_web.run(host="0.0.0.0", port=port)

# ─────────────────────────────────────────────
# CAMADA 1 — Intent Classification
# ─────────────────────────────────────────────
INTENT_MAP = {
    "financeiro": [
        "boleto", "2 via", "segunda via", "pagar", "vencimento",
        "fatura", "pagamento", "mensalidade", "débito", "reajuste", "aumento",
        "coparticipação", "cancelamento",
    ],
    "reembolso": [
        "reembolso", "reembouço", "reembolsar", "dinheiro de volta",
        "ressarcimento", "cadê meu dinheiro", "demorou", "atrasado",
    ],
    "rede_credenciada": [
        "hospital", "clinica", "médico", "dentista", "perto", "credenciado",
        "laboratório", "rede", "onde fica", "endereço",
    ],
    "cobertura": [
        "cobre", "coberto", "cobertura", "inclui", "tem", "exame",
        "cirurgia", "consulta", "procedimento", "tratamento", "carência",
    ],
    "autorizacao": [
        "autorização", "autorizado", "autorizar", "liberar", "aprovado",
        "senha médica", "meu exame foi",
    ],
    "telemedicina": [
        "telemedicina", "consulta online", "médico online", "teleconsulta",
        "video", "digital",
    ],
    "emergencia": [
        "emergência", "urgência", "pronto socorro", "hospital agora",
        "acidente", "uti", "infarto", "avc",
    ],
    "dependente": [
        "dependente", "filho", "cônjuge", "esposa", "marido",
        "incluir", "inclusão", "autista", "deficiência", "movimentação cadastral",
    ],
    "portabilidade": [
        "trocar de plano", "portabilidade", "mudar de plano", "migrar",
        "outro plano", "carência ao trocar",
    ],
    "reclamacao": [
        "indignado", "absurdo", "péssimo", "demora", "revoltado",
        "reclamar", "reclamação", "negligência",
    ],
    "atendente": [
        "atendente", "humano", "pessoa", "falar com alguém",
        "suporte", "gerente",
    ],
    "odonto": [
        "odonto", "dentista", "dente", "canal", "extração", "tártaro",
        "ortodontia", "radiografia", "limpeza",
    ],
    "cadastro": [
        "carteirinha", "documentação", "dados",
    ],
}

EMOCOES_NEGATIVAS = [
    "indignado", "revoltado", "absurdo", "ódio", "péssimo",
    "horrível", "furioso", "irritado", "nervoso", "decepcionado",
]

ALUCINACAO_TRIGGERS = [
    "marte", "lua", "planeta", "outro país", "universo",
    "extraterrestre", "ficção",
]


def classificar_intencao(texto: str) -> str:
    txt = texto.lower()
    for intent, palavras in INTENT_MAP.items():
        if any(p in txt for p in palavras):
            return intent
    return "geral"


def detectar_emocao(texto: str) -> bool:
    txt = texto.lower()
    return any(e in txt for e in EMOCOES_NEGATIVAS)


def detectar_alucinacao(texto: str) -> bool:
    txt = texto.lower()
    return any(t in txt for t in ALUCINACAO_TRIGGERS)


# ─────────────────────────────────────────────
# CAMADA 3 — Guardrails
# ─────────────────────────────────────────────
GUARDRAIL_ALUCINACAO = (
    "❌ Não encontrei informação sobre esse procedimento ou cobertura. "
    "Consulte as condições gerais do seu plano ou ligue para nossa central. "
    "Posso ajudar com outra dúvida?"
)

GUARDRAIL_ATENDENTE = (
    "👤 Certo! Vou te transferir para um atendente humano.\n\n"
    "📞 Central de Atendimento: *0800 000 0000*\n"
    "🕐 Horário: Segunda a Sexta, 8h às 20h | Sábado 8h às 14h\n\n"
    "Ou acesse o chat no app para atendimento em tempo real."
)

GUARDRAIL_EMOCAO = (
    "Compreendo sua frustração e sinto muito pelo transtorno. 😔 "
    "Isso não é o padrão que queremos para você. "
    "Me informe seu *CPF* e verei o que está acontecendo para resolver agora."
)

# ─────────────────────────────────────────────
# CAMADA 2 — RAG: Carregar e construir FAISS
# ─────────────────────────────────────────────
def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )


def carregar_ou_construir_faiss() -> FAISS:
    embeddings = _get_embeddings()
    if FAISS_DIR.exists():
        logger.info("Carregando índice FAISS existente...")
        return FAISS.load_local(
            str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True
        )
    logger.info("Construindo índice FAISS do corpus...")
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(FAISS_DIR))
    logger.info(f"FAISS salvo com {len(chunks)} chunks.")
    return vs


# ─────────────────────────────────────────────
# CRM
# ─────────────────────────────────────────────
def carregar_crm() -> dict:
    with open(CRM_FILE, encoding="utf-8") as f:
        data = json.load(f)
    crm = {}
    for c in data["clientes"]:
        chave = re.sub(r"[^\d]", "", c["cpf"])
        crm[chave] = c
    logger.info(f"CRM carregado: {len(crm)} clientes")
    return crm


CRM: dict = carregar_crm()


def extrair_cpf(texto: str) -> str | None:
    m = CPF_REGEX.search(texto)
    return re.sub(r"[^\d]", "", m.group(0)) if m else None


def montar_contexto_crm(cliente: dict) -> str:
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


# ─────────────────────────────────────────────
# LangChain Chain
# ─────────────────────────────────────────────
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


def criar_chain(vectorstore: FAISS) -> RetrievalQA:
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
        logger.info(f"Chat LLM do Google Gemini (gemini-2.0-flash) configurado.")
    except Exception as e:
        logger.error(f"Erro ao configurar o Google Gemini. Verifique sua API Key e permissões: {e}")
        llm = None
        
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": SYSTEM_PROMPT},
        return_source_documents=False,
    )


def limpar_resposta(texto: str) -> str:
    for tag in ["</s>", "<s>", "[INST]", "[/INST]", "Resposta:"]:
        texto = texto.replace(tag, "")
    return texto.strip()


# ─────────────────────────────────────────────
# Teclado de menu
# ─────────────────────────────────────────────
MENU_KEYBOARD = ReplyKeyboardMarkup(
    [
        ["1️⃣ Cobertura", "2️⃣ Reembolso"],
        ["3️⃣ Rede Credenciada", "4️⃣ Financeiro"],
        ["5️⃣ Autorizações", "6️⃣ Falar com atendente"],
    ],
    resize_keyboard=True,
    one_time_keyboard=False,
)

BOAS_VINDAS = (
    "Olá! Sou a assistente virtual da *Seguradora InsurMinds* 😊\n\n"
    "Posso ajudar com:\n\n"
    "1️⃣ Cobertura\n"
    "2️⃣ Reembolso\n"
    "3️⃣ Rede Credenciada\n"
    "4️⃣ Financeiro\n"
    "5️⃣ Autorizações\n"
    "6️⃣ Falar com atendente\n\n"
    "Ou simplesmente me escreva sua dúvida em linguagem natural! 💬\n"
    "Para consultas personalizadas, informe seu *CPF* 🔒"
)


# ─────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Comando /start recebido.")
    if context.user_data is not None:
        context.user_data.clear()
    if update.effective_message:
        await update.effective_message.reply_text(
            BOAS_VINDAS,
            parse_mode="Markdown",
            reply_markup=MENU_KEYBOARD,
        )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message:
        await update.effective_message.reply_text(
        "📌 *Comandos:*\n\n"
        "/start — Reiniciar atendimento\n"
        "/help — Esta mensagem\n"
        "/limpar — Limpar dados da sessão\n\n"
        "📞 Central 24h: *0800 000 0000*\n"
        "🌐 ANS: 0800 701 9656",
        parse_mode="Markdown",
    )


async def cmd_limpar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data is not None:
        context.user_data.clear()
    if update.effective_message:
        await update.effective_message.reply_text(
            "🗑️ Sessão limpa. Seus dados foram removidos desta conversa.",
            reply_markup=MENU_KEYBOARD,
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
        
    texto = msg.text.strip()
    chain: RetrievalQA = context.application.bot_data["chain"]

    # ── GUARDRAIL: Alucinação ────────────────
    if detectar_alucinacao(texto):
        await msg.reply_text(GUARDRAIL_ALUCINACAO, parse_mode="Markdown")
        return

    # ── CAMADA 1: Intent ─────────────────────
    intencao = classificar_intencao(texto)
    logger.info(f"Mensagem recebida: '{texto[:60]}' | Intenção: {intencao}")

    # ── Atendente humano ─────────────────────
    if intencao == "atendente":
        await msg.reply_text(GUARDRAIL_ATENDENTE, parse_mode="Markdown")
        return

    # ── GUARDRAIL: Emoção negativa ───────────
    if detectar_emocao(texto):
        await msg.reply_text(GUARDRAIL_EMOCAO, parse_mode="Markdown")
        return

    # ── Detecta CPF ──────────────────────────
    cpf_digits = extrair_cpf(texto)
    if cpf_digits:
        cpfs_vistos = context.user_data.get("cpfs_vistos", set()) if context.user_data is not None else set()
        if cpf_digits not in cpfs_vistos:
            cliente = CRM.get(cpf_digits)
            if cliente:
                if context.user_data is not None:
                    context.user_data["contexto_crm"] = montar_contexto_crm(cliente)
                    context.user_data["nome_cliente"] = cliente["nome"].split()[0]
                    cpfs_vistos.add(cpf_digits)
                    context.user_data["cpfs_vistos"] = cpfs_vistos
                nome = cliente["nome"].split()[0]
                await msg.reply_text(
                    f"✅ Olá, *{nome}*! Encontrei seu cadastro. Como posso ajudar?",
                    parse_mode="Markdown",
                )
            else:
                await msg.reply_text(
                    "❌ Não localizei esse CPF. Verifique o número ou ligue: *0800 000 0000*",
                    parse_mode="Markdown",
                )
                return

    # ── Monta pergunta com contexto CRM ──────
    user_data = context.user_data if context.user_data is not None else {}
    contexto_crm = user_data.get("contexto_crm", "")
    if contexto_crm:
        nome = user_data.get("nome_cliente", "")
        pergunta_final = (
            f"{contexto_crm}\n\n"
            f"Com base nesses dados do beneficiário {nome}, responda: {texto}"
        )
    else:
        pergunta_final = texto

    # ── Envia "digitando…" ────────────────────
    await msg.chat.send_action("typing")

    try:
        loop = asyncio.get_running_loop()
        resultado = await loop.run_in_executor(
            None,
            lambda: chain.invoke({"query": pergunta_final}),
        )
        resposta = limpar_resposta(resultado.get("result", ""))

        if not resposta or len(resposta) < 10:
            resposta = (
                "Não encontrei uma resposta precisa para isso. "
                "Ligue para nossa central: *0800 000 0000*."
            )
    except Exception as e:
        logger.error(f"Erro LLM: {e}")
        resposta = (
            "⚠️ Tive um problema técnico. Tente novamente em instantes "
            "ou ligue: *0800 000 0000*."
        )

    try:
        await msg.reply_text(resposta, parse_mode="Markdown")
    except BadRequest as e:
        logger.warning(f"Erro de formatação Markdown. Reenviando como texto puro. Detalhe: {e}")
        await msg.reply_text(resposta)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN não definido no .env")
    if not HF_API_TOKEN:
        raise ValueError("HF_API_TOKEN não definido no .env")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY não definido no .env")

    logger.info("Carregando/construindo índice FAISS...")
    vectorstore = carregar_ou_construir_faiss()

    logger.info("Criando chain LangChain...")
    chain = criar_chain(vectorstore)

    logger.info("Iniciando servidor web Flask em thread separada...")
    web_thread = threading.Thread(target=run_web)
    web_thread.daemon = True
    web_thread.start()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.bot_data["chain"] = chain

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("limpar", cmd_limpar))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("✅ InsurMinds Saúde Bot rodando. Ctrl+C para encerrar.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
