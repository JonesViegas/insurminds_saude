# 🏥 InsurMinds Saúde — Chatbot Telegram v2.0

Bot de atendimento de **plano de saúde** via Telegram com arquitetura RAG de 3 camadas, Mock CRM e Golden Dataset de 17 casos.

---

## 📁 Estrutura do Projeto

```
insurminds_saude/
├── bot.py                     ← Bot principal (3 camadas)
├── build_index.py             ← Gera índice FAISS (1x)
├── requirements.txt
├── .env.example               → copie para .env
│
├── data/
│   ├── corpus_faqs.json       ← 20 FAQs de plano de saúde (RAG)
│   ├── mock_crm.json          ← 4 clientes (4 personas)
│   └── golden_dataset.json    ← 17 casos de teste
│
├── faiss_index/               ← Gerado pelo build_index.py
│
├── tests/
│   └── test_golden.py         ← Validação automática
│
└── .vscode/
    ├── launch.json            ← F5 para rodar no VSCode
    └── settings.json
```

---

## 🏗️ Arquitetura — 3 Camadas (Lean Doc)

```
Mensagem do usuário
        │
┌───────▼────────────────────────┐
│  CAMADA 1 — Intent Classifier  │  financeiro, reembolso, cobertura,
│  (regex + keywords)            │  autorizacao, emergencia, atendente...
└───────┬────────────────────────┘
        │
┌───────▼────────────────────────┐
│  CAMADA 2 — RAG Retrieval      │  FAISS + corpus_faqs.json
│  LangChain + Mistral-7B        │  + injeção de contexto CRM via CPF
└───────┬────────────────────────┘
        │
┌───────▼────────────────────────┐
│  CAMADA 3 — Guardrails         │  Anti-alucinação, LGPD,
│                                │  detecção de emoção, compliance
└───────┬────────────────────────┘
        │
   Resposta ao usuário
```

---

## 👥 Personas Suportadas

| Persona | Exemplo de mensagem | Como o bot responde |
|---|---|---|
| Beneficiário Idoso | "meu exame foi autorizado?" | Linguagem simples, pede CPF |
| RH Empresarial | "regra de inclusão de dependente?" | Resposta técnica completa |
| Usuário Premium | "meu reembolso está atrasado" | Busca CRM + protocolo |
| Jovem Digital | "boleto" / "2 via" | Resposta direta e rápida |

---

## 🚀 Instalação Passo a Passo (VSCode)

### 1. Criar o Bot no Telegram
1. Abra o Telegram → procure `@BotFather`
2. Envie `/newbot` e siga as instruções
3. Copie o **token** gerado

### 2. Token Hugging Face
1. Acesse [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Crie um token **Read**

### 3. Abrir no VSCode
```bash
# Abra a pasta no VSCode
code insurminds_saude/
```

### 4. Configurar ambiente
```bash
# Terminal integrado do VSCode (Ctrl+`)
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 5. Configurar .env
```bash
cp .env.example .env
# Edite o .env com seus tokens
```

### 6. Build do índice FAISS
No VSCode, pressione **F5** e selecione **"🔧 Build FAISS Index"**

Ou pelo terminal:
```bash
python build_index.py
```

### 7. Testar o pipeline
```bash
python tests/test_golden.py
```
Exibe pass/fail para os 17 casos do Golden Dataset.

### 8. Rodar o bot
Pressione **F5** → **"▶ Rodar Bot"**

Ou:
```bash
python bot.py
```

Acesse o Telegram, encontre seu bot e envie `/start` 🎉

---

## 💬 Exemplos de Interação

```
Usuário: /start
Bot: Olá! Sou a assistente virtual da Seguradora InsurMinds 😊
     [menu com 6 opções]

Usuário: boleto
Bot: A segunda via do boleto pode ser obtida pelo app ou portal...

Usuário: meu reembolso está atrasado
Bot: Compreendo sua frustração. Me informe seu CPF e verifico agora.

Usuário: 333.333.333-33
Bot: ✅ Olá, Roberto! Encontrei seu cadastro.

Usuário: qual o status do meu reembolso?
Bot: Roberto, o sinistro SIN-9921 (reembolso de consulta psiquiátrica)
     está em análise desde 10/04. O prazo legal é 30 dias...

Usuário: Vocês cobrem cirurgia em Marte?
Bot: ❌ Não encontrei informação sobre esse procedimento...
```

---

## 🧪 Golden Dataset — 17 Casos

| Tipo | Qtd | Exemplos |
|---|---|---|
| Simples | 6 | "boleto", "prazo de reembolso", "telemedicina?" |
| Compostas | 5 | CRM + reembolso atrasado, cobertura + rede |
| Ambíguas | 1 | "Não consigo usar meu plano" |
| Edge Cases | 2 | Filho autista após 24 anos, autorização negada |
| Informais | 1 | "cadê meu dinheiro do reembolso" |
| Ortografia | 1 | "reembouço demorando" |
| Emoções | 1 | "Estou indignado com essa demora" |
| Hallucination | 1 | "cobrem cirurgia em Marte?" |

---

## ⚙️ Personalização

**Trocar o LLM** — em `bot.py`, altere `repo_id`:
```python
"mistralai/Mistral-7B-Instruct-v0.3"  # padrão
"google/flan-t5-large"                 # mais leve
```

**Adicionar FAQs** — edite `data/corpus_faqs.json` e rode `build_index.py` novamente.

**Adicionar clientes** — edite `data/mock_crm.json`.

---

## 📞 Contatos de Referência (PoC)
- ANS: 0800 701 9656 | www.ans.gov.br
- Central fictícia InsurMinds: 0800 000 0000
