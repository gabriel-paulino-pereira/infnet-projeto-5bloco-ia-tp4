# Plataforma de Inteligência de Dados ESG & ODS 11 com LLM Local

Aplicação interativa desenvolvida ao longo do **Projeto de Bloco: Inteligência Artificial Aplicada** (Instituto Infnet), estruturada segundo os ciclos de vida **TDSP (Team Data Science Process)** e **CRISP-DM**. A solução realiza coleta automatizada de notícias sobre sustentabilidade urbana (ODS 11 e ODS 12), disponibiliza os dados via API RESTful e executa análise de sentimento localmente por meio de um modelo de linguagem (LLM) da HuggingFace.

---

## Arquitetura e Funcionalidades por Etapa

- **TP1 — Estruturação e Coleta Estática:** Configuração do ambiente virtual (`venv`), governança inicial (`docs/project_charter.md`) e pipeline de Web Scraping estático com `BeautifulSoup` e `Requests` (`src/data/scraping_noticias.py`) para extração de notícias da Agência Brasil.
- **TP2 — Interface Interativa e Gestão de Dados:** Desenvolvimento do painel analítico em `Streamlit` com gerenciamento de estado (`st.session_state`), cache de leitura (`@st.cache_data`), filtros dinâmicos, além de upload e exportação de datasets em CSV.
- **TP3 — Arquitetura Multipages, Scraping Dinâmico e API RESTful:**
  - Evolução do frontend para arquitetura multipáginas (`src/pages/`).
  - Implementação de coleta dinâmica em páginas renderizadas por JavaScript utilizando `Selenium` (`src/data/scraping_dinamico.py`).
  - Construção de pipeline de sanitização e preparação de corpus textual (`src/data/preparacao_llm.py`).
  - Criação do backend em `FastAPI` (`src/api/main.py`) com rotas `GET` e `POST` integradas à interface.
- **TP4 — Integração de LLM Local, Response Models e Tratamento de Erros:**
  - Integração local do modelo `lxyuan/distilbert-base-multilingual-cased-sentiments-student` via `HuggingFace Transformers` e `PyTorch` operando em CPU (`device=-1`).
  - Endpoint `POST /processar_texto` para classificação de sentimento ESG (`Positivo`, `Neutro`, `Negativo`) com medição de confiança e latência.
  - Validação estrita de entrada e saída em todas as rotas com `Response Models` (`Pydantic`).
  - Tratamento robusto de exceções HTTP (`400`, `404`, `422`, `500`, `503`, `504`) e manipulador global de erros de validação.

---

## Estrutura de Diretórios

```text
tp1/
├── data/
│   ├── processed/
│   │   └── dataset_llm_prep.json          # Corpus textual sanitizado para inferência
│   ├── raw/
│   │   ├── corpus_noticias.txt            # Texto bruto consolidado
│   │   ├── noticias_dinamicas.csv         # Coleta dinâmica via Selenium
│   │   └── noticias_sustentabilidade.csv  # Coleta estática via BeautifulSoup
│   └── sample/                            # Amostras de dados para testes
├── docs/
│   ├── data_summary_report.md             # Relatório exploratório dos dados
│   ├── modelo_llm_justificativa.md        # Justificativa técnica do modelo LLM local (TP4)
│   └── project_charter.md                 # Termo de abertura e escopo do projeto
├── notebooks/                             # Notebooks de exploração e prototipação
├── src/
│   ├── api/
│   │   └── main.py                        # Servidor FastAPI + LLM local + Response Models
│   ├── data/
│   │   ├── preparacao_llm.py              # Sanitização e estruturação de textos para IA
│   │   ├── scraping_dinamico.py           # Extração automatizada com Selenium
│   │   └── scraping_noticias.py           # Extração estática com BeautifulSoup
│   ├── pages/
│   │   ├── 1_Painel_Principal.py          # Visão geral e indicadores ESG
│   │   ├── 2_Gestao_Arquivos.py           # Upload, visualização e download de CSVs
│   │   ├── 3_API_FastAPI.py               # Consumo das rotas CRUD da FastAPI
│   │   └── 4_Preparacao_IA.py             # Classificação de sentimento em tempo real via LLM
│   ├── utils/                             # Funções auxiliares
│   └── app.py                             # Ponto de entrada da aplicação Streamlit
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Requisitos de Hardware e Ambiente

O modelo local foi selecionado e otimizado para execução 100% *offline* sem necessidade de GPU dedicada:
- **Sistema Operacional:** Windows
- **Processador:** AMD Ryzen 7 7735HS (inferência em CPU via `device=-1`)
- **Memória RAM:** 16 GB (consumo aproximado de ~1,3 GB pelo modelo DistilBERT carregado em memória)
- **Armazenamento:** ~550 MB em cache local (`~/.cache/huggingface/`) após o primeiro download

---

## Como Executar Localmente

### 1. Clonar o repositório e ativar o ambiente virtual
```powershell
git clone <URL_DO_REPOSITORIO>
cd tp1
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Instalar as dependências
```powershell
pip install -r requirements.txt
```

### 3. Iniciar o Backend FastAPI (Terminal 1)
Na primeira inicialização, o modelo `lxyuan/distilbert-base-multilingual-cased-sentiments-student` será baixado automaticamente para o cache local:
```powershell
uvicorn src.api.main:app --reload
```
- **Documentação Interativa (Swagger UI):** `http://127.0.0.1:8000/docs`

### 4. Iniciar o Frontend Streamlit (Terminal 2)
Em um segundo terminal com a `venv` ativada:
```powershell
streamlit run src/app.py
```
- **Interface Web:** `http://localhost:8501`

---

## Principais Endpoints da API

| Método | Rota | Descrição | Response Model |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Verifica status da API e carregamento do modelo local | `StatusAPIResponse` |
| `GET` | `/noticias` | Lista todas as notícias consolidadas na base | `List[NoticiaResponse]` |
| `GET` | `/noticias/{noticia_id}` | Consulta uma notícia específica pelo ID | `NoticiaResponse` |
| `POST` | `/noticias` | Cadastra uma nova notícia na base em memória | `CadastroNoticiaResponse` |
| `POST` | `/processar_texto` | Executa classificação de sentimento local via DistilBERT | `ProcessamentoTextoResponse` |