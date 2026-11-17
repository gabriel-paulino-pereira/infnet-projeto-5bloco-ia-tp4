# Seleção e Justificativa do Modelo de Linguagem (LLM Local) - Fase TP4

## 1. Modelo Selecionado
- **Identificador no HuggingFace Hub:** `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
- **Arquitetura Base:** DistilBERT (Transformer encoder destilado a partir do `bert-base-multilingual-cased`)
- **Tarefa Principal:** Classificação de Sentimentos Multilíngue (com suporte nativo ao Português)
- **Aplicação no Projeto (ODS 11 / ESG):** Classificação automatizada do teor das notícias e ocorrências urbanas coletadas (`Positivo`, `Neutro` ou `Negativo`), permitindo triagem rápida entre avanços sustentáveis e alertas de impacto ambiental.

---

## 2. Análise dos Critérios de Seleção

### 2.1. Desempenho e Adequação à Tarefa
- **Precisão e Suporte Linguístico:** O modelo foi treinado via destilação de conhecimento (*Knowledge Distillation*) a partir de um modelo professor *zero-shot* (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`), mantendo alta acurácia na interpretação semântica em português sem exigir *fine-tuning* adicional.
- **Tempo de Resposta (Latência):** Por possuir 40% menos parâmetros que o BERT tradicional (~135M de parâmetros), entrega inferências em frações de segundo (~100ms a 200ms por texto) mesmo sem aceleração por GPU dedicada, tornando-o ideal para consumo síncrono em endpoints REST do FastAPI.

### 2.2. Custo Computacional Local (Compatibilidade de Hardware)
A seleção foi dimensionada diretamente para o ambiente de desenvolvimento local disponível:
- **Processador:** AMD Ryzen 7 7735HS (8 núcleos / 16 threads, 3.20 GHz) — excelente desempenho multi-thread para operações matriciais em CPU.
- **Memória RAM:** 16,0 GB instalados (13,7 GB utilizáveis) — o modelo consome cerca de **1,3 GB de RAM** após carregado, evitando paginação em disco (*swap*) e preservando recursos para o servidor Uvicorn e o frontend Streamlit.
- **Placa Gráfica:** AMD Radeon Graphics integrada (2 GB VRAM) no Windows — como o ecossistema padrão do PyTorch no Windows prioriza CUDA (NVIDIA), o pipeline foi configurado para operar estritamente em **CPU (`device=-1`)**, eliminando incompatibilidades de drivers gráficos.
- **Armazenamento:** ~542 MB necessários para download dos pesos (`model.safetensors`) e tokenizador em cache local (`~/.cache/huggingface/`).

### 2.3. Acessibilidade Local e Ecossistema
- **Disponibilidade Open-Source:** Hospedado publicamente no HuggingFace Model Hub sob licença Apache 2.0, sem necessidade de chaves de API externas, custos de assinatura ou envio de dados para a nuvem.
- **Integração Nativa:** Carregado localmente através das bibliotecas `transformers` e `torch`, permitindo cache em disco após o primeiro download e operação 100% *offline* nas execuções subsequentes.

---

## 3. Configuração do Ambiente Local
Para habilitar a execução local do modelo na máquina Windows, foram instaladas as seguintes dependências no ambiente virtual (`venv`):
- `transformers`: Carregamento do tokenizador, arquitetura do modelo e abstração `pipeline`.
- `torch` (CPU): Motor de execução de tensores otimizado para processadores x86_64 sem dependência de CUDA.