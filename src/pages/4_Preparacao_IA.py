"""
Página 4: Inteligência Artificial e Classificação de Sentimento (LLM Local)
Descrição: Consome o endpoint POST /processar_texto da FastAPI para classificar
notícias e textos sobre sustentabilidade utilizando o modelo DistilBERT local.
"""

import os
import sys
import json
import requests
import streamlit as st
import pandas as pd

RAIZ_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if RAIZ_PROJETO not in sys.path:
    sys.path.insert(0, RAIZ_PROJETO)

from src.data.preparacao_llm import processar_corpus_para_llm

st.set_page_config(page_title="IA & LLM Local - ESG", page_icon="🤖", layout="wide")

st.title("🤖 Análise de Sentimento ESG com LLM Local")
st.caption("Integração entre Streamlit, FastAPI e HuggingFace Transformers (DistilBERT Multilingual).")

URL_API = "http://127.0.0.1:8000"

# ==========================================================
# 1. INTERAÇÃO EM TEMPO REAL COM O ENDPOINT POST /processar_texto
# ==========================================================
st.subheader("🧠 Classificador de Sentimento em Tempo Real")
st.write("Envie um texto ou selecione uma notícia coletada para inferência local via FastAPI:")

caminho_json = os.path.join("data", "processed", "dataset_llm_prep.json")
opcoes_textos = ["Digitar texto personalizado..."]

if os.path.exists(caminho_json):
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados_llm = json.load(f)
    opcoes_textos.extend([item["texto"] for item in dados_llm[:15]])

escolha = st.selectbox("Selecione uma amostra do corpus ou digite abaixo:", opcoes_textos)

texto_para_analise = st.text_area(
    "Texto para processamento pelo modelo local:",
    value="" if escolha == "Digitar texto personalizado..." else escolha,
    height=100,
    placeholder="Ex.: Prefeitura inaugura novo parque solar e reduz emissão de carbono em 30%."
)

if st.button("Analisar Sentimento via API (LLM Local)"):
    with st.spinner("Processando inferência no modelo DistilBERT local..."):
        try:
            resposta = requests.post(
                f"{URL_API}/processar_texto",
                json={"texto": texto_para_analise},
                timeout=15
            )
            
            if resposta.status_code == 200:
                resultado = resposta.json()
                st.success("Análise concluída e validada pelo Response Model!")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Sentimento Identificado", resultado["sentimento"])
                c2.metric("Confiança do Modelo", f"{resultado['confianca'] * 100:.2f}%")
                c3.metric("Latência CPU Local", f"{resultado['tempo_processamento_ms']:.1f} ms")
                
                with st.expander("Visualizar JSON Estruturado (Response Model)"):
                    st.json(resultado)
            else:
                corpo_erro = resposta.json()
                detalhe_erro = corpo_erro.get("detail", resposta.text)
                sugestao = corpo_erro.get("sugestao", "")
                st.error(f"**Erro HTTP {resposta.status_code}:** {detalhe_erro}")
                if sugestao:
                    st.info(f"💡 **Orientação:** {sugestao}")
                
        except requests.exceptions.Timeout:
            st.error("Erro de Timeout (504): O servidor demorou mais de 15 segundos para responder.")
        except requests.exceptions.ConnectionError:
            st.error("Falha de Conexão (503): Não foi possível conectar ao servidor FastAPI na porta 8000.")
        except Exception as e:
            st.error(f"Erro inesperado na requisição: {e}")

st.divider()

# ==========================================================
# 2. PIPELINE DE SANITIZAÇÃO DO CORPUS
# ==========================================================
st.subheader("⚙️ Base Textual Sanitizada para IA")

col1, col2 = st.columns([1, 2])

with col1:
    st.write("Atualize o dataset sanitizado em `data/processed/dataset_llm_prep.json` a partir dos textos coletados:")
    if st.button("Reprocessar Corpus Textual"):
        docs = processar_corpus_para_llm()
        st.success(f"Pipeline concluído! {len(docs)} documentos sanitizados.")
        st.rerun()

with col2:
    if os.path.exists(caminho_json):
        with open(caminho_json, "r", encoding="utf-8") as f:
            dados_llm = json.load(f)
        st.metric("Documentos Disponíveis para Inferência", len(dados_llm))
        df_llm = pd.DataFrame(dados_llm)
        st.dataframe(df_llm[["id", "texto", "tamanho_caracteres"]], width="stretch", hide_index=True)