import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

# Baixar dados necessários do NLTK
@st.cache_resource
def baixar_dados_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

baixar_dados_nltk()

# Carregar e pré-processar dados
@st.cache_data
def carregar_e_preprocessar_dados():
    df = pd.read_csv('all_movies.csv', sep=';')
    
    # Função de pré-processamento de texto
    def preprocessar_texto(texto):
        # Converter para minúsculas
        texto = texto.lower()
        # Remover caracteres especiais e dígitos
        texto = re.sub(r'[^a-zA-Z\s]', '', texto)
        # Tokenizar (dividir por espaços)
        tokens = texto.split()
        # Remover stopwords
        stop_words = set(stopwords.words('portuguese'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    
    df['sinopse_processada'] = df['sinopse'].apply(preprocessar_texto)
    return df

# Treinar modelo e adicionar cluster ao DataFrame
@st.cache_resource
def treinar_modelo_e_agrupar(df):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['sinopse_processada'])
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    df_com_clusters = df.copy()
    df_com_clusters['cluster'] = clusters
    
    return vectorizer, kmeans, X, df_com_clusters

# Carregar dados, treinar modelo e adicionar clusters
df = carregar_e_preprocessar_dados()
vectorizer, modelo, X, df_com_clusters = treinar_modelo_e_agrupar(df)

# Código do aplicativo Streamlit
st.title('Sistema de Recomendação de Filmes')

metodo = st.radio("Escolha um método de recomendação:", 
                  ('1. Escolher entre sinopses', '2. Escrever sua própria sinopse'), 
                  key="metodo_recomendacao")

if metodo == '1. Escolher entre sinopses':
    st.subheader("Escolha uma sinopse que você goste:")
    sinopses_amostra = df_com_clusters.sample(n=3)['sinopse'].tolist()
    sinopse_escolhida = st.radio("Selecione uma sinopse:", sinopses_amostra, key="selecao_sinopse")
    
    if st.button('Obter Recomendações', key="obter_recomendacoes_1"):
        filme_escolhido = df_com_clusters[df_com_clusters['sinopse'] == sinopse_escolhida].iloc[0]
        cluster = filme_escolhido['cluster']
        
        # Obter filmes do mesmo cluster
        filmes_cluster = df_com_clusters[df_com_clusters['cluster'] == cluster]
        
        # Ordenar por avaliação e selecionar os 5 melhores
        recomendacoes = filmes_cluster.sort_values('rating', ascending=False).head(5)
        
        st.subheader("Filmes Recomendados:")
        for _, filme in recomendacoes.iterrows():
            st.write(f"{filme['title_pt']} ({filme['year']}) - Avaliação: {filme['rating']}")
            st.write(f"Gênero: {filme['genre']}")
            st.write(f"Sinopse: {filme['sinopse']}")
            st.write("---")

elif metodo == '2. Escrever sua própria sinopse':
    sinopse_usuario = st.text_area("Escreva uma sinopse de filme:", key="sinopse_usuario")
    
    if st.button('Obter Recomendações', key="obter_recomendacoes_2"):
        # Pré-processar e vetorizar a sinopse do usuário
        sinopse_processada = carregar_e_preprocessar_dados().preprocessar_texto(sinopse_usuario)
        vetor_usuario = vectorizer.transform([sinopse_processada])
        
        # Prever o cluster
        cluster_usuario = modelo.predict(vetor_usuario)[0]
        
        # Obter filmes do mesmo cluster
        filmes_cluster = df_com_clusters[df_com_clusters['cluster'] == cluster_usuario]
        
        # Calcular similaridade com a sinopse do usuário
        vetores_cluster = vectorizer.transform(filmes_cluster['sinopse_processada'])
        similaridades = cosine_similarity(vetor_usuario, vetores_cluster).flatten()
        
        # Ordenar por similaridade e selecionar os 5 melhores
        indices_similares = similaridades.argsort()[-5:][::-1]
        recomendacoes = filmes_cluster.iloc[indices_similares]
        
        st.subheader("Filmes Recomendados:")
        for _, filme in recomendacoes.iterrows():
            st.write(f"{filme['title_pt']} ({filme['year']}) - Avaliação: {filme['rating']}")
            st.write(f"Gênero: {filme['genre']}")
            st.write(f"Sinopse: {filme['sinopse']}")
            st.write("---")

st.sidebar.write("Este aplicativo usa um modelo de agrupamento para recomendar filmes com base nas suas preferências.")

