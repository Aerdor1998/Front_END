import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Configuração da página
st.set_page_config(page_title="Sistema de Recomendação de Filmes", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('all_movies.csv', sep=';')
    return df

@st.cache_resource
def load_models():
    tfidf = joblib.load('tfidf_model.joblib')
    kmeans_full = joblib.load('kmeans_full_model.joblib')
    scaler_full = joblib.load('scaler_full_model.joblib')
    kmeans_synopses = joblib.load('kmeans_synopses_model.joblib')
    return tfidf, kmeans_full, scaler_full, kmeans_synopses

# Carregando dados e modelos
df = load_data()
tfidf, kmeans_full, scaler_full, kmeans_synopses = load_models()

# Funções de recomendação
def recommend_movies_method1(chosen_synopsis, n_recommendations=5):
    chosen_tfidf = tfidf.transform([chosen_synopsis])
    chosen_features = np.hstack(([df['year'].mean()], chosen_tfidf.toarray()[0]))
    chosen_scaled = scaler_full.transform([chosen_features])
    predicted_cluster = kmeans_full.predict(chosen_scaled)[0]
    cluster_movies = df[df['cluster'] == predicted_cluster]
    recommendations = cluster_movies.sample(n_recommendations)
    return recommendations[['title_pt', 'sinopse']]

def recommend_movies_method2(user_synopsis, n_recommendations=5):
    user_tfidf = tfidf.transform([user_synopsis])
    predicted_cluster = kmeans_synopses.predict(user_tfidf)[0]
    cluster_movies = df[df['cluster'] == predicted_cluster]
    recommendations = cluster_movies.sample(n_recommendations)
    return recommendations[['title_pt', 'sinopse']]

# Interface do usuário
st.title('Sistema de Recomendação de Filmes')

method = st.radio("Escolha o método de recomendação:", 
                  ('Escolher entre sinopses existentes', 'Escrever sua própria sinopse'))

if method == 'Escolher entre sinopses existentes':
    st.subheader("Sinopses Disponíveis")
    sample_synopses = df['sinopse'].sample(5).tolist()
    chosen_synopsis = st.selectbox("Escolha uma sinopse:", 
                                   options=sample_synopses,
                                   format_func=lambda x: x[:200] + "..." if len(x) > 200 else x)
    
    if st.button('Recomendar Filmes'):
        with st.spinner('Buscando recomendações...'):
            recommendations = recommend_movies_method1(chosen_synopsis)
        
        st.subheader("Filmes Recomendados:")
        for _, movie in recommendations.iterrows():
            st.markdown(f"**{movie['title_pt']}**")
            st.write(movie['sinopse'][:200] + "...")
            st.markdown("---")

else:
    st.subheader("Escreva Sua Própria Sinopse")
    user_synopsis = st.text_area("Digite uma breve descrição do tipo de filme que você gostaria de assistir:")
    
    if st.button('Recomendar Filmes'):
        if user_synopsis:
            with st.spinner('Analisando sua sinopse e buscando recomendações...'):
                recommendations = recommend_movies_method2(user_synopsis)
            
            st.subheader("Filmes Recomendados:")
            for _, movie in recommendations.iterrows():
                st.markdown(f"**{movie['title_pt']}**")
                st.write(movie['sinopse'][:200] + "...")
                st.markdown("---")
        else:
            st.warning("Por favor, escreva uma sinopse antes de pedir recomendações.")

# Adicionar informações sobre o dataset
st.sidebar.title("Informações do Dataset")
st.sidebar.write(f"Total de filmes: {len(df)}")
st.sidebar.write(f"Período: {df['year'].min()} - {df['year'].max()}")

# Mostrar os gêneros mais comuns
genres = df['genre'].str.split(',').explode().str.strip().value_counts()
st.sidebar.write("Gêneros mais comuns:")
st.sidebar.bar_chart(genres.head(10))
