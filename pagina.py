import streamlit as st
import groq
import json
import requests
import time
import uuid
import numpy as np
import os
import random
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------------
#  CONFIGURACIÓN INICIAL DE LA PÁGINA
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="TecnoBot – Instituto 13 de Julio", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="auto" # Muestra la barra lateral abierta en escritorio y cerrada en móvil.
)

# ----------------------------------------------------------------------------------
#  LEER SECRETS Y CONFIGURAR CONSTANTES GLOBALES
# ----------------------------------------------------------------------------------

# 1. Leemos las claves de API desde los "Secrets" de Hugging Face.
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID")
FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# 2. Validamos que todas las claves necesarias estén presentes.
if not all([FIREBASE_PROJECT_ID, FIREBASE_API_KEY, GROQ_API_KEY]):
    st.error("Error: Faltan una o más claves de API en la configuración de Secrets. Revisa los nombres y valores y reinicia el Space.")
    st.stop()
