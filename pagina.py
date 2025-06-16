import streamlit as st
import groq
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Chatbot del Instituto 13 de Julio",
    page_icon="🎓",
    layout="wide"
)

# --- CONSTANTES Y CONFIGURACIÓN INICIAL ---
# Se ha eliminado el modelo Mixtral de la lista.
MODELOS = ["llama3-8b-8192", "llama3-70b-8192"]
SYSTEM_PROMPT = """
Eres un asistente virtual experto del "Instituto 13 de Julio" llamado "TecnoBot".
Tu única función es responder preguntas sobre el instituto, basándote EXCLUSIVAMENTE en la información proporcionada en el CONTEXTO RELEVANTE.
NO PUEDES usar conocimiento externo o buscar en la web. Tu única fuente de verdad es el contexto.
Si la pregunta del usuario no se puede responder con el contexto, DEBES decir amablemente: "No tengo información sobre ese tema. Mi conocimiento se limita a los datos del instituto. Te sugiero reformular tu pregunta o contactar a secretaría."
No inventes nada. Sé amable, servicial y preséntate como "TecnoBot" en tu primer saludo.
"""

# --- FUNCIONES DE LÓGICA (CACHEADAS PARA EFICIENCIA) ---

@st.cache_data
def cargar_base_de_conocimiento(ruta_archivo='conocimiento.json'):
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

@st.cache_data
def aplanar_conocimiento(_base_de_conocimiento):
    documentos = []
    if _base_de_conocimiento is None: return documentos
    for topic, data in _base_de_conocimiento.items():
        if topic == "material_academico": continue
        if isinstance(data, dict) and 'content' in data:
            documentos.append(f"Información sobre {topic.replace('_', ' ').title()}: {data['content']}")
    if "material_academico" in _base_de_conocimiento:
        for year, subjects in _base_de_conocimiento["material_academico"].items():
            for subject_name, subject_data in subjects.items():
                if isinstance(subject_data, dict):
                    info = f"Materia: {subject_name.replace('_', ' ').title()} de {year.replace('_', ' ')}. {subject_data.get('content', '')} Profesor/a: {subject_data.get('profesor', 'No asignado')}."
                    if subject_data.get('evaluaciones'):
                        info += " Próximas Evaluaciones: " + "".join([f"Fecha: {e.get('fecha', 'N/A')}, Temas: {e.get('temas', 'N/A')}. " for e in subject_data['evaluaciones']])
                    documentos.append(info.strip())
    return [doc for doc in documentos if doc]

@st.cache_resource
def cargar_recursos_ia():
    try:
        modelo = SentenceTransformer('all-MiniLM-L6-v2')
        base_de_conocimiento = cargar_base_de_conocimiento()
        documentos = aplanar_conocimiento(base_de_conocimiento)
        if not documentos:
            return None, None, None
        indice = modelo.encode(documentos)
        return modelo, documentos, indice
    except Exception as e:
        st.error(f"Error crítico al cargar recursos de IA: {e}")
        return None, None, None

def buscar_contexto_semantico(query, _modelo, documentos, embeddings_corpus, top_k=3, umbral=0.4):
    if embeddings_corpus is None or not hasattr(_modelo, 'encode'): return "La base de conocimientos no está lista."
    embedding_consulta = _modelo.encode([query])
    similitudes = cosine_similarity(embedding_consulta, embeddings_corpus)[0]
    indices_similares = np.argsort(similitudes)[::-1]
    contexto_encontrado = ""
    textos_ya_anadidos = set()
    for idx in indices_similares:
        if similitudes[idx] > umbral and len(textos_ya_anadidos) < top_k:
            texto = documentos[idx]
            if texto not in textos_ya_anadidos:
                contexto_encontrado += f"- {texto}\n"
                textos_ya_anadidos.add(texto)
    return contexto_encontrado or "No se encontró información relevante para tu consulta."

def generar_respuesta_stream(cliente_groq, modelo_seleccionado, historial_chat):
    try:
        stream = cliente_groq.chat.completions.create(model=modelo_seleccionado, messages=historial_chat, temperature=0.5, max_tokens=1024, stream=True)
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""
    except Exception as e:
        st.error(f"Ocurrió un error al contactar la API de Groq: {e}")
        yield ""

# --- INTERFAZ DE USUARIO Y RENDERIZADO ---

def aplicar_estilos_css():
    st.markdown(f"""
    <style>
        /* --- DEFINICIÓN DE ANIMACIONES --- */
        @keyframes pulse {{ 0% {{ box-shadow: 0 0 10px #a1c9f4; }} 50% {{ box-shadow: 0 0 25px #a1c9f4; }} 100% {{ box-shadow: 0 0 10px #a1c9f4; }} }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        @keyframes thinking-pulse {{ 0% {{ opacity: 0.7; }} 50% {{ opacity: 1; }} 100% {{ opacity: 0.7; }} }}

        /* --- ESTILOS GENERALES --- */
        .stApp {{
            background-color: #2d2a4c;
            background-image: repeating-linear-gradient(45deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px, transparent 20px), repeating-linear-gradient(-45deg, rgba(161, 201, 244, 0.05) 1px, transparent 1px, transparent 20px), linear-gradient(180deg, #2d2a4c 0%, #4f4a7d 100%);
        }}
        /* Ocultar elementos de Streamlit que no queremos */
        .main > div:first-child {{ padding-top: 0; }}
        header {{ display: none !important; }}
        [data-testid="stToolbar"] {{ display: none !important; }}

        /* --- PANTALLA DE BIENVENIDA --- */
        .splash-container {{
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            position: fixed; top: 0; left: 0;
            width: 100vw; height: 100vh;
            z-index: 9999; animation: fadeIn 1.5s ease-in-out;
        }}
        .splash-logo {{ width: 180px; height: 180px; border-radius: 50%; margin-bottom: 2rem; animation: pulse 3s infinite; }}
        .splash-title {{ font-size: 2.5rem; color: #e6e6fa; text-shadow: 0 0 10px rgba(161, 201, 244, 0.7); }}

        /* --- APP PRINCIPAL --- */
        .main-container {{
            animation: fadeIn 0.8s ease-in-out;
            max-width: 900px; margin: auto; padding-top: 2rem;
        }}
        [data-testid="stSidebar"] {{ border-right: 2px solid #a1c9f4; background-color: #2d2a4c; }}
        .sidebar-logo {{ width: 120px; height: 120px; border-radius: 50%; border: 3px solid #a1c9f4; display: block; margin: 2rem auto; animation: pulse 4s infinite ease-in-out; }}
        h1 {{ color: #e6e6fa; text-shadow: 0 0 8px rgba(161, 201, 244, 0.7); text-align: center; }}
        .chat-wrapper {{ border: 2px solid #4f4a7d; box-shadow: 0 0 20px -5px #a1c9f4; border-radius: 20px; background-color: rgba(45, 42, 76, 0.8); padding: 1rem; margin-top: 1rem; }}
        [data-testid="stChatMessage"] {{ animation: fadeIn 0.4s ease-out; }}
        [data-testid="stChatMessage"][data-testid-stream-message-type="assistant"] {{ background-color: #4f4a7d; border: 1px solid #a1c9f4; }}
        [data-testid="stChatMessage"][data-testid-stream-message-type="user"] {{ background-color: #3b3861; }}
        [data-testid="stChatInput"] {{ background-color: transparent; border-top: 2px solid #a1c9f4; padding-top: 1rem; }}
        .thinking-indicator {{ font-style: italic; color: rgba(230, 230, 250, 0.8); animation: thinking-pulse 1.5s infinite; }}

        /* --- DISEÑO RESPONSIVO (CELULARES) --- */
        @media (max-width: 768px) {{
            .main-container {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
            .splash-logo {{ width: 120px; height: 120px; }}
            .splash-title {{ font-size: 1.8rem; text-align: center; }}
            .chat-wrapper {{ margin-top: 0.5rem; padding: 0.5rem; }}
            h1 {{ font-size: 1.8rem; padding-top: 0; }}
            .sidebar-logo {{ width: 80px; height: 80px; }}
        }}
    </style>
    """, unsafe_allow_html=True)

def render_chat_ui(cliente_groq, modelo_embeddings, documentos_planos, indice_embeddings):
    # LOGO: Se ha actualizado con tu URL.
    # NOTA: Si no carga, es probable que el sitio web del instituto bloquee la carga de imágenes desde otros dominios.
    # La solución más segura es subir el logo a un servicio como imgur.com y usar ese nuevo enlace.
    LOGO_URL = "https://13dejulio.edu.ar/wp-content/uploads/2022/03/Isologotipo-13-de-Julio-400.png"
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f'<img src="{LOGO_URL}" class="sidebar-logo">', unsafe_allow_html=True)
        st.header("Configuración")
        modelo_seleccionado = st.selectbox("Elige tu modelo de IA:", MODELOS, index=1)
        st.info("Este chatbot entiende el significado, no solo las palabras.")
    st.title("🎓 Chatbot del Instituto 13 de Julio")
    
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = [{"role": "assistant", "content": "¡Hola! Soy TecnoBot. ¿En qué puedo ayudarte?"}]
        
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    chat_container = st.container(height=500)
    # El historial de chat ahora se renderiza después de procesar el nuevo mensaje para evitar el "salto".
    # El mensaje del usuario y la respuesta del bot se manejan en el bloque `if prompt_usuario`.
    st.markdown('</div>', unsafe_allow_html=True)

    if prompt_usuario := st.chat_input("Escribe tu pregunta aquí..."):
        # Añade y muestra el mensaje del usuario inmediatamente.
        st.session_state.mensajes.append({"role": "user", "content": prompt_usuario})
        with chat_container:
            for mensaje in st.session_state.mensajes:
                with st.chat_message(mensaje["role"], avatar="🤖" if mensaje["role"] == "assistant" else "🧑‍💻"):
                    st.markdown(mensaje["content"], unsafe_allow_html=True)
        
        # Muestra el indicador de carga y genera la respuesta.
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown('<p class="thinking-indicator">Tirando magia...</p>', unsafe_allow_html=True)
            
            # Se añade un tiempo de espera fijo de 3 segundos para el indicador.
            time.sleep(3)
            
            contexto_rag = buscar_contexto_semantico(prompt_usuario, modelo_embeddings, documentos_planos, indice_embeddings)
            system_prompt_con_contexto = f"{SYSTEM_PROMPT}\n\nCONTEXTO RELEVANTE:\n{contexto_rag}"
            
            historial_para_api = [{"role": "system", "content": system_prompt_con_contexto}]
            mensajes_relevantes = [msg for msg in st.session_state.mensajes if msg['role'] != 'system']
            historial_para_api.extend(mensajes_relevantes[-10:])
            
            response_stream = generar_respuesta_stream(cliente_groq, modelo_seleccionado, historial_para_api)
            full_response = placeholder.write_stream(response_stream)
            
        st.session_state.mensajes.append({"role": "assistant", "content": full_response})
        st.rerun()

    # Si no hay nuevo prompt, solo renderiza el historial existente.
    else:
        with chat_container:
            for mensaje in st.session_state.mensajes:
                with st.chat_message(mensaje["role"], avatar="🤖" if mensaje["role"] == "assistant" else "🧑‍💻"):
                    st.markdown(mensaje["content"], unsafe_allow_html=True)


def main():
    aplicar_estilos_css()
    if 'app_ready' not in st.session_state:
        st.session_state.app_ready = False

    # Intenta cargar los recursos
    if not st.session_state.get('recursos_cargados', False):
        modelo_embeddings, documentos_planos, indice_embeddings = cargar_recursos_ia()
        if modelo_embeddings is not None and documentos_planos and indice_embeddings is not None:
            st.session_state.modelo_embeddings = modelo_embeddings
            st.session_state.documentos_planos = documentos_planos
            st.session_state.indice_embeddings = indice_embeddings
            st.session_state.recursos_cargados = True
        else:
            st.error("No se pudieron cargar los recursos de IA. La base de conocimientos podría estar vacía o hubo un error de conexión.")
            st.stop()

    # Muestra la splash screen y luego transiciona
    if not st.session_state.app_ready:
        LOGO_URL = "https://13dejulio.edu.ar/wp-content/uploads/2022/03/Isologotipo-13-de-Julio-400.png"
        st.markdown(f'<div class="splash-container"><img src="{LOGO_URL}" class="splash-logo"><h1 class="splash-title">Bienvenido a TecnoBot</h1></div>', unsafe_allow_html=True)
        time.sleep(3)
        st.session_state.app_ready = True
        st.rerun()
    else:
        try:
            cliente_groq = groq.Groq(api_key=st.secrets["GROQ_API_KEY"])
            render_chat_ui(cliente_groq, st.session_state.modelo_embeddings, st.session_state.documentos_planos, st.session_state.indice_embeddings)
        except Exception as e:
            st.error(f"No se pudo inicializar el cliente de la API: {e}")

if __name__ == "__main__":
    main()
