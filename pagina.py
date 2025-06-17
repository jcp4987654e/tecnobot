import streamlit as st
import groq
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import pyrebase
import uuid
from datetime import datetime

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="Chatbot del Instituto 13 de Julio",
    page_icon="🎓",
    layout="wide"
)

# --- CONSTANTES ---
MODELOS = ["llama3-8b-8192", "llama3-70b-8192"]
SYSTEM_PROMPT = """
Eres TecnoBot, el asistente virtual del Instituto 13 de Julio. Tu función es responder preguntas sobre el instituto basándote EXCLUSIVAMENTE en el CONTEXTO RELEVANTE que se te proporciona. Si la pregunta es personal (ej: 'mis notas'), usa los datos personales del usuario que también se incluyen en el contexto. No puedes usar conocimiento externo. Si no tienes la información, indica amablemente que no puedes responder y sugiere contactar a secretaría. Sé siempre amable y servicial.
"""
CODIGO_SECRETO_PROFESOR = "PROFESOR2025"
CODIGO_SECRETO_AUTORIDAD = "AUTORIDAD2025"

# --- FUNCIONES DE CONEXIÓN A FIREBASE ---

@st.cache_resource
def inicializar_firebase():
    """Inicializa la conexión con Firebase usando los secrets."""
    try:
        firebase_config = dict(st.secrets["firebase_config"])
        firebase = pyrebase.initialize_app(firebase_config)
        return firebase.auth(), firebase.database()
    except Exception as e:
        st.error(f"Error al conectar con Firebase. Detalle: {e}")
        return None, None

# --- FUNCIONES DE LÓGICA DE IA (CACHEADAS) ---

@st.cache_data
def cargar_base_de_conocimiento(ruta_archivo='conocimiento.json'):
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

@st.cache_data
def aplanar_conocimiento(_base_de_conocimiento):
    documentos = []
    if not _base_de_conocimiento: return documentos
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
    """Carga todos los modelos y datos pesados una sola vez."""
    try:
        modelo = SentenceTransformer('all-MiniLM-L6-v2')
        base_de_conocimiento = cargar_base_de_conocimiento()
        documentos = aplanar_conocimiento(base_de_conocimiento)
        if not documentos: return None, None, None
        indice = modelo.encode(documentos)
        return modelo, documentos, indice
    except Exception as e:
        st.error(f"Error crítico al cargar recursos de IA: {e}")
        return None, None, None

def buscar_contexto(query, _modelo, documentos, embeddings_corpus, datos_usuario):
    """Busca contexto público y añade el contexto privado del usuario."""
    contexto_publico = ""
    if embeddings_corpus is not None and hasattr(_modelo, 'encode'):
        embedding_consulta = _modelo.encode([query])
        similitudes = cosine_similarity(embedding_consulta, embeddings_corpus)[0]
        indices_similares = np.argsort(similitudes)[::-1]
        textos_ya_anadidos = set()
        for idx in indices_similares:
            if similitudes[idx] > 0.4 and len(textos_ya_anadidos) < 3:
                texto = documentos[idx]
                if texto not in textos_ya_anadidos:
                    contexto_publico += f"- {texto}\n"
                    textos_ya_anadidos.add(texto)
    contexto_privado = ""
    if datos_usuario:
        contexto_privado = "\n--- DATOS PERSONALES DEL USUARIO ---\n" + json.dumps({k: v for k, v in datos_usuario.items() if k != 'chats'})
    return (contexto_publico + contexto_privado) or "No se encontró información relevante."

def generar_respuesta_stream(cliente_groq, modelo_seleccionado, historial_chat):
    """Genera una respuesta de la IA en tiempo real."""
    try:
        stream = cliente_groq.chat.completions.create(model=modelo_seleccionado, messages=historial_chat, temperature=0.5, max_tokens=1024, stream=True)
        for chunk in stream: yield chunk.choices[0].delta.content or ""
    except Exception as e:
        st.error(f"Ocurrió un error con el modelo de IA: {e}")
        yield ""

def generar_titulo_chat(cliente_groq, primer_mensaje):
    """Genera un título corto para la conversación."""
    try:
        respuesta = cliente_groq.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "Genera un título muy corto (3-5 palabras) para una conversación que empieza con este mensaje. Responde solo con el título."},
                {"role": "user", "content": primer_mensaje}
            ]
        )
        return respuesta.choices[0].message.content.strip().replace('"', '')
    except:
        return "Nuevo Chat"


# --- LÓGICA DE INTERFAZ (UI) ---

def aplicar_estilos_css():
    st.markdown("""
    <style>
        /* ... (Todo el código CSS de animaciones y diseño va aquí, sin cambios) ... */
        @keyframes pulse { 0%{box-shadow:0 0 10px #a1c9f4} 50%{box-shadow:0 0 25px #a1c9f4} 100%{box-shadow:0 0 10px #a1c9f4} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
        @keyframes thinking-pulse { 0%{opacity:0.7} 50%{opacity:1} 100%{opacity:0.7} }
        .stApp { background-color:#2d2a4c; background-image:repeating-linear-gradient(45deg,rgba(255,255,255,0.03) 1px,transparent 1px,transparent 20px),repeating-linear-gradient(-45deg,rgba(161,201,244,0.05) 1px,transparent 1px,transparent 20px),linear-gradient(180deg,#2d2a4c 0%,#4f4a7d 100%); }
        .main > div:first-child { padding-top: 0; }
        header, [data-testid="stToolbar"] { display: none !important; }
        .main-container { animation:fadeIn 0.8s ease-in-out; max-width:900px; margin:auto; padding:2rem 1rem; }
        [data-testid="stSidebar"] { border-right:2px solid #a1c9f4; background-color:#2d2a4c; }
        .sidebar-logo { width:120px; height:120px; border-radius:50%; border:3px solid #a1c9f4; display:block; margin:2rem auto; animation:pulse 4s infinite ease-in-out; }
        h1 { color:#e6e6fa; text-shadow:0 0 8px rgba(161,201,244,0.7); text-align:center; }
        .chat-wrapper { border:2px solid #4f4a7d; box-shadow:0 0 20px -5px #a1c9f4; border-radius:20px; background-color:rgba(45,42,76,0.8); padding:1rem; margin-top:1rem; }
        [data-testid="stChatMessage"] { animation:fadeIn 0.4s ease-out; }
        .thinking-indicator { font-style:italic; color:rgba(230,230,250,0.8); animation:thinking-pulse 1.5s infinite; }
        .stButton>button { width: 100%; margin-bottom: 5px; } /* Estilo para botones de historial */
        @media (max-width:768px) { .main-container{padding-left:1rem!important;padding-right:1rem!important} h1{font-size:1.8rem} .sidebar-logo{width:80px;height:80px} }
    </style>
    """, unsafe_allow_html=True)

def render_login_page(auth, db):
    st.title("Bienvenido a TecnoBot")
    login_tab, register_tab = st.tabs(["Iniciar Sesión", "Registrarse"])

    with login_tab:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Contraseña", type="password", key="login_pass")
        if st.button("Ingresar", key="login_button"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state.logged_in = True
                st.session_state.user_uid = user['localId']
                st.rerun()
            except Exception: st.error("Email o contraseña incorrectos.")

    with register_tab:
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Contraseña", type="password", key="reg_pass")
        nombre = st.text_input("Nombre", key="reg_nombre")
        apellido = st.text_input("Apellido", key="reg_apellido")
        legajo = st.text_input("N° de Legajo", key="reg_legajo", help="Obligatorio para alumnos.")
        rol_code = st.text_input("Código de Rol (dejar en blanco si eres alumno)", type="password", key="reg_code")
        
        if st.button("Registrarse", key="reg_button"):
            rol, coleccion = ("profesor", "profesores") if rol_code == CODIGO_SECRETO_PROFESOR else \
                             ("autoridad", "autoridades") if rol_code == CODIGO_SECRETO_AUTORIDAD else ("alumno", "alumnos")
            try:
                user = auth.create_user_with_email_and_password(reg_email, reg_password)
                db.child(coleccion).child(user['localId']).set({"nombre": nombre, "apellido": apellido, "email": reg_email, "rol": rol, "legajo": legajo})
                st.success("¡Registro exitoso! Ahora puedes iniciar sesión."); time.sleep(2); st.rerun()
            except: st.error("No se pudo completar el registro. El email ya podría estar en uso.")

def render_chat_ui(cliente_groq, auth, db, modelo_embeddings, documentos_planos, indice_embeddings):
    LOGO_URL = "https://i.imgur.com/gJ5Ym2W.png"
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f'<img src="{LOGO_URL}" class="sidebar-logo">', unsafe_allow_html=True)
        user_data = st.session_state.user_data
        st.write(f"Bienvenido, {user_data.get('nombre', 'Usuario')}")
        
        if st.button("➕ Nuevo Chat", use_container_width=True):
            start_new_chat()
        
        st.markdown("---")
        st.subheader("Chats Recientes")
        if st.session_state.chat_history:
            # Ordenar chats por timestamp, del más reciente al más antiguo
            sorted_chats = sorted(st.session_state.chat_history.items(), key=lambda item: item[1]['timestamp'], reverse=True)
            for chat_id, chat_data in sorted_chats[:5]: # Mostrar solo los últimos 5
                if st.button(chat_data.get('titulo', 'Chat'), key=chat_id, use_container_width=True):
                    st.session_state.active_chat_id = chat_id
                    st.rerun()
        else:
            st.write("No hay chats recientes.")

        st.markdown("---")
        modelo_seleccionado = st.selectbox("Elige tu modelo de IA:", MODELOS, index=1)
        if st.button("Cerrar Sesión", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    st.title("🎓 Chatbot del Instituto 13 de Julio")
    
    # Renderizar el chat activo
    active_chat = st.session_state.chat_history.get(st.session_state.active_chat_id, {"mensajes": []})
    
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    chat_container = st.container(height=500)
    with chat_container:
        for msg in active_chat["mensajes"]:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑‍💻"):
                st.markdown(msg["content"], unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        active_chat["mensajes"].append({"role": "user", "content": prompt})
        
        # Generar respuesta y actualizar la UI en tiempo real
        with chat_container:
            for msg in active_chat["mensajes"]:
                with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑‍💻"):
                    st.markdown(msg["content"], unsafe_allow_html=True)
            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                placeholder.markdown('<p class="thinking-indicator">Tirando magia...</p>', unsafe_allow_html=True)
                time.sleep(2.5)
                contexto_rag = buscar_contexto(prompt, modelo_embeddings, documentos_planos, indice_embeddings, user_data)
                historial_para_api = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{contexto_rag}"}] + active_chat["mensajes"][-10:]
                response_stream = generar_respuesta_stream(cliente_groq, modelo_seleccionado, historial_para_api)
                full_response = placeholder.write_stream(response_stream)
        
        active_chat["mensajes"].append({"role": "assistant", "content": full_response})
        
        # Generar título si es un chat nuevo
        if len(active_chat["mensajes"]) == 2:
            active_chat["titulo"] = generar_titulo_chat(cliente_groq, prompt)

        # Guardar la conversación actualizada en Firebase
        db.child(user_data['rol'] + 's').child(st.session_state.user_uid).child("chats").child(st.session_state.active_chat_id).set(active_chat)
        st.rerun()

def start_new_chat():
    """Crea una nueva sesión de chat."""
    new_chat_id = str(uuid.uuid4())
    st.session_state.active_chat_id = new_chat_id
    st.session_state.chat_history[new_chat_id] = {
        "titulo": "Nuevo Chat",
        "timestamp": datetime.utcnow().isoformat(),
        "mensajes": []
    }
    st.rerun()

# --- FLUJO PRINCIPAL DE LA APLICACIÓN ---
def main():
    aplicar_estilos_css()
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    auth, db = inicializar_firebase()
    if not auth or not db:
        st.stop()

    if not st.session_state.logged_in:
        render_login_page(auth, db)
    else:
        # Cargar datos del usuario y de la IA una sola vez después del login
        if 'recursos_cargados' not in st.session_state:
            with st.spinner("Cargando tu sesión y el motor de IA..."):
                # Cargar datos del usuario
                user_uid = st.session_state.user_uid
                user_data = None
                for coleccion in ["alumnos", "profesores", "autoridades"]:
                    data = db.child(coleccion).child(user_uid).get().val()
                    if data:
                        user_data = data
                        break
                st.session_state.user_data = user_data
                
                # Cargar historial de chats
                st.session_state.chat_history = user_data.get("chats", {})
                
                # Cargar modelos de IA
                modelo_embeddings, documentos_planos, indice_embeddings = cargar_recursos_ia()
                if modelo_embeddings and documentos_planos and indice_embeddings is not None:
                    st.session_state.modelo_embeddings, st.session_state.documentos_planos, st.session_state.indice_embeddings = modelo_embeddings, documentos_planos, indice_embeddings
                    st.session_state.recursos_cargados = True
                else:
                    st.error("No se pudieron cargar los recursos de IA."); st.stop()

                # Seleccionar chat activo o crear uno nuevo
                if not st.session_state.chat_history:
                    start_new_chat()
                else:
                    # Seleccionar el chat más reciente como activo
                    latest_chat = max(st.session_state.chat_history.items(), key=lambda item: item[1]['timestamp'])
                    st.session_state.active_chat_id = latest_chat[0]
                
                st.rerun()
        
        cliente_groq = groq.Groq(api_key=st.secrets["GROQ_API_KEY"])
        render_chat_ui(cliente_groq, auth, db, st.session_state.modelo_embeddings, st.session_state.documentos_planos, st.session_state.indice_embeddings)

if __name__ == "__main__":
    main()
