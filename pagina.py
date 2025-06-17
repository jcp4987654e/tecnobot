import streamlit as st
import groq
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import uuid
from datetime import datetime
import requests # Usamos requests para comunicarnos directamente con Firebase

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="Chatbot del Instituto 13 de Julio",
    page_icon="�",
    layout="wide"
)

# --- CONSTANTES ---
MODELO_PREDETERMINADO = "llama3-8b-8192"
SYSTEM_PROMPT = """
Eres TecnoBot, el asistente virtual del Instituto 13 de Julio. Tu función es responder preguntas sobre el instituto basándote EXCLUSIVAMENTE en el CONTEXTO RELEVANTE que se te proporciona. Si la pregunta es personal (ej: 'mis notas'), usa los datos personales del usuario que también se incluyen en el contexto. No puedes usar conocimiento externo. Si no tienes la información, indica amablemente que no puedes responder y sugiere contactar a secretaría. Sé siempre amable y servicial.
"""
CODIGO_SECRETO_PROFESOR = "PROFESOR2025"
CODIGO_SECRETO_AUTORIDAD = "AUTORIDAD2025"
DOMINIO_INSTITUCIONAL = "@13dejulio.edu.ar" # Dominio permitido para el registro

# --- FUNCIONES DE FIREBASE (CON API REST) ---

def firebase_api_auth(endpoint, data):
    """Función central para llamar a la API REST de Firebase Auth."""
    api_key = st.secrets["firebase_config"]["apiKey"]
    url = f"https://identitytoolkit.googleapis.com/v1/{endpoint}?key={api_key}"
    response = requests.post(url, json=data)
    return response.json()

def firebase_db_call(method, path, data=None, token=None):
    """Función para interactuar con Realtime Database."""
    db_url = f"https://{st.secrets['firebase_config']['projectId']}-default-rtdb.firebaseio.com"
    full_path = f"{db_url}/{path}.json"
    
    params = {}
    if token:
        params['auth'] = token

    if method.lower() == 'get':
        response = requests.get(full_path, params=params)
    elif method.lower() == 'put':
        response = requests.put(full_path, json=data, params=params)
    else:
        return None
        
    return response.json() if response.status_code == 200 else None

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
    if datos_usuario and not st.session_state.get('guest_mode', False):
        contexto_privado = "\n--- DATOS PERSONALES DEL USUARIO ---\n" + json.dumps({k: v for k, v in datos_usuario.items() if k != 'chats'})
    return (contexto_publico + contexto_privado) or "No se encontró información relevante."

def generar_respuesta_stream(cliente_groq, historial_chat):
    try:
        stream = cliente_groq.chat.completions.create(model=MODELO_PREDETERMINADO, messages=historial_chat, temperature=0.5, max_tokens=1024, stream=True)
        for chunk in stream: yield chunk.choices[0].delta.content or ""
    except Exception as e:
        st.error(f"Ocurrió un error con el modelo de IA: {e}"); yield ""

def generar_titulo_chat(cliente_groq, primer_mensaje):
    try:
        respuesta = cliente_groq.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "system", "content": "Genera un título muy corto (3-5 palabras) para esta conversación. Responde solo con el título."}, {"role": "user", "content": primer_mensaje}])
        return respuesta.choices[0].message.content.strip().replace('"', '')
    except: return "Nuevo Chat"

# --- LÓGICA DE INTERFAZ (UI) ---

def aplicar_estilos_css():
    st.markdown(f"""
    <style>
        /* --- DEFINICIÓN DE ANIMACIONES --- */
        @keyframes pulse {{ 0%{{box-shadow:0 0 10px #a1c9f4}} 50%{{box-shadow:0 0 25px #a1c9f4}} 100%{{box-shadow:0 0 10px #a1c9f4}} }}
        @keyframes fadeIn {{ from{{opacity:0;transform:translateY(20px)}} to{{opacity:1;transform:translateY(0)}} }}
        @keyframes thinking-pulse {{ 0%{{opacity:0.7}} 50%{{opacity:1}} 100%{{opacity:0.7}} }}
        @keyframes pointAndFade {{
            0% {{ opacity: 0; transform: scale(0.8); }}
            25% {{ opacity: 1; transform: scale(1.1); }}
            75% {{ opacity: 1; transform: scale(1.1); }}
            100% {{ opacity: 0; transform: scale(0.8); }}
        }}

        /* --- ESTILOS GENERALES --- */
        .stApp {{
            background-color:#2d2a4c;
            background-image:repeating-linear-gradient(45deg,rgba(255,255,255,0.03) 1px,transparent 1px,transparent 20px),repeating-linear-gradient(-45deg,rgba(161,201,244,0.05) 1px,transparent 1px,transparent 20px),linear-gradient(180deg,#2d2a4c 0%,#4f4a7d 100%);
        }}
        .main > div:first-child {{ padding-top: 0rem; }}
        header, [data-testid="stToolbar"] {{ display: none !important; }}

        /* --- PANTALLA DE BIENVENIDA --- */
        .splash-container {{ display:flex; flex-direction:column; justify-content:center; align-items:center; position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:9999; animation:fadeIn 1.5s ease-in-out; }}
        .splash-logo {{ width:180px; height:180px; border-radius:50%; margin-bottom:2rem; animation:pulse 3s infinite; }}
        .splash-title {{ font-size:2.5rem; color:#e6e6fa; text-shadow:0 0 10px rgba(161,201,244,0.7); text-align: center; padding: 0 1rem;}}

        /* --- APP PRINCIPAL --- */
        .main-container {{ animation:fadeIn 0.8s ease-in-out; max-width:900px; margin:auto; padding:2rem 1rem; }}
        .login-container {{ max-width: 450px; margin: auto; padding-top: 5rem; }}
        [data-testid="stSidebar"] {{ border-right:2px solid #a1c9f4; background-color:#2d2a4c; }}
        .sidebar-logo {{ width:120px; height:120px; border-radius:50%; border:3px solid #a1c9f4; display:block; margin:2rem auto; animation:pulse 4s infinite ease-in-out; }}
        h1 {{ color:#e6e6fa; text-shadow:0 0 8px rgba(161,201,244,0.7); text-align:center; }}
        .chat-wrapper {{ border:2px solid #4f4a7d; box-shadow:0 0 20px -5px #a1c9f4; border-radius:20px; background-color:rgba(45,42,76,0.8); padding:1rem; margin-top:1rem; }}
        [data-testid="stChatMessage"] {{ animation:fadeIn 0.4s ease-out; }}
        .thinking-indicator {{ font-style:italic; color:rgba(230,230,250,0.8); animation:thinking-pulse 1.5s infinite; }}
        .stButton>button {{ width: 100%; margin-bottom: 5px; }}

        /* --- NUEVA ANIMACIÓN PARA SIDEBAR EN MÓVILES --- */
        .sidebar-pointer {{
            display: none; /* Oculto por defecto en PC */
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 10001; /* Por encima de todo */
            color: white;
            font-size: 2.5rem;
            text-shadow: 0 0 8px #a1c9f4;
            animation: pointAndFade 1.5s ease-in-out forwards; /* La animación que pediste */
        }}
        
        /* --- DISEÑO RESPONSIVO (CELULARES) --- */
        @media (max-width: 768px) {{
            .main-container, .login-container {{ padding: 1rem 0.5rem !important; }}
            .splash-logo {{ width: 120px; height: 120px; }}
            .splash-title {{ font-size: 1.5rem; text-align: center; }}
            .chat-wrapper {{ margin-top: 0.5rem; padding: 0.5rem; }}
            .st-emotion-cache-1fplz1o {{ height: 65vh !important; }}
            h1 {{ font-size: 1.8rem; padding-top: 0; }}
            .sidebar-logo {{ width: 80px; height: 80px; }}
            .sidebar-pointer {{ display: block; }} /* Solo se muestra en móviles */
        }}
    </style>
    """, unsafe_allow_html=True)

def render_login_page():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title("Bienvenido a TecnoBot")
    if st.button("Ingresar como Invitado", use_container_width=True):
        st.session_state.logged_in = True; st.session_state.guest_mode = True
        st.session_state.user_data = {"nombre": "Invitado", "rol": "invitado", "chats": {}}
        st.rerun()
    st.markdown("---")
    login_tab, register_tab = st.tabs(["Iniciar Sesión", "Registrarse"])
    with login_tab:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Contraseña", type="password", key="login_pass")
        if st.button("Ingresar", key="login_button", use_container_width=True):
            response = firebase_api_auth("accounts:signInWithPassword", {"email": email, "password": password, "returnSecureToken": True})
            if "localId" in response:
                st.session_state.logged_in = True; st.session_state.user_token = response['idToken']
                st.session_state.user_uid = response['localId']; st.session_state.guest_mode = False
                st.rerun()
            else: st.error("Email o contraseña incorrectos.")
    with register_tab:
        st.subheader("Crear una Cuenta")
        nombre = st.text_input("Nombre", key="reg_nombre")
        apellido = st.text_input("Apellido", key="reg_apellido")
        reg_email = st.text_input("Email Institucional", key="reg_email", help=f"Debe ser una cuenta con dominio {DOMINIO_INSTITUCIONAL}")
        reg_password = st.text_input("Contraseña", type="password", key="reg_pass")
        rol_code = st.text_input("Código de Rol (dejar en blanco si eres alumno)", type="password", key="reg_code")
        if st.button("Registrarse", key="reg_button", use_container_width=True):
            if not reg_email.endswith(DOMINIO_INSTITUCIONAL):
                st.error(f"Registro no permitido. Debes usar un correo institucional."); return
            if not all([nombre, apellido, reg_email, reg_password]):
                st.warning("Por favor, completa todos los campos obligatorios."); return
            rol, coleccion = ("profesor", "profesores") if rol_code == CODIGO_SECRETO_PROFESOR else \
                             ("autoridad", "autoridades") if rol_code == CODIGO_SECRETO_AUTORIDAD else ("alumno", "alumnos")
            response = firebase_api_auth("accounts:signUp", {"email": reg_email, "password": reg_password, "returnSecureToken": True})
            if "localId" in response:
                uid, id_token = response['localId'], response['idToken']
                legajo = str(int(time.time() * 100))[-6:]
                datos_usuario = {"nombre": nombre, "apellido": apellido, "email": reg_email, "rol": rol, "legajo": legajo}
                write_response = firebase_db_call('put', f"{coleccion}/{uid}", datos_usuario, id_token)
                if write_response is not None:
                    st.success(f"✅ ¡Registro exitoso! Tu N° de legajo es {legajo}."); st.balloons(); time.sleep(4); st.rerun()
                else: st.error("Error: Tu cuenta fue creada, pero no se pudo guardar tu perfil.")
            else: st.error("No se pudo registrar. El email ya podría estar en uso.")
    st.markdown('</div>', unsafe_allow_html=True)

def render_chat_ui(cliente_groq, modelo_embeddings, documentos_planos, indice_embeddings):
    LOGO_URL = "https://13dejulio.edu.ar/wp-content/uploads/2022/03/Isologotipo-13-de-Julio-400.png"
    # Lógica para mostrar la animación del sidebar solo una vez
    if 'sidebar_hint_shown' not in st.session_state:
        st.markdown('<div class="sidebar-pointer">➔</div>', unsafe_allow_html=True)
        st.session_state.sidebar_hint_shown = True

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f'<img src="{LOGO_URL}" class="sidebar-logo">', unsafe_allow_html=True)
        user_data = st.session_state.user_data
        st.write(f"Bienvenido, {user_data.get('nombre', 'Usuario')}")
        if st.button("➕ Nuevo Chat", use_container_width=True): start_new_chat()
        st.markdown("---"); st.subheader("Chats Recientes")
        if st.session_state.chat_history:
            sorted_chats = sorted(st.session_state.chat_history.items(), key=lambda item: item[1]['timestamp'], reverse=True)
            for chat_id, chat_data in sorted_chats[:5]:
                if st.button(chat_data.get('titulo', 'Chat'), key=chat_id, use_container_width=True):
                    st.session_state.active_chat_id = chat_id; st.rerun()
        else: st.write("No hay chats recientes.")
        st.markdown("---")
        if st.button("Cerrar Sesión", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
    st.title("🎓 Chatbot del Instituto 13 de Julio")
    active_chat = st.session_state.chat_history.get(st.session_state.active_chat_id, {"mensajes": []})
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    chat_container = st.container()
    with chat_container:
        for msg in active_chat["mensajes"]:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑‍💻"):
                st.markdown(msg["content"], unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        active_chat["mensajes"].append({"role": "user", "content": prompt})
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
                response_stream = generar_respuesta_stream(cliente_groq, historial_para_api)
                full_response = placeholder.write_stream(response_stream)
        active_chat["mensajes"].append({"role": "assistant", "content": full_response})
        if len(active_chat["mensajes"]) == 2: active_chat["titulo"] = generar_titulo_chat(cliente_groq, prompt)
        if not st.session_state.get('guest_mode', False):
            firebase_db_call('put', f"{user_data['rol'] + 's'}/{st.session_state.user_uid}/chats/{st.session_state.active_chat_id}", active_chat, st.session_state.user_token)
        st.rerun()

def start_new_chat():
    new_chat_id = str(uuid.uuid4())
    st.session_state.active_chat_id = new_chat_id
    st.session_state.chat_history[new_chat_id] = {"titulo": "Nuevo Chat", "timestamp": datetime.utcnow().isoformat(), "mensajes": []}
    st.rerun()

# --- FLUJO PRINCIPAL DE LA APLICACIÓN ---
def main():
    aplicar_estilos_css()
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if not st.session_state.logged_in:
        render_login_page()
    else:
        if 'recursos_cargados' not in st.session_state:
            with st.spinner("Cargando tu sesión y el motor de IA..."):
                if not st.session_state.get('guest_mode', False):
                    user_uid, user_token = st.session_state.user_uid, st.session_state.user_token
                    user_data = None
                    for coleccion in ["alumnos", "profesores", "autoridades"]:
                        data = firebase_db_call('get', f"{coleccion}/{user_uid}", token=user_token)
                        if data: user_data = data; break
                    if user_data is None:
                        st.error("Error: No se encontró tu perfil."); st.session_state.logged_in = False; time.sleep(5); st.rerun(); st.stop()
                    st.session_state.user_data = user_data
                    st.session_state.chat_history = user_data.get("chats", {})
                else:
                    st.session_state.user_data = {"nombre": "Invitado", "rol": "invitado"}; st.session_state.chat_history = {}
                
                modelo_embeddings, documentos_planos, indice_embeddings = cargar_recursos_ia()
                if modelo_embeddings and documentos_planos and indice_embeddings is not None:
                    st.session_state.modelo_embeddings, st.session_state.documentos_planos, st.session_state.indice_embeddings = modelo_embeddings, documentos_planos, indice_embeddings
                    st.session_state.recursos_cargados = True
                else: st.error("No se pudieron cargar los recursos de IA."); st.stop()
                
                if not st.session_state.chat_history: start_new_chat()
                else: st.session_state.active_chat_id = max(st.session_state.chat_history.items(), key=lambda item: item[1]['timestamp'])[0]
                st.rerun()
                
        cliente_groq = groq.Groq(api_key=st.secrets["GROQ_API_KEY"])
        render_chat_ui(cliente_groq, st.session_state.modelo_embeddings, st.session_state.documentos_planos, st.session_state.indice_embeddings)

if __name__ == "__main__":
    main()
