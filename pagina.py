import streamlit as st
import groq, json, requests, time, uuid, numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------------
#  CONFIGURACIÓN INICIAL
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="TecnoBot – Instituto 13 de Julio", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------------------
#  CONSTANTES Y CONFIGURACIÓN
# ----------------------------------------------------------------------------------
MODELO_PREDETERMINADO = "llama3-8b-8192"
SYSTEM_PROMPT = """
Eres TecnoBot, el asistente virtual del Instituto 13 de Julio. Responde ÚNICAMENTE con el CONTEXTO proporcionado
(público + datos personales). Si la respuesta no está, indica que no la tienes y sugiere contactar a secretaría.
Sé siempre amable y servicial.
"""
DOMINIO_INSTITUCIONAL = "@13dejulio.edu.ar"
FIREBASE_DB = f"https://{st.secrets['firebase_config']['projectId']}-default-rtdb.firebaseio.com"
LOGO_URL = "https://i.imgur.com/gJ5Ym2W.png"

# --- CÓDIGOS SECRETOS SIMPLIFICADOS PARA PRUEBAS ---
CODIGO_SECRETO_PROFESOR = "ADMIN2025TEST"
CODIGO_SECRETO_AUTORIDAD = "ADMIN2025TEST"

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# --- MODIFICAR EL TEXTO DEL ANUNCIO AQUÍ ---
# Si quieres ocultar la barra, deja el texto vacío: ANUNCEMENT_TEXT = ""
ANNOUNCEMENT_TEXT = "Aviso Importante: El próximo lunes es feriado. No habrá actividades en el instituto."
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


# ----------------------------------------------------------------------------------
#  UTILIDADES GENERALES
# ----------------------------------------------------------------------------------
def iso_now():                 return datetime.utcnow().isoformat()
def iso_plus_days(d):         return (datetime.utcnow()+timedelta(days=d)).isoformat()
def is_expired(inv):          return datetime.fromisoformat(inv["expires"]) < datetime.utcnow()

# ----------------------------------------------------------------------------------
#  ENVOLTORIOS FIREBASE
# ----------------------------------------------------------------------------------
def fb_auth(endpoint, data):
    url = f"https://identitytoolkit.googleapis.com/v1/{endpoint}?key={st.secrets['firebase_config']['apiKey']}"
    return requests.post(url, json=data).json()

def fb_db(method, path, data=None, token=None):
    params = {}
    if token: params["auth"] = token
    url = f"{FIREBASE_DB}/{path}.json"
    r = getattr(requests, method.lower())(url, json=data, params=params)
    if not r.ok: st.warning(f"Firebase {method} fail → {r.text}")
    return r.json() if r.ok else None

# ----------------------------------------------------------------------------------
#  SEGURIDAD – FUNCIONES
# ----------------------------------------------------------------------------------
def send_password_reset(email):
    return "email" in fb_auth("accounts:sendOobCode", {"requestType": "PASSWORD_RESET", "email": email})

def log_action(actor_uid, action, payload=None):
    node = f"audit/{datetime.utcnow().strftime('%Y/%m/%d')}/{uuid.uuid4()}"
    fb_db("put", node, {"actor":actor_uid,"act":action,"payload":payload or {}, "ts":iso_now()})

# ----------------------------------------------------------------------------------
#  IA – CARGA EMBEDDINGS
# ----------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_base_de_conocimiento(ruta='conocimiento.json'):
    try:
        with open(ruta,'r',encoding='utf8') as f: return json.load(f)
    except: return None

@st.cache_data(show_spinner=False)
def aplanar_conocimiento(bd):
    docs=[]
    if not bd: return docs
    for top,data in bd.items():
        if top=="material_academico": continue
        docs.append(f"Información sobre {top.replace('_',' ').title()}: {data['content']}")
    if "material_academico" in bd:
        for y,subs in bd["material_academico"].items():
            for s,sd in subs.items():
                row=f"Materia {s.title()} ({y}). {sd.get('content','')}. Prof: {sd.get('profesor','-')}."
                if sd.get('evaluaciones'):
                    row+=" Próximas evaluaciones: "+"; ".join(f"{e['fecha']}: {e['temas']}" for e in sd['evaluaciones'])
                docs.append(row)
    return docs

@st.cache_resource(show_spinner=False)
def recursos_ia():
    modelo=SentenceTransformer('all-MiniLM-L6-v2')
    docs=aplanar_conocimiento(cargar_base_de_conocimiento())
    idx=modelo.encode(docs) if docs else None
    return modelo, docs, idx

def buscar_contexto(q, modelo, docs, idx, datos_usuario):
    contexto=""
    if idx is not None:
        sims=cosine_similarity(modelo.encode([q]), idx)[0]
        for i in np.argsort(sims)[::-1][:3]:
            if sims[i]>0.4: contexto+=f"- {docs[i]}\n"
    if datos_usuario and not st.session_state.get("guest_mode",False):
        contexto+="\n--- DATOS PERSONALES ---\n"+json.dumps({k:v for k,v in datos_usuario.items() if k!='chats'})
    return contexto or "No se encontró información relevante."

def stream_respuesta(cliente, historial):
    try:
        for ch in cliente.chat.completions.create(model=MODELO_PREDETERMINADO, messages=historial, temperature=0.5, max_tokens=1024, stream=True):
            yield ch.choices[0].delta.content or ""
    except Exception as e:
        st.error(f"Error IA: {e}"); yield ""

# ----------------------------------------------------------------------------------
#  UI – ESTÉTICA Y CSS
# ----------------------------------------------------------------------------------
def estilos():
    is_light = st.session_state.get("theme") == "light"
    # Añadimos un padding al cuerpo para que la barra de anuncios no tape el contenido
    top_padding = "45px" if ANNOUNCEMENT_TEXT else "0px"
    st.markdown(f"""
    <style>
      :root {{
         --bg-primary: {"#f0f2f6" if is_light else "#2d2a4c"};
         --bg-secondary: {"#ffffff" if is_light else "#4f4a7d"};
         --text-primary: {"#31333f" if is_light else "#e6e6fa"};
         --text-secondary: {"#555" if is_light else "#a1c9f4"};
         --brand-color: {"#0068c9" if is_light else "#a1c9f4"};
         --bg-user-msg: {"#e8f0fe" if is_light else "#3b3861"};
      }}
      @keyframes pulse {{ 0%{{box-shadow:0 0 10px var(--brand-color)}} 50%{{box-shadow:0 0 25px var(--brand-color)}} 100%{{box-shadow:0 0 10px var(--brand-color)}} }}
      @keyframes fadeIn {{ from{{opacity:0;transform:translateY(20px)}} to{{opacity:1;transform:translateY(0)}} }}
      @keyframes thinking-pulse {{ 0%{{opacity:0.7}} 50%{{opacity:1}} 100%{{opacity:0.7}} }}

      .stApp {{
          background-color: var(--bg-primary);
          background-image: repeating-linear-gradient(45deg, rgba(161, 201, 244, 0.05), rgba(161, 201, 244, 0.05) 1px, transparent 1px, transparent 20px), linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
      }}
      .main {{ padding-top: {top_padding}; }}
      header, [data-testid="stToolbar"] {{ display: none !important; }}

      .announcement-bar {{
          position: fixed; top: 0; left: 0; width: 100%;
          background-color: #111827; color: #e5e7eb;
          text-align: center; padding: 12px 16px;
          font-size: 15px; font-weight: 500;
          z-index: 9999;
      }}

      .login-container {{ max-width: 450px; margin: auto; padding-top: 5rem; }}
      h1,h2,h3 {{ color: var(--text-primary); text-align:center;}}
      h1 {{ text-shadow:0 0 8px var(--brand-color); }}
      
      .main-container {{ animation:fadeIn 0.8s ease-in-out; max-width:900px; margin:auto; padding:2rem 1rem; }}
      [data-testid="stSidebar"] {{ border-right:2px solid var(--brand-color); background-color: var(--bg-primary); }}
      .sidebar-logo {{ width:120px; height:120px; border-radius:50%; border:3px solid var(--brand-color); display:block; margin:2rem auto; animation:pulse 4s infinite ease-in-out; }}
      
      .chat-wrapper {{ border:1px solid var(--brand-color); box-shadow:0 0 20px -5px var(--brand-color); border-radius:20px; background-color:var(--bg-secondary); padding:1rem; margin-top:1rem; }}
      [data-testid="stChatMessage"] {{ animation:fadeIn 0.4s ease-out; background-color: var(--bg-secondary) !important; border-radius: 15px; border: 1px solid var(--brand-color); }}
      [data-testid="stChatMessage"][data-testid-stream-message-type="user"] {{ background-color: var(--bg-user-msg) !important; border-color: transparent; }}
      .thinking-indicator {{ font-style:italic; color: var(--text-secondary); animation:thinking-pulse 1.5s infinite; }}
      .stButton>button {{ width: 100%; margin-bottom: 5px; border-color: var(--text-secondary); }}
      .stButton>button:hover {{ border-color: var(--brand-color); color: var(--brand-color); }}

      @media (max-width:768px) {{ 
          .main-container, .login-container {{ padding:1rem 0.5rem!important; }}
          h1{{font-size:1.8rem}} 
          .sidebar-logo{{width:80px;height:80px}} 
      }}
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
#  LOGIN + REGISTRO
# ----------------------------------------------------------------------------------
def pagina_login():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title("Bienvenido a TecnoBot")

    if st.button("Ingresar como Invitado", use_container_width=True):
        st.session_state.update({"logged_in":True,"guest_mode":True, "user_data":{"nombre":"Invitado","rol":"invitado"}, "chat_history":{}})
        st.rerun()
    
    if st.button("🔧 Ingresar como Admin (Test)", use_container_width=True):
        st.session_state.update({"logged_in":True,"guest_mode":False,
                                 "user_data":{"nombre":"Admin","rol":"autoridad", "email":"admin@test.com"},
                                 "user_uid": "admin_test_uid", "user_token":"admin_test_token",
                                 "chat_history":{}})
        st.rerun()

    st.markdown("---")
    tabs = st.tabs(["Iniciar Sesión","Registrarse"])

    with tabs[0]:
        em = st.text_input("Email", key="login_email")
        pw = st.text_input("Contraseña", type="password", key="login_pass")
        col1,col2 = st.columns(2)
        if col1.button("Ingresar", use_container_width=True):
            r = fb_auth("accounts:signInWithPassword", {"email":em,"password":pw,"returnSecureToken":True})
            if "localId" in r:
                st.session_state.update({"logged_in":True, "user_uid":r["localId"], "user_token":r["idToken"], "guest_mode":False})
                st.rerun()
            else: st.error("Credenciales inválidas.")
        if col2.button("¿Olvidaste tu contraseña?", use_container_width=True):
            if send_password_reset(em): st.success("Te enviamos un email para restablecer tu contraseña.")
            else: st.error("No se pudo enviar el correo. Verifica el email.")

    with tabs[1]:
        nom = st.text_input("Nombre", key="reg_nombre")
        ape = st.text_input("Apellido", key="reg_apellido")
        rem = st.text_input("Email institucional", key="reg_email", help=f"Debe terminar en {DOMINIO_INSTITUCIONAL}")
        rpw = st.text_input("Contraseña", type="password", key="reg_password")
        inv_code = st.text_input("Código de Invitación (opcional)", key="reg_inv_code")
        if st.button("Registrarse", use_container_width=True):
            if not rem.endswith(DOMINIO_INSTITUCIONAL):
                st.error("Usa tu mail institucional."); st.stop()
            rol, colec = "alumno", "alumnos"
            if inv_code:
                inv = fb_db("get", f"invites/{inv_code}")
                if not inv or inv.get("used") or is_expired(inv):
                    st.error("Invitación inválida/expirada."); st.stop()
                rol, colec = inv["type"], inv["type"]+"s"
            r = fb_auth("accounts:signUp", {"email":rem,"password":rpw,"returnSecureToken":True})
            if "localId" in r:
                uid, tok = r["localId"], r["idToken"]
                profile={"nombre":nom,"apellido":ape,"email":rem,"rol":rol,"legajo":str(int(time.time()*100))[-6:]}
                fb_db("put", f"{colec}/{uid}", profile, tok)
                if inv_code: fb_db("put", f"invites/{inv_code}/used", True, tok)
                st.success("¡Registro exitoso! Inicia sesión."); log_action(uid,"register",{"rol":rol})
            else: st.error("No se pudo registrar.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
#  PÁGINAS ADICIONALES (PERFIL, ANUNCIOS, ADMIN)
# ----------------------------------------------------------------------------------
def pagina_perfil():
    st.header("Mi Perfil")
    datos = st.session_state.user_data
    if st.session_state.get("guest_mode"):
        st.warning("El perfil no está disponible en modo invitado."); return
    cols=st.columns(2)
    for k in ("nombre","apellido","telefono"):
        datos[k]=cols[0 if k in ("nombre","telefono") else 1].text_input(k.title(),value=datos.get(k,""))
    if st.button("Guardar cambios"):
        col = datos["rol"]+"s"
        fb_db("put", f"{col}/{st.session_state.user_uid}", datos, st.session_state.user_token)
        log_action(st.session_state.user_uid,"update_profile"); st.success("Perfil actualizado.")

def pagina_anuncios():
    st.header("📢 Novedades")
    posts = fb_db("get","announcements", token=st.session_state.get("user_token")) or {}
    for pid,p in sorted(posts.items(), key=lambda x:x[1]["createdAt"], reverse=True):
        st.subheader(p["title"]); st.markdown(p["body"]); st.caption(f"Publicado el {p['createdAt']}")
    if st.session_state.user_data.get("rol")=="autoridad":
        st.markdown("---"); st.subheader("Publicar aviso")
        t=st.text_input("Título", key="ann_title"); b=st.text_area("Contenido", key="ann_body")
        if st.button("Publicar"):
            fb_db("put", f"announcements/{uuid.uuid4()}", {"title":t,"body":b,"createdAt":iso_now(),"author":st.session_state.user_uid}, st.session_state.user_token)
            log_action(st.session_state.user_uid,"post_announcement",{"t":t}); st.success("Publicado."); st.rerun()

def pagina_admin():
    st.header("👑 Panel de Administración"); tabs = st.tabs(["Gestión de Usuarios", "Generar Invitaciones"])
    with tabs[0]:
        colecciones={"Alumnos":"alumnos","Profesores":"profesores","Autoridades":"autoridades"}
        tab=st.radio("Seleccionar Colección", list(colecciones), horizontal=True)
        datos = fb_db("get", colecciones[tab], token=st.session_state.user_token) or {}
        for uid,u in datos.items():
            with st.expander(f"{u.get('nombre','')} {u.get('apellido','')} ({u.get('rol')})"):
                c = st.columns((3,2,2,1));
                rol = c[0].selectbox("Rol",["alumno","profesor","autoridad"],index=["alumno","profesor","autoridades"].index(u["rol"]),key=f"r{uid}")
                dis = c[1].checkbox("Desactivado", value=u.get("disabled",False), key=f"d{uid}")
                if c[2].button("💾 Guardar", key=f"s{uid}"):
                    fb_db("patch", f"{colecciones[tab]}/{uid}", {"rol":rol,"disabled":dis}, st.session_state.user_token)
                    log_action(st.session_state.user_uid,"admin_update",{uid:{"rol":rol,"disabled":dis}}); st.success("Actualizado"); st.rerun()
    with tabs[1]:
        st.subheader("Generar código de invitación")
        new_code = st.text_input("Código (vacío para generar uno aleatorio)")
        rol_inv = st.selectbox("Asignar Rol",["profesor","autoridad"])
        if st.button("Crear código"):
            code = new_code or str(uuid.uuid4())[:8].upper()
            fb_db("put", f"invites/{code}", {"type":rol_inv,"createdBy":st.session_state.user_uid,"expires":iso_plus_days(7),"used":False}, st.session_state.user_token)
            st.success(f"Código de invitación creado: **{code}** (válido por 7 días)"); log_action(st.session_state.user_uid, "create_invite", {"code":code, "rol":rol_inv})

# ----------------------------------------------------------------------------------
#  CHAT
# ----------------------------------------------------------------------------------
def pagina_chat():
    usuario = st.session_state.user_data
    modelo, docs, idx = st.session_state.recursos_ia
    cliente = groq.Groq(api_key=st.secrets["GROQ_API_KEY"])
    with st.sidebar:
        st.image(LOGO_URL, width=120)
        st.write(f"Hola, **{usuario.get('nombre','')}**")
        if st.toggle("Modo claro", value=st.session_state.get("theme")=="light", key="theme_toggle"): st.session_state["theme"]="light"; st.rerun()
        else: st.session_state["theme"]="dark"
        if st.button("➕ Nuevo Chat", use_container_width=True): start_new_chat()
        st.markdown("---"); st.subheader("Chats")
        for cid,cdata in sorted(st.session_state.chat_history.items(), key=lambda x:x[1]['timestamp'], reverse=True):
            if st.button(cdata.get("titulo","Chat"), key=cid, use_container_width=True):
                st.session_state.active_chat_id=cid; st.rerun()
        if (ac:=st.session_state.get("active_chat_id")):
            st.markdown("---")
            if st.button("✏️ Renombrar Chat", use_container_width=True): st.session_state.renaming_chat = True
            if st.button("🗑 Borrar Chat", use_container_width=True):
                st.session_state.chat_history.pop(ac,None); persist_chat(ac, delete=True); start_new_chat()
        st.markdown("---")
        if st.button("Cerrar Sesión", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
    if st.session_state.get("renaming_chat"):
        ac = st.session_state.active_chat_id
        nuevo = st.text_input("Nuevo título para el chat:", value=st.session_state.chat_history[ac]["titulo"])
        if st.button("Guardar"):
            st.session_state.chat_history[ac]["titulo"] = nuevo; persist_chat(ac); st.session_state.renaming_chat = False; st.rerun()
    st.title("🎓 TecnoBot · Chat")
    active = st.session_state.chat_history.get(st.session_state.active_chat_id)
    if not active: start_new_chat(); active = st.session_state.chat_history[st.session_state.active_chat_id]
    for m in active["mensajes"]:
        with st.chat_message(m["role"], avatar="🤖" if m["role"]=="assistant" else "🧑‍💻"):
            st.markdown(m["content"], unsafe_allow_html=True)
    if prompt:=st.chat_input("Escribe aquí..."):
        active["mensajes"].append({"role":"user","content":prompt})
        contexto = buscar_contexto(prompt,modelo,docs,idx,usuario)
        hist=[{"role":"system","content":SYSTEM_PROMPT+"\n\n"+contexto}]+active["mensajes"][-10:]
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty(); placeholder.markdown('<p class="thinking-indicator">Pensando…</p>', unsafe_allow_html=True)
            full="".join(stream_respuesta(cliente, hist))
            placeholder.markdown(full, unsafe_allow_html=True)
            if not st.session_state.get("guest_mode"):
                cgood,cbad = st.columns(2)
                if cgood.button("👍", key=uuid.uuid4()): fb_db("put", f"feedback/{uuid.uuid4()}", {"msg":full,"score":1,"by":st.session_state.user_uid})
                if cbad.button("👎", key=uuid.uuid4()): fb_db("put", f"feedback/{uuid.uuid4()}", {"msg":full,"score":-1,"by":st.session_state.user_uid})
        active["mensajes"].append({"role":"assistant","content":full})
        if len(active["mensajes"])==2: active["titulo"]=prompt[:30]+"..."
        persist_chat(st.session_state.active_chat_id); st.rerun()

def persist_chat(cid, delete=False):
    if st.session_state.get("guest_mode"): return
    col = st.session_state.user_data["rol"]+"s"
    path = f"{col}/{st.session_state.user_uid}/chats/{cid}"
    fb_db("delete" if delete else "put", path, None if delete else st.session_state.chat_history[cid], st.session_state.user_token)

def start_new_chat():
    cid=str(uuid.uuid4()); st.session_state.active_chat_id=cid
    st.session_state.chat_history[cid]={"titulo":"Nuevo Chat","timestamp":iso_now(),"mensajes":[]}
    if "renaming_chat" in st.session_state: del st.session_state.renaming_chat

# ----------------------------------------------------------------------------------
#  ROUTER PRINCIPAL
# ----------------------------------------------------------------------------------
def app():
    estilos()
    if ANNOUNCEMENT_TEXT: # Muestra la barra si hay texto
        st.markdown(f'<div class="announcement-bar">{ANNOUNCEMENT_TEXT}</div>', unsafe_allow_html=True)

    if not st.session_state.get("logged_in"):
        pagina_login(); return
    if not st.session_state.get("init_loaded"):
        with st.spinner("Cargando sesión..."):
            if not st.session_state.get("guest_mode"):
                uid, tok = st.session_state.user_uid, st.session_state.user_token
                for c in ("alumnos","profesores","autoridades"):
                    if (d:=fb_db("get",f"{c}/{uid}",token=tok)): st.session_state.user_data=d; break
                if not st.session_state.get('user_data'):
                    st.error("No se pudo cargar tu perfil."); st.stop()
            st.session_state.chat_history = st.session_state.user_data.get("chats",{})
            st.session_state.active_chat_id = next(iter(st.session_state.chat_history), None)
            st.session_state.init_loaded=True
    
    if 'recursos_ia' not in st.session_state:
        st.session_state.recursos_ia = recursos_ia()
    
    menu_options = ["Chat"]
    if not st.session_state.get("guest_mode"):
        menu_options.extend(["Mi Perfil", "Anuncios"])
        if st.session_state.user_data.get("rol")=="autoridad":
            menu_options.append("Admin")
    
    menu = st.sidebar.selectbox("Ir a …", menu_options, key="navigation")

    if menu=="Chat": pagina_chat()
    elif menu=="Mi Perfil": pagina_perfil()
    elif menu=="Anuncios": pagina_anuncios()
    elif menu=="Admin": pagina_admin()

# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    app()

