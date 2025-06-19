import streamlit as st
import groq, json, requests, time, uuid, numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------------
#  CONFIGURACIÓN INICIAL
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="TecnoBot – Instituto 13 de Julio", page_icon="🎓", layout="wide")
# ──────────────────────────────────────────────────────────────────────────────
#  2.A  –  Cargar tipografías (una sola vez)  ### NUEVO ###
# ──────────────────────────────────────────────────────────────────────────────
def cargar_fuentes():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# Llama a esta función justo después de la línea st.set_page_config(…):
# st.set_page_config(...)
# cargar_fuentes()                              ← AÑADIR

# ──────────────────────────────────────────────────────────────────────────────
#  2.B  –  Hoja de estilos completa (modo claro / oscuro, glass, neon, hovers)
#         Sustituye la función aplicar_estilos_css() original por ésta
# ──────────────────────────────────────────────────────────────────────────────
def aplicar_estilos_css():
    tema = st.session_state.get("theme", "dark")          # 'dark' por defecto
    claro = tema == "light"
    st.markdown(f"""
    <style>
    /* -------------------------------------------------
       Variables globales
    ------------------------------------------------- */
    :root {{
        --clr-bg-prim   : {"#f4f7ff" if claro else "#2d2a4c"};
        --clr-bg-sec    : {"#ffffff" if claro else "rgba(45,42,76,0.6)"};
        --clr-acento    : #a1c9f4;
        --clr-text-main : {"#2d2a4c" if claro else "#e6e6fa"};
        --clr-text-sub  : {"#3e3e68" if claro else "#b8b8d4"};
        --borde-neon    : 0 0 12px var(--clr-acento);
        --radio-base    : 18px;
        --vel-trans     : .25s;
    }}

    /* -------------------------------------------------
       Tipografía general
    ------------------------------------------------- */
    html, body, [class*="st-"] {{
        font-family: "Inter", sans-serif;
        color: var(--clr-text-main);
    }}
    h1,h2,h3,h4,h5,h6  {{ font-family:"Orbitron", sans-serif; color:var(--clr-text-main); }}

    /* -------------------------------------------------
       Fondo con sutil textura de circuitos
    ------------------------------------------------- */
    .stApp {{
        background:
            repeating-linear-gradient(145deg,transparent 0 2px,rgba(255,255,255,.03) 2px 4px),
            url("data:image/svg+xml;utf8,\
            <svg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120' opacity='0.04'>\
               <g stroke='%23a1c9f4' stroke-width='1'>\
                 <path d='M0 60h20m80 0h20M60 0v20m0 80v20'/>\
                 <circle cx='60' cy='60' r='8' fill='none'/>\
               </g></svg>") center/240px var(--clr-bg-prim);
        background-color: var(--clr-bg-prim);
    }}

    /* -------------------------------------------------
       Contenedores principales
    ------------------------------------------------- */
    .main > div:first-child {{ padding-top: 0rem; }}
    header, [data-testid="stToolbar"] {{ display:none !important; }}
    .chat-wrapper {{
        background: var(--clr-bg-sec);
        backdrop-filter: blur(10px);          /* Glassmorphism */
        border:1px solid rgba(161,201,244,.25);
        border-radius:var(--radio-base);
        box-shadow:0 8px 25px -6px rgba(0,0,0,.35);
        padding:1.2rem;
        transition:box-shadow var(--vel-trans);
    }}
    .chat-wrapper:hover {{ box-shadow:0 0 15px -2px var(--clr-acento); }}

    /* -------------------------------------------------
       Sidebar
    ------------------------------------------------- */
    [data-testid="stSidebar"] {{
        border-right:2px solid var(--clr-acento);
        background:rgba(45,42,76,0.85);
        backdrop-filter: blur(6px);
    }}
    .sidebar-logo {{
        width:110px; height:110px; border-radius:50%;
        border:2px solid var(--clr-acento);
        box-shadow:var(--borde-neon);
        display:block; margin:2rem auto;
        transition:transform .6s;
    }}
    .sidebar-logo:hover {{ transform:rotate(360deg) scale(1.05); }}

    /* Botones en el sidebar */
    .stButton>button {{
        background:linear-gradient(135deg,var(--clr-acento) 0%,#7db0ff 100%);
        color:#0a0a23; border:none; border-radius:10px;
        box-shadow:var(--borde-neon);
        transition:transform var(--vel-trans), box-shadow var(--vel-trans);
    }}
    .stButton>button:hover {{
        transform:translateY(-3px) scale(1.03);
        box-shadow:0 0 18px var(--clr-acento);
    }}

    /* -------------------------------------------------
       Mensajes del chat y micro‑interacciones
    ------------------------------------------------- */
    [data-testid="stChatMessage"] {{
        animation:fadeIn .35s ease-out both;
    }}
    .stChatMessage .stMarkdown:hover {{
        background:rgba(161,201,244,.08);
        border-radius:var(--radio-base);
        padding:.2rem .5rem;
        transition:background .2s;
    }}
    @keyframes fadeIn {{ from{{opacity:0;transform:translateY(15px)}} to{{opacity:1;transform:translateY(0)}} }}
    @keyframes pulse {{ 0%{{box-shadow:0 0 6px var(--clr-acento)}} 50%{{box-shadow:0 0 18px var(--clr-acento)}} 100%{{box-shadow:0 0 6px var(--clr-acento)}} }}

    /* “Thinking…” indicador */
    .thinking-indicator {{
        font-style:italic; color:var(--clr-text-sub);
        animation:pulse 1.4s infinite;
    }}

    /* -------------------------------------------------
       Responsive
    ------------------------------------------------- */
    @media (max-width:768px) {{
        .chat-wrapper {{ padding:.6rem; border-radius:14px; }}
        .sidebar-logo {{ width:80px;height:80px; }}
        h1 {{ font-size:1.7rem; }}
    }}
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
#  CONSTANTES
# ----------------------------------------------------------------------------------
MODELO_PREDETERMINADO = "llama3-8b-8192"
SYSTEM_PROMPT = """
Eres TecnoBot, el asistente virtual del Instituto 13 de Julio. Responde ÚNICAMENTE con el CONTEXTO proporcionado
(público + datos personales). Si la respuesta no está, indica que no la tienes y sugiere contactar a secretaría.
Sé siempre amable y servicial.
"""

DOMINIO_INSTITUCIONAL = "@13dejulio.edu.ar"               # <- mails válidos
FIREBASE_DB = f"https://{st.secrets['firebase_config']['projectId']}-default-rtdb.firebaseio.com"

# ----------------------------------------------------------------------------------
#  UTILIDADES GENERALES
# ----------------------------------------------------------------------------------
def iso_now():                 return datetime.utcnow().isoformat()
def iso_plus_days(d):         return (datetime.utcnow()+timedelta(days=d)).isoformat()
def is_expired(inv):          return datetime.fromisoformat(inv["expires"]) < datetime.utcnow()

# ----------------------------------------------------------------------------------
#  ENVOLTORIOS FIREBASE
# ----------------------------------------------------------------------------------
def fb_auth(endpoint, data):   # Auth REST
    url = f"https://identitytoolkit.googleapis.com/v1/{endpoint}?key={st.secrets['firebase_config']['apiKey']}"
    return requests.post(url, json=data).json()

def fb_db(method, path, data=None, token=None, params=None):
    if params is None: params = {}
    if token: params["auth"] = token
    url = f"{FIREBASE_DB}/{path}.json"
    r = getattr(requests, method)(url, json=data, params=params)
    if not r.ok: st.warning(f"Firebase {method} fail → {r.text}")
    return r.json() if r.ok else None

# ----------------------------------------------------------------------------------
#  SEGURIDAD – FUNCIONES
# ----------------------------------------------------------------------------------
def send_password_reset(email):
    return "email" in fb_auth("accounts:sendOobCode",
                              {"requestType": "PASSWORD_RESET", "email": email})

def log_action(actor_uid, action, payload=None):     # auditoría
    node = f"audit/{datetime.utcnow().strftime('%Y/%m/%d')}/{uuid.uuid4()}"
    fb_db("put", node, {"actor":actor_uid,"act":action,"payload":payload or {}, "ts":iso_now()})

# ----------------------------------------------------------------------------------
#  IA – CARGA EMBEDDINGS  (sin cambios sustanciales)
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
                    row+=" Próximas evaluaciones: "+"; ".join(
                        f"{e['fecha']}: {e['temas']}" for e in sd['evaluaciones']
                    )
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
        for ch in cliente.chat.completions.create(model=MODELO_PREDETERMINADO,
                                                  messages=historial,
                                                  temperature=0.5, max_tokens=1024,
                                                  stream=True):
            yield ch.choices[0].delta.content or ""
    except Exception as e:
        st.error(f"Error IA: {e}"); yield ""

# ----------------------------------------------------------------------------------
#  UI – CSS (original)            
# ----------------------------------------------------------------------------------
def estilos():
    st.markdown(f"""
    <style>
      :root {{
         /* modo claro/oscuro dinámico ↓ */
         --bg-primary: {"#f0f2f6" if st.session_state.get("theme")=="light" else "#2d2a4c"};
         --bg-sec: {"#ffffff" if st.session_state.get("theme")=="light" else "#4f4a7d"};
      }}
      .stApp {{ background-color:var(--bg-primary); }}
      /* resto de tu CSS sin cambios ... */
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
#  LOGIN + REGISTRO
# ----------------------------------------------------------------------------------
def pagina_login():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title("Bienvenido a TecnoBot")

    # ------------ acceso invitado
    if st.button("Ingresar como Invitado", use_container_width=True):
        st.session_state.update({"logged":True,"guest_mode":True,
                                 "user_data":{"nombre":"Invitado","rol":"invitado"},
                                 "chat_history":{}})
        st.experimental_rerun()

    st.markdown("---")
    tabs = st.tabs(["Iniciar Sesión","Registrarse"])

    # ------------ login
    with tabs[0]:
        em = st.text_input("Email")
        pw = st.text_input("Contraseña", type="password")
        col1,col2 = st.columns(2)
        if col1.button("Ingresar", use_container_width=True):
            r = fb_auth("accounts:signInWithPassword",
                        {"email":em,"password":pw,"returnSecureToken":True})
            if "localId" in r:
                st.session_state.update({"logged":True,
                                         "user_uid":r["localId"],
                                         "user_token":r["idToken"],
                                         "guest_mode":False})
                st.experimental_rerun()
            else: st.error("Credenciales inválidas.")
        #  --- RESET CONTRASEÑA  ### NUEVO ###
        if col2.button("¿Olvidaste tu contraseña?", use_container_width=True):
            if send_password_reset(em):
                st.success("Te enviamos un email para restablecer tu contraseña.")
            else:
                st.error("No se pudo enviar el correo. Verifica el email.")

    # ------------ registro
    with tabs[1]:
        nom = st.text_input("Nombre")
        ape = st.text_input("Apellido")
        rem = st.text_input("Email institucional", help=f"Debe terminar en {DOMINIO_INSTITUCIONAL}")
        rpw = st.text_input("Contraseña", type="password")
        inv_code = st.text_input("Código de Invitación (lo genera la autoridad)")
        if st.button("Registrarse", use_container_width=True):
            if not rem.endswith(DOMINIO_INSTITUCIONAL):
                st.error("Usa tu mail institucional."); st.stop()
            # validar código de invitación  ### NUEVO ###
            rol="alumno"; colec="alumnos"
            if inv_code:
                inv = fb_db("get", f"invites/{inv_code}")
                if not inv or inv["used"] or is_expired(inv):
                    st.error("Invitación inválida/expirada."); st.stop()
                rol, colec = inv["type"], inv["type"]+"s"
            r = fb_auth("accounts:signUp", {"email":rem,"password":rpw,"returnSecureToken":True})
            if "localId" in r:
                uid, tok = r["localId"], r["idToken"]
                profile={"nombre":nom,"apellido":ape,"email":rem,"rol":rol,"legajo":str(int(time.time()*100))[-6:]}
                fb_db("put", f"{colec}/{uid}", profile, tok)
                if inv_code: fb_db("put", f"invites/{inv_code}/used", True, tok)
                st.success("¡Registro exitoso! Inicia sesión.")
                log_action(uid,"register",{"rol":rol})
            else: st.error("No se pudo registrar.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
#  NOTIFICACIONES PROACTIVAS  ### NUEVO ###
# ----------------------------------------------------------------------------------
def unseen_notifications(uid, tok):
    data = fb_db("get", f"notifications/{uid}", token=tok) or {}
    return {k:v for k,v in data.items() if not v.get("seen")}

def mark_seen(uid, nid, tok):
    fb_db("patch", f"notifications/{uid}/{nid}", {"seen":True}, tok)

# ----------------------------------------------------------------------------------
#  PÁGINA DE PERFIL  ### NUEVO ###
# ----------------------------------------------------------------------------------
def pagina_perfil():
    st.header("Mi Perfil")
    datos = st.session_state.user_data
    cols=st.columns(2)
    for k in ("nombre","apellido","telefono"):
        datos[k]=cols[0 if k in ("nombre","telefono") else 1].text_input(k.title(),value=datos.get(k,""))
    if st.button("Guardar cambios"):
        col = datos["rol"]+"s"
        fb_db("put", f"{col}/{st.session_state.user_uid}", datos, st.session_state.user_token)
        log_action(st.session_state.user_uid,"update_profile")
        st.success("Perfil actualizado.")

# ----------------------------------------------------------------------------------
#  TABLÓN DE ANUNCIOS  ### NUEVO ###
# ----------------------------------------------------------------------------------
def pagina_anuncios():
    st.header("📢 Novedades")
    posts = fb_db("get","announcements") or {}
    for pid,p in sorted(posts.items(), key=lambda x:x[1]["createdAt"], reverse=True):
        st.subheader(p["title"]); st.markdown(p["body"]); st.caption(p["createdAt"])
    if st.session_state.user_data["rol"]=="autoridad":
        st.markdown("---"); st.subheader("Publicar aviso")
        t=st.text_input("Título"); b=st.text_area("Contenido")
        if st.button("Publicar"):
            fb_db("put", f"announcements/{uuid.uuid4()}",
                  {"title":t,"body":b,"createdAt":iso_now(),"author":st.session_state.user_uid},
                  st.session_state.user_token)
            log_action(st.session_state.user_uid,"post_announcement",{"t":t})
            st.success("Publicado."); st.experimental_rerun()

# ----------------------------------------------------------------------------------
#  DASHBOARD ADMINISTRACIÓN  ### NUEVO ###
# ----------------------------------------------------------------------------------
def pagina_admin():
    st.header("👑 Panel de Administración")
    colecciones={"Alumnos":"alumnos","Profesores":"profesores","Autoridades":"autoridades"}
    tab=st.radio("Colección", list(colecciones))
    datos = fb_db("get", colecciones[tab]) or {}
    for uid,u in datos.items():
        c = st.columns((3,2,2,1))
        c[0].write(f"{u.get('nombre','')} {u.get('apellido','')}")
        rol = c[1].selectbox("Rol",["alumno","profesor","autoridad"],index=["alumno","profesor","autoridad"].index(u["rol"]),key=f"r{uid}")
        dis = c[2].checkbox("Desactivado", value=u.get("disabled",False), key=f"d{uid}")
        if c[3].button("💾", key=f"s{uid}"):
            fb_db("patch", f"{colecciones[tab]}/{uid}", {"rol":rol,"disabled":dis}, st.session_state.user_token)
            log_action(st.session_state.user_uid,"admin_update",{uid:{"rol":rol,"disabled":dis}})
            st.success("Actualizado"); st.experimental_rerun()
    st.markdown("---"); st.subheader("Generar código de invitación")
    new_code = st.text_input("Código (vacío → UUID)")
    rol_inv = st.selectbox("Rol",["profesor","autoridad"])
    if st.button("Crear código"):
        code = new_code or str(uuid.uuid4())[:8].upper()
        fb_db("put", f"invites/{code}",
              {"type":rol_inv,"createdBy":st.session_state.user_uid,"expires":iso_plus_days(7),"used":False},
              st.session_state.user_token)
        st.success(f"Creado: {code}")

# ----------------------------------------------------------------------------------
#  CHAT (mejorado con rename/delete/feedback/notifs)
# ----------------------------------------------------------------------------------
def pagina_chat():
    LOGO = "https://13dejulio.edu.ar/wp-content/uploads/2022/03/Isologotipo-13-de-Julio-400.png"
    usuario = st.session_state.user_data
    modelo, docs, idx = recursos_ia()
    cliente = groq.Groq(api_key=st.secrets["GROQ_API_KEY"])

    # ---------------- Sidebar
    with st.sidebar:
        st.image(LOGO, width=120)
        st.write(f"Hola, **{usuario.get('nombre','')}**")
        # toggle tema luz/oscuro
        if st.toggle("Modo claro", value=st.session_state.get("theme")=="light"):
            st.session_state["theme"]="light"
        else: st.session_state["theme"]="dark"
        if st.button("➕ Nuevo Chat", use_container_width=True): start_new_chat()
        st.markdown("---"); st.subheader("Chats")
        for cid,cdata in sorted(st.session_state.chat_history.items(),
                                key=lambda x:x[1]['timestamp'], reverse=True):
            if st.button(cdata.get("titulo","Chat"), key=cid, use_container_width=True):
                st.session_state.active_chat_id=cid; st.experimental_rerun()
        if (ac:=st.session_state.get("active_chat_id")):
            # rename/delete  ### NUEVO ###
            c1,c2=st.columns(2)
            if c1.button("✏️ Renombrar", use_container_width=True):
                nuevo=st.text_input("Nuevo título", value=st.session_state.chat_history[ac]["titulo"])
                if st.button("Guardar título"):
                    st.session_state.chat_history[ac]["titulo"]=nuevo
                    persist_chat(ac)
                    st.success("Renombrado."); st.experimental_rerun()
            if c2.button("🗑 Borrar", use_container_width=True):
                st.session_state.chat_history.pop(ac,None); persist_chat(ac, delete=True)
                st.experimental_rerun()
        st.markdown("---")
        if st.button("Cerrar Sesión", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.experimental_rerun()

    # ---------------- Notificaciones proactivas  ### NUEVO ###
    if not st.session_state.get("guest_mode",False):
        for nid,n in unseen_notifications(st.session_state.user_uid, st.session_state.user_token).items():
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**{n['title']}**  \n{n['body']}")
            mark_seen(st.session_state.user_uid, nid, st.session_state.user_token)

    # ---------------- Historial
    estilos()
    active = st.session_state.chat_history.get(st.session_state.active_chat_id)
    if not active: start_new_chat(); active = st.session_state.chat_history[st.session_state.active_chat_id]
    st.title("🎓 TecnoBot · Chat")
    for m in active["mensajes"]:
        with st.chat_message(m["role"], avatar="🤖" if m["role"]=="assistant" else "🧑‍💻"):
            st.markdown(m["content"], unsafe_allow_html=True)

    if prompt:=st.chat_input("Escribe aquí..."):
        active["mensajes"].append({"role":"user","content":prompt})
        contexto = buscar_contexto(prompt,modelo,docs,idx,usuario)
        hist=[{"role":"system","content":SYSTEM_PROMPT+"\n\n"+contexto}]+active["mensajes"][-10:]
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty(); placeholder.markdown("*Pensando…*")
            full="".join(stream_respuesta(cliente, hist))
            placeholder.markdown(full, unsafe_allow_html=True)
            # thumbs feedback  ### NUEVO ###
            cgood,cbad = st.columns(2)
            if cgood.button("👍", key=f"g{uuid.uuid4()}"):
                fb_db("put", f"feedback/{uuid.uuid4()}",
                      {"msg":full,"score":1,"by":st.session_state.user_uid})
            if cbad.button("👎", key=f"b{uuid.uuid4()}"):
                fb_db("put", f"feedback/{uuid.uuid4()}",
                      {"msg":full,"score":-1,"by":st.session_state.user_uid})
        active["mensajes"].append({"role":"assistant","content":full})
        if len(active["mensajes"])==2: active["titulo"]=prompt[:30]+"..."
        persist_chat(st.session_state.active_chat_id)
        st.experimental_rerun()

def persist_chat(cid, delete=False):
    if st.session_state.get("guest_mode"): return
    col = st.session_state.user_data["rol"]+"s"
    path = f"{col}/{st.session_state.user_uid}/chats/{cid}"
    fb_db("put", path, None if delete else st.session_state.chat_history[cid],
          st.session_state.user_token)

def start_new_chat():
    cid=str(uuid.uuid4())
    st.session_state.active_chat_id=cid
    st.session_state.chat_history[cid]={"titulo":"Nuevo Chat","timestamp":iso_now(),"mensajes":[]}

# ----------------------------------------------------------------------------------
#  ROUTER PRINCIPAL
# ----------------------------------------------------------------------------------
def app():
    if not st.session_state.get("logged"):
        pagina_login(); return

    # carga perfil y chats al primer login
    if not st.session_state.get("init_loaded") and not st.session_state.get("guest_mode"):
        uid, tok = st.session_state.user_uid, st.session_state.user_token
        for c in ("alumnos","profesores","autoridades"):
            if (d:=fb_db("get",f"{c}/{uid}",token=tok)): st.session_state.user_data=d; break
        st.session_state.chat_history = st.session_state.user_data.get("chats",{})
        st.session_state.active_chat_id = next(iter(st.session_state.chat_history), None)
        st.session_state.init_loaded=True

    # ---------- Navegación (solo un archivo) ----------
    menu = st.sidebar.selectbox(
        "Ir a …",
        ["Chat","Mi Perfil","Anuncios"] + (["Admin"] if st.session_state.user_data["rol"]=="autoridad" else [])
    )
    if menu=="Chat":            pagina_chat()
    elif menu=="Mi Perfil":     pagina_perfil()
    elif menu=="Anuncios":      pagina_anuncios()
    elif menu=="Admin":         pagina_admin()

# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    app()
