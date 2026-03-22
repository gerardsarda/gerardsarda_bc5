# ============================================================
# CABECERA
# ============================================================
# Alumno: Nombre Apellido
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT =  """
Eres un Analista de Datos experto en música. Tu tarea es responder a las preguntas del usuario sobre su historial de Spotify generando código Python (Pandas y Plotly).

Cuentas con un DataFrame llamado `df` con datos desde {fecha_min} hasta {fecha_max}.
Columnas disponibles:
- `ts`: Fecha y hora (datetime).
- `minutes_played`: Minutos escuchados (Usa esta para medir "cuánto" o "tiempo").
- `track_name`: Canción.
- `artist_name`: Artista.
- `album_name`: Álbum.
- `reason_start`: Motivo de inicio. (Valores: {reason_start_values})
- `reason_end`: Motivo de fin. (Valores: {reason_end_values})
- `platform`: Plataforma usada. (Valores: {plataformas})
- `shuffle`: Booleano (modo aleatorio).
- `skipped`: Booleano (si se saltó la canción).
- `hour`, `day_name`, `month_name`, `is_weekend`, `season`: `season`: Estación del año precalculada. (Valores reales en el dataset: {seasons})
- `year_month`: Año y mes (ej. '2023-01'). Úsala SIEMPRE para gráficos de evolución temporal.
- `is_discovery`: Booleano (True/False). True si es la primera vez que el usuario escucha esa canción. Úsala para calcular descubrimientos.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con un objeto JSON válido. Sin texto antes ni después.
2. Si la pregunta es sobre el historial musical, devuelve:
   {{"tipo": "grafico", "codigo": "TU_CODIGO_PYTHON", "interpretacion": "Breve análisis de 2 líneas de lo que muestra el gráfico."}}
3. Si la pregunta NO es sobre música (ej. clima, recetas, etc.), devuelve:
   {{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "Lo siento, solo analizo datos de Spotify."}}
4. El código Python debe generar obligatoriamente una figura usando Plotly Express (`px`) o Graph Objects (`go`).
5. CRÍTICO: El código debe guardar el gráfico resultante en una variable llamada `fig`. (Ej: fig = px.bar(...))
6. ESTÉTICA DEL GRÁFICO: Usa siempre el color verde corporativo de Spotify (#1DB954) como color principal. Añade títulos descriptivos al gráfico, nombra correctamente los ejes (sin guiones bajos) y usa `fig.update_layout(template='plotly_white')` para que el fondo del gráfico sea limpio y elegante.
7. Agrupa los datos correctamente y, si es un ranking, muestra solo el Top 10.
8. Para gráficos de evolución temporal (ej. por mes), usa SIEMPRE la columna `year_month` en el eje X para que las barras se agrupen bien. Si hay muchas etiquetas en el eje X, rótalas 45 grados (`fig.update_layout(xaxis_tickangle=-45)`).
9. ESTILO DEL TEXTO: En el campo "interpretacion" del JSON, redacta el texto con un tono entusiasta de experto musical. Empieza SIEMPRE la interpretación con un emoji relacionado con el dato (ej. 🎸, 📅, 🎧, 🔥).
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # 1. Fechas y orden cronológico estricto (CRÍTICO para descubrimientos)
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df = df.sort_values('ts') 

    # 2. Renombrar columnas
    df.rename(columns={
        'master_metadata_track_name': 'track_name',
        'master_metadata_album_artist_name': 'artist_name',
        'master_metadata_album_album_name': 'album_name'
    }, inplace=True)

    # 3. Columnas temporales precalculadas
    df['hour'] = df['ts'].dt.hour
    df['day_name'] = df['ts'].dt.day_name()
    df['month_name'] = df['ts'].dt.month_name()
    df['is_weekend'] = df['ts'].dt.dayofweek >= 5
    
    # --- NUEVAS COLUMNAS POTENTES ---
    df['year_month'] = df['ts'].dt.strftime('%Y-%m') # Orden temporal perfecto
    df['is_discovery'] = ~df.duplicated(subset=['track_name', 'artist_name'], keep='first') # Detecta el primer play
    # --------------------------------

    # 4. Estaciones
    mes_a_estacion = {
        12: 'Invierno', 1: 'Invierno', 2: 'Invierno',
        3: 'Primavera', 4: 'Primavera', 5: 'Primavera',
        6: 'Verano', 7: 'Verano', 8: 'Verano',
        9: 'Otoño', 10: 'Otoño', 11: 'Otoño'
    }
    df['season'] = df['ts'].dt.month.map(mes_a_estacion)

    # 5. Minutos y nulos
    df['minutes_played'] = df['ms_played'] / 60000
    df['skipped'] = df['skipped'].fillna(False)

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()
    # NUEVO: Calculamos los valores únicos reales de 'season' (Verano, Invierno, etc.)
    seasons = df["season"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
        # NUEVO: Inyectamos el placeholder seasons
        seasons=seasons,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    La arquitectura separa la lógica analítica de los datos brutos. El LLM recibe únicamente un System Prompt con el esquema del DataFrame (columnas, rangos, valores precalculados) junto a la pregunta del usuario. 
#    A cambio, devuelve estrictamente un objeto JSON que contiene código Python (Pandas y Plotly) y una interpretación en texto. Este código generado se ejecuta localmente en el servidor de Streamlit mediante la función exec(). 
#    El LLM nunca recibe los datos masivos directamente por tres motivos críticos: privacidad (protegemos historiales personales), eficiencia de la API (enviar miles de filas saturaría los tokens y dispararía el coste), y porque los modelos son excelentes escribiendo código analítico, pero ineficientes realizando cálculos matemáticos directos sobre grandes volúmenes de datos.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    Al LLM le proporciono el esquema exacto de columnas, variables temporales precalculadas (`year_month`, `season`) y reglas estrictas de salida (formato JSON y uso obligatorio de la variable `fig`). Lo hago para acotar su margen de invención; si la información es ambigua, la IA asume datos erróneos y el código falla. 
#- Ejemplo de éxito: "Compara mis artistas en verano e invierno" funciona porque inyectamos dinámicamente los valores reales de la columna season (`{seasons}`). Antes de hacerlo, el modelo "alucinaba" filtrando por 'summer' en inglés y devolvía gráficos vacíos.
#- Ejemplo de fallo: Si eliminásemos la regla "guarda el gráfico en la variable `fig`", el LLM podría llamarla `grafico_barras`. La función `execute_chart` (que busca específicamente `fig`) devolvería None y la app fallaría sin mostrar el gráfico.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#1) El usuario introduce una pregunta en el chat. 
#2) La app calcula los valores únicos del dataset actual y los inyecta en el System Prompt para darle contexto. 
#3) Se envía la pregunta y el prompt a la API de OpenAI. 
#4) El modelo devuelve un string en formato JSON con el código de Plotly y un texto explicativo. 
#5) La función `parse_response` limpia el texto y lo convierte en un diccionario de Python. 
#6) `execute_chart` ejecuta el código extraído en el entorno local sobre el DataFrame `df`, generando el objeto visual. 
#7) Streamlit detecta la figura y renderiza el gráfico interactivo junto a la interpretación en la interfaz del usuario.