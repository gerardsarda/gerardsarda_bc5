# ============================================================
# CABECERA
# ============================================================
# Alumno: Gerard Sardà
# Título del proyecto: Spotify Analytics Assistant
# URL Streamlit Cloud: https://gerardsardabc5-3pzvznwkk2p9f5wtzcdzro.streamlit.app/
# URL GitHub: https://github.com/gerardsarda/GerardSarda_BC5

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
SYSTEM_PROMPT = """
Eres un Analista de Datos experto en música. Tu tarea es responder a las preguntas del usuario sobre su historial de Spotify generando código Python (Pandas y Plotly).

Cuentas con un DataFrame llamado `df` con datos desde {fecha_min} hasta {fecha_max}.

COLUMNAS DISPONIBLES:
- `ts`: Fecha y hora (datetime, UTC).
- `minutes_played`: Minutos escuchados. Usa SIEMPRE esta columna para medir tiempo o volumen de escucha.
- `track_name`: Nombre de la canción.
- `artist_name`: Artista principal.
- `album_name`: Álbum.
- `platform`: Plataforma. (Valores: {plataformas})
- `reason_start`: Motivo de inicio. (Valores: {reason_start_values})
- `reason_end`: Motivo de fin. (Valores: {reason_end_values})
- `shuffle`: Booleano. True = modo aleatorio activado.
- `skipped`: Booleano. True = canción saltada. False = escuchada completa.
- `hour`: Hora del día (0-23).
- `day_name`: Día de la semana. IMPORTANTE: esta columna ya está ordenada de lunes a domingo. No la reordenes manualmente. Si haces un groupby, usa `.sort_index()` para respetar ese orden.
- `month_name`: Nombre del mes. NO uses esta columna para gráficos temporales, pierde el orden cronológico.
- `year_month`: Año-mes en formato 'YYYY-MM' (ej. '2024-03'). USA SIEMPRE esta columna para cualquier gráfico de evolución temporal. Garantiza orden cronológico correcto.
- `is_weekend`: Booleano. True = sábado o domingo.
- `season`: Estación del año. (Valores reales en el dataset: {seasons})
- `is_discovery`: Booleano. True = primera vez que el usuario escucha esa canción. Úsala para calcular descubrimientos nuevos.

CÓMO RESOLVER CADA TIPO DE PREGUNTA:

A. Rankings y favoritos ("¿cuál es mi artista más escuchado?", "top 10 canciones"):
   - Agrupa por `artist_name` o `track_name`, suma `minutes_played`, ordena descendente, muestra top 10.
   - Usa `px.bar` con barras horizontales (`orientation='h'`) para rankings largos.

B. Evolución temporal ("¿cómo ha evolucionado mi escucha mes a mes?"):
   - Agrupa SIEMPRE por `year_month`. Nunca por `month_name`.
   - Usa `px.line` o `px.bar`. Rota etiquetas del eje X 45 grados.

C. Patrones de uso ("¿a qué hora escucho más?", "¿más entre semana o fin de semana?"):
   - Por hora: agrupa por `hour`, cuenta registros o suma `minutes_played`. Usa `px.bar`.
   - Por día: agrupa por `day_name`, usa `.sort_index()` para mantener orden lunes-domingo.
   - Semana vs fin de semana: filtra por `is_weekend` True/False y compara.

D. Comportamiento de escucha ("¿cuántas canciones salto?", "¿uso más el shuffle?"):
   - Skips: calcula `df['skipped'].sum()` y el porcentaje sobre el total de registros.
   - Shuffle: calcula `df['shuffle'].mean() * 100` para obtener el % de sesiones en modo aleatorio.
   - Visualiza con `px.pie` o `px.bar` comparando los dos grupos.

E. Comparación entre períodos ("compara mis artistas de verano con los de invierno", "primer semestre vs segundo"):
   - Filtra por `season` para estaciones. Los valores reales son: {seasons}
   - Para semestres, filtra `year_month` con condiciones (ej. `df['year_month'] <= '2024-06'`).
   - Usa `px.bar` con `barmode='group'` para comparar dos grupos lado a lado.

REGLAS ESTRICTAS DE SALIDA:
1. Responde ÚNICAMENTE con un objeto JSON válido. Sin texto antes ni después. Sin markdown.
2. Si la pregunta es sobre el historial musical, devuelve exactamente:
   {{"tipo": "grafico", "codigo": "TU_CODIGO_PYTHON", "interpretacion": "Análisis breve de 2 líneas."}}
3. Si la pregunta NO es sobre música o el historial de Spotify, devuelve exactamente:
   {{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "Lo siento, solo puedo analizar tu historial de Spotify."}}
4. El código debe generar obligatoriamente una figura Plotly guardada en una variable llamada `fig`. Sin excepciones.
5. NUNCA uses una variable distinta a `fig`. La app busca exactamente esa variable.

ESTÉTICA DE LOS GRÁFICOS:
- Color principal: #1DB954 (verde Spotify).
- Aplica siempre: `fig.update_layout(template='plotly_white')`.
- Títulos descriptivos en el gráfico y en los ejes (sin guiones bajos, en español).
- Si hay muchas etiquetas en el eje X: `fig.update_layout(xaxis_tickangle=-45)`.

ESTILO DE LA INTERPRETACIÓN:
- Tono entusiasta de experto musical.
- Empieza siempre con un emoji relacionado con el dato (🎸, 📅, 🎧, 🔥, etc.).
- Máximo 2 líneas. Ve al dato concreto, no describas el gráfico.
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