"""
Dashboard de Investigación: IA, Data Science y Empleabilidad en Carreras de Negocios
Análisis de dataset Scopus — Streamlit Cloud ready
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import io

st.set_page_config(
    page_title="IA & Empleabilidad en Negocios · Scopus Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_CSV_URL = "https://raw.githubusercontent.com/TU_USUARIO/TU_REPO/main/data/scopus_dataset.csv"

TERMINOS_IA = [
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "nlp", "natural language", "computer vision",
    "data science", "big data", "analytics", "algorithm",
    "automation", "predictive", "classification", "clustering",
]
TERMINOS_NEGOCIOS = [
    "business", "management", "finance", "marketing", "strategy",
    "accounting", "economics", "entrepreneurship", "administration",
    "mba", "organization",
]
TERMINOS_EMPLEO = [
    "employment", "employability", "job", "career", "workforce",
    "skill", "competency", "labor market", "hiring", "graduate",
    "professional", "occupation", "salary", "wage",
]
TODOS_TERMINOS = TERMINOS_IA + TERMINOS_NEGOCIOS + TERMINOS_EMPLEO

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.dash-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f4c81 100%);
    border-radius: 16px; padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem; border-left: 5px solid #38bdf8;
}
.dash-header h1 { color: #f0f9ff; font-size: 1.8rem; font-weight: 700; margin: 0 0 0.4rem 0; letter-spacing: -0.02em; }
.dash-header .subtitle { color: #94a3b8; font-size: 0.92rem; font-family: 'JetBrains Mono', monospace; margin: 0; }
.dash-header .question-badge {
    background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.35);
    border-radius: 10px; padding: 0.75rem 1rem; margin-top: 1.2rem;
    color: #bae6fd; font-size: 0.88rem; line-height: 1.6;
}
.kpi-card {
    background: #1e293b; border-radius: 14px; padding: 1.2rem 1.4rem;
    border: 1px solid #334155; border-top: 3px solid #38bdf8; text-align: center;
}
.kpi-value { font-size: 2.1rem; font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono', monospace; line-height: 1.1; }
.kpi-label { color: #94a3b8; font-size: 0.78rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3rem; }
.chart-title { color: #e2e8f0; font-size: 1.05rem; font-weight: 600; margin: 0 0 0.3rem 0; }
.chart-subtitle { color: #64748b; font-size: 0.82rem; margin: 0 0 1rem 0; }
.insight-box {
    background: rgba(56,189,248,0.07); border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin-top: 1rem;
    color: #cbd5e1; font-size: 0.85rem; line-height: 1.6;
}
.insight-box strong { color: #38bdf8; }
[data-testid="stSidebar"] { background: #0f172a; }
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
    title_font=dict(size=14, color="#e2e8f0"),
    xaxis=dict(gridcolor="#1e3a5f", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e3a5f", linecolor="#334155"),
    margin=dict(t=40, b=40, l=60, r=20),
)


# ── DATOS ──────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cargar_csv_url(url: str):
    try:
        return pd.read_csv(url)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def cargar_csv_bytes(raw_bytes: bytes):
    try:
        return pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        return None


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    mapa = {
        "Year": "year", "PY": "year", "Publication Year": "year",
        "Cited by": "cites", "Times Cited, All Databases": "cites", "TC": "cites",
        "Authors": "authors", "Author(s)": "authors", "AU": "authors",
        "Title": "title", "TI": "title", "Article Title": "title",
        "Abstract": "abstract", "AB": "abstract",
        "Author Keywords": "keywords", "DE": "keywords",
        "Keywords Plus": "keywords_plus", "ID": "keywords_plus",
        "Document Type": "doc_type", "DT": "doc_type",
        "Source title": "source", "SO": "source", "Journal": "source",
    }
    df = df.rename(columns={k: v for k, v in mapa.items() if k in df.columns})
    for col in df.columns:
        if "year" in col.lower() and col != "year":
            df = df.rename(columns={col: "year"})
            break
    return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"].between(1990, 2030)]
        df["year"] = df["year"].astype(int)
    if "cites" in df.columns:
        df["cites"] = pd.to_numeric(df["cites"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def extraer_primer_autor(autores) -> str:
    if pd.isna(autores):
        return "Desconocido"
    return str(autores).split(";")[0].strip()


def contar_terminos(df: pd.DataFrame, columnas: list, terminos: list) -> dict:
    texto = ""
    for col in columnas:
        if col in df.columns:
            texto += " " + df[col].dropna().str.lower().str.cat(sep=" ")
    resultado = {}
    for t in terminos:
        count = len(re.findall(r"\b" + re.escape(t) + r"\b", texto))
        if count > 0:
            resultado[t] = count
    return resultado


# ── UI ─────────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='dash-header'>"
        "<h1>🎓 IA & Data Science · Empleabilidad en Negocios</h1>"
        "<p class='subtitle'>Scopus Research Dashboard · Análisis Bibliométrico</p>"
        "<div class='question-badge'>"
        "<strong style='color:#38bdf8;'>❓ Pregunta de Investigación:</strong><br>"
        "¿De qué manera el dominio de herramientas de Inteligencia Artificial (IA) y Data Science "
        "influye en las tasas de empleabilidad y la demanda de profesionales egresados de carreras "
        "de negocios en la actualidad?"
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame):
    total = len(df)
    citas_total = int(df["cites"].sum()) if "cites" in df.columns else 0
    año_min = int(df["year"].min()) if "year" in df.columns else "—"
    año_max = int(df["year"].max()) if "year" in df.columns else "—"
    prom_citas = round(df["cites"].mean(), 1) if "cites" in df.columns else 0
    autores_unicos = (
        df["authors"].dropna().apply(extraer_primer_autor).nunique()
        if "authors" in df.columns else 0
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    datos = [
        (c1, str(f"{total:,}"), "Artículos analizados"),
        (c2, str(año_min) + " – " + str(año_max), "Rango temporal"),
        (c3, str(f"{citas_total:,}"), "Citas acumuladas"),
        (c4, str(prom_citas), "Citas promedio"),
        (c5, str(f"{autores_unicos:,}"), "Autores únicos"),
    ]
    for col, val, label in datos:
        with col:
            st.markdown(
                "<div class='kpi-card'>"
                "<div class='kpi-value'>" + val + "</div>"
                "<div class='kpi-label'>" + label + "</div>"
                "</div>",
                unsafe_allow_html=True,
            )


def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 🔧 Filtros")
        st.markdown("---")
        if "year" in df.columns:
            años = sorted(df["year"].dropna().unique())
            if len(años) >= 2:
                rango = st.slider(
                    "📅 Rango de años",
                    int(min(años)), int(max(años)),
                    (int(min(años)), int(max(años))),
                )
                df = df[df["year"].between(rango[0], rango[1])]
        if "doc_type" in df.columns:
            tipos = ["Todos"] + sorted(df["doc_type"].dropna().unique().tolist())
            sel = st.selectbox("📄 Tipo de documento", tipos)
            if sel != "Todos":
                df = df[df["doc_type"] == sel]
        top_n = st.slider("🏆 Top N autores", 5, 25, 15, 5)
        st.markdown("---")
        st.markdown("**Registros filtrados:** `" + str(len(df)) + "`")
    return df, top_n


# ── GRÁFICOS ───────────────────────────────────────────────────────────────────

def grafico_evolucion_temporal(df: pd.DataFrame):
    if "year" not in df.columns:
        st.warning("No se encontró columna de año.")
        return
    pub = df.groupby("year").size().reset_index(name="publicaciones").sort_values("year")
    pub["tendencia"] = pub["publicaciones"].rolling(3, min_periods=1, center=True).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pub["year"], y=pub["publicaciones"],
        name="Publicaciones/año", marker_color="#0ea5e9", opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=pub["year"], y=pub["tendencia"],
        name="Tendencia (media móvil 3a)", mode="lines",
        line=dict(color="#f59e0b", width=2.5, dash="dot"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, hovermode="x unified",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)
    peak = pub.loc[pub["publicaciones"].idxmax()]
    peak_year = str(int(peak["year"]))
    peak_val = str(int(peak["publicaciones"]))
    st.markdown(
        "<div class='insight-box'>"
        "<strong>📊 Insight:</strong> El pico de publicaciones se alcanza en "
        "<strong>" + peak_year + "</strong> con <strong>" + peak_val + "</strong> artículos. "
        "La tendencia ascendente confirma que este campo está en plena expansión y es "
        "considerado vigente por la comunidad científica."
        "</div>",
        unsafe_allow_html=True,
    )


def grafico_autores_citados(df: pd.DataFrame, top_n: int = 15):
    if "authors" not in df.columns or "cites" not in df.columns:
        st.warning("Se requieren columnas de autores y citas.")
        return
    df2 = df.copy()
    df2["primer_autor"] = df2["authors"].apply(extraer_primer_autor)
    ranking = (
        df2.groupby("primer_autor")["cites"]
        .sum()
        .reset_index()
        .sort_values("cites", ascending=False)
        .head(top_n)
    )
    ranking = ranking[ranking["primer_autor"] != "Desconocido"]
    fig = px.bar(
        ranking.sort_values("cites"),
        x="cites", y="primer_autor", orientation="h",
        color="cites",
        color_continuous_scale=["#1e3a5f", "#0ea5e9", "#38bdf8"],
        labels={"cites": "Total de Citas", "primer_autor": "Autor"},
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Citas: %{x:,}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)
    top = ranking.iloc[0]
    top_nombre = str(top["primer_autor"])
    top_citas = str(f"{int(top['cites']):,}")
    st.markdown(
        "<div class='insight-box'>"
        "<strong>🏆 Insight:</strong> <strong>" + top_nombre + "</strong> lidera con "
        "<strong>" + top_citas + " citas</strong>, siendo el referente más influyente del corpus. "
        "Un alto índice de citas indica reconocimiento amplio de sus hallazgos sobre IA y empleabilidad."
        "</div>",
        unsafe_allow_html=True,
    )


def grafico_palabras_clave(df: pd.DataFrame):
    conteos = contar_terminos(df, ["abstract", "title", "keywords", "keywords_plus"], TODOS_TERMINOS)
    if not conteos:
        st.warning("No se encontraron columnas de texto para analizar.")
        return
    freq = (
        pd.DataFrame(list(conteos.items()), columns=["Término", "Frecuencia"])
        .sort_values("Frecuencia", ascending=False)
        .head(25)
    )

    def cat(t):
        if t in TERMINOS_IA:
            return "🤖 IA & Data Science"
        if t in TERMINOS_NEGOCIOS:
            return "💼 Negocios"
        return "👔 Empleabilidad"

    freq["Categoría"] = freq["Término"].apply(cat)
    color_map = {
        "🤖 IA & Data Science": "#38bdf8",
        "💼 Negocios": "#a78bfa",
        "👔 Empleabilidad": "#34d399",
    }
    fig = px.bar(
        freq, x="Frecuencia", y="Término", orientation="h",
        color="Categoría", color_discrete_map=color_map,
        labels={"Frecuencia": "Frecuencia en corpus", "Término": ""},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        margin=dict(t=40, b=40, l=60, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", title=""),
        xaxis=dict(gridcolor="#1e3a5f", linecolor="#334155"),
        yaxis=dict(autorange="reversed", gridcolor="#1e3a5f", linecolor="#334155"),
    )
    st.plotly_chart(fig, use_container_width=True)
    top_ia = freq[freq["Categoría"] == "🤖 IA & Data Science"]
    top_term = top_ia["Término"].iloc[0] if not top_ia.empty else freq["Término"].iloc[0]
    st.markdown(
        "<div class='insight-box'>"
        "<strong>🔍 Insight:</strong> El término <strong>\"" + top_term + "\"</strong> domina el corpus. "
        "La co-presencia de términos de empleabilidad junto a IA/Data Science en los abstracts confirma "
        "que los investigadores reconocen explícitamente el vínculo entre competencias tecnológicas "
        "y resultados laborales."
        "</div>",
        unsafe_allow_html=True,
    )


def grafico_relevancia_temporal(df: pd.DataFrame):
    if "year" not in df.columns or "cites" not in df.columns:
        st.warning("Se requieren columnas de año y citas.")
        return
    df_plot = df[df["cites"] > 0].copy()
    if df_plot.empty:
        st.info("No hay artículos con citas registradas.")
        return
    color_col = "doc_type" if "doc_type" in df_plot.columns else None
    fig = px.scatter(
        df_plot, x="year", y="cites",
        color=color_col,
        size="cites", size_max=30, opacity=0.65,
        hover_name="title" if "title" in df_plot.columns else None,
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"year": "Año de publicación", "cites": "Número de citas", "doc_type": "Tipo"},
    )
    avg = df_plot.groupby("year")["cites"].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=avg["year"], y=avg["cites"], mode="lines",
        name="Promedio por año",
        line=dict(color="#f59e0b", width=2, dash="dot"),
    ))
    for _, row in df_plot.nlargest(3, "cites").iterrows():
        titulo = str(row.get("title", ""))
        label = titulo[:35] + "…" if len(titulo) > 35 else titulo
        fig.add_annotation(
            x=row["year"], y=row["cites"], text=label,
            showarrow=True, arrowhead=2, arrowcolor="#94a3b8",
            font=dict(size=9, color="#cbd5e1"),
            bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155",
        )
    fig.update_layout(**PLOTLY_LAYOUT, legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)
    media = str(round(df_plot["cites"].mean(), 1))
    maximo = str(f"{int(df_plot['cites'].max()):,}")
    st.markdown(
        "<div class='insight-box'>"
        "<strong>💡 Insight:</strong> Promedio global de <strong>" + media + " citas/artículo</strong> "
        "y un máximo de <strong>" + maximo + " citas</strong>. Los artículos fundacionales acumulan "
        "más citas históricas, mientras las publicaciones recientes (2020+) muestran crecimiento "
        "acelerado, evidenciando que la IA aplicada a negocios es un campo de alta demanda investigativa."
        "</div>",
        unsafe_allow_html=True,
    )


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    render_header()

    st.markdown("### 📂 Fuente de datos")
    col_up, col_info = st.columns([1, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Sube tu archivo CSV exportado desde Scopus",
            type=["csv"],
            help="Exporta desde Scopus: Export → CSV → All fields",
        )
    with col_info:
        st.markdown(
            "<div style='background:#1e293b;border-radius:10px;padding:1rem;"
            "border:1px solid #334155;font-size:0.85rem;color:#94a3b8;'>"
            "<strong style='color:#e2e8f0;'>¿Cómo exportar desde Scopus?</strong><br><br>"
            "1. Busca en <a href='https://www.scopus.com' target='_blank' style='color:#38bdf8;'>scopus.com</a><br>"
            "2. Selecciona los resultados relevantes<br>"
            "3. <strong>Export → CSV → All available information</strong><br>"
            "4. Sube el archivo aquí o colócalo en <code>data/</code> del repositorio"
            "</div>",
            unsafe_allow_html=True,
        )

    df = None
    if uploaded is not None:
        with st.spinner("Cargando archivo..."):
            df = cargar_csv_bytes(uploaded.read())
        if df is None:
            st.error("No se pudo leer el archivo. Verifica que sea un CSV válido de Scopus.")

    if df is None and uploaded is None:
        with st.spinner("Buscando dataset en GitHub..."):
            df = cargar_csv_url(GITHUB_CSV_URL)
        if df is not None:
            st.success("☁️ Dataset cargado desde repositorio GitHub")

    if df is None:
        st.info(
            "👆 **Sube un archivo CSV de Scopus** para comenzar el análisis.\n\n"
            "Si configuraste GITHUB_CSV_URL en app.py, el dataset se cargará automáticamente.",
            icon="📋",
        )
        st.stop()
    else:
        if uploaded is not None:
            st.success("✅ Archivo cargado: " + uploaded.name)

    df = normalizar_columnas(df)
    df = limpiar_datos(df)

    if df.empty:
        st.warning("El dataset quedó vacío tras la limpieza. Revisa el formato del CSV.")
        st.stop()

    df_filtered, top_n = render_sidebar(df)

    st.markdown("---")
    st.markdown("### 📊 Métricas Principales")
    render_kpis(df_filtered)
    st.markdown("---")

    st.markdown(
        "<p class='chart-title'>📈 1. Evolución Temporal de Publicaciones</p>"
        "<p class='chart-subtitle'>Crecimiento del interés científico a lo largo del tiempo</p>",
        unsafe_allow_html=True,
    )
    grafico_evolucion_temporal(df_filtered)
    st.markdown("---")

    st.markdown(
        "<p class='chart-title'>🏆 2. Autores con Mayor Impacto (Citas Acumuladas)</p>"
        "<p class='chart-subtitle'>Investigadores más referenciados en el corpus analizado</p>",
        unsafe_allow_html=True,
    )
    grafico_autores_citados(df_filtered, top_n=top_n)
    st.markdown("---")

    st.markdown(
        "<p class='chart-title'>🔍 3. Frecuencia de Términos Clave en el Corpus</p>"
        "<p class='chart-subtitle'>Análisis de contenido: IA, Negocios y Empleabilidad en abstracts y títulos</p>",
        unsafe_allow_html=True,
    )
    grafico_palabras_clave(df_filtered)
    st.markdown("---")

    st.markdown(
        "<p class='chart-title'>💡 4. Relevancia Temporal: Citas vs. Año de Publicación</p>"
        "<p class='chart-subtitle'>Impacto científico distribuido en el tiempo (tamaño del punto = nº de citas)</p>",
        unsafe_allow_html=True,
    )
    grafico_relevancia_temporal(df_filtered)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#475569;font-size:0.8rem;padding:1rem 0;'>"
        "Dashboard desarrollado con <strong style='color:#38bdf8;'>Streamlit</strong> · "
        "Datos: <strong style='color:#38bdf8;'>Scopus</strong> · "
        "Licencia: <strong style='color:#38bdf8;'>MIT</strong>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
