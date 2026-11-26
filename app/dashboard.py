"""Dashboard do VerIdade (Streamlit).

Consome a API (que lê o PostgreSQL alimentado pelo MQTT + Airflow) e atualiza sozinho a cada poucos segundos.
Por privacidade, nenhuma imagem é exibida: só previsões e métricas.
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
SERVICOS = {"API": f"{API}/health",
            "MLflow": os.getenv("MLFLOW_URL", "http://localhost:5000") + "/health",
            "Airflow": os.getenv("AIRFLOW_URL", "http://localhost:8080") + "/health"}
LINKS = {"API (docs)": "http://localhost:8000/docs", "MLflow": "http://localhost:5000", "Airflow": "http://localhost:8080"}
AZUL, LARANJA, CINZA = "#2E6F9E", "#D9822B", "#9BB7D4"

st.set_page_config(page_title="VerIdade · Monitoramento", page_icon="🛡️", layout="wide")


def get(rota, **params):
    try:
        r = requests.get(f"{API}{rota}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.title("🛡️ VerIdade")
    st.caption("Triagem de menores de idade em verificação facial · ECA Digital")
    st.subheader("Serviços")
    for nome, url in SERVICOS.items():
        try:
            ok = requests.get(url, timeout=2).ok
        except requests.RequestException:
            ok = False
        st.markdown(f"{'🟢' if ok else '🔴'} **{nome}**")
    st.subheader("Atalhos")
    for nome, url in LINKS.items():
        st.markdown(f"[{nome}]({url})")
    info = get("/modelo")
    if info:
        st.subheader("Modelo em produção")
        st.markdown(f"`{info['versao']}`")
        st.caption(f"Cabeças: {', '.join(str(c) for c in info['cabecas'])} anos · recall alvo {info['recall_alvo']}")
    intervalo = st.slider("Atualizar a cada (s)", 2, 30, 3)

st.title("Monitoramento em tempo real")
st.caption("Totens publicam embeddings via MQTT → ingestor grava no PostgreSQL → Airflow valida e chama a API (MultiAge) a cada minuto → predições e monitoramento aparecem aqui.")


@st.fragment(run_every=intervalo)
def painel():
    m = get("/metricas")
    if m is None:
        st.error("API indisponível. Confira se o container `veridade-api` está rodando.")
        return
    c, g = m["capturas"], m["geral"]

    # --------------------------------------------------------- indicadores
    k = st.columns(6)
    k[0].metric("Capturas recebidas", f"{c['recebidas']:,}".replace(",", "."))
    k[1].metric("Processadas", f"{c['processadas']:,}".replace(",", "."))
    k[2].metric("Na fila (pendentes)", c["pendentes"], help="Aguardando a próxima execução da DAG (a cada 1 min)")
    k[3].metric("Bloqueadas como < 18", pct(g["pct_bloqueados_18"]))
    k[4].metric("Falso adulto", pct(g["taxa_falso_adulto"]),
                help="Menores que passaram como adultos (meta: ~1%). Só é medível na simulação, onde a idade real é conhecida.")
    k[5].metric("Erro médio de idade", "–" if g["mae"] is None else f"{g['mae']:.1f} anos")

    # --------------------------------------------------------- fluxo e decisões
    e, d = st.columns([3, 2])
    with e:
        st.subheader("Capturas recebidas por minuto")
        pm = pd.DataFrame(m["por_minuto"])
        if not pm.empty:
            pm["minuto"] = pd.to_datetime(pm["minuto"])
            fig = px.area(pm, x="minuto", y="capturas", color_discrete_sequence=[AZUL])
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title="capturas")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Aguardando as primeiras capturas...")
    with d:
        st.subheader("Decisão por limiar legal")
        pl = pd.DataFrame(m["por_limiar"])
        if not pl.empty:
            pl["limiar"] = "< " + pl["limiar"].astype(str)
            pl["pct"] = pl["pct_bloqueados"].astype(float) * 100
            cores = [LARANJA if x in ("< 13", "< 16") else AZUL for x in pl["limiar"]]
            fig = go.Figure(go.Bar(x=pl["limiar"], y=pl["pct"], marker_color=cores,
                                   text=[f"{v:.0f}%" for v in pl["pct"]], textposition="outside"))
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="% classificados abaixo",
                              yaxis_range=[0, max(100, pl["pct"].max() * 1.15)])
            st.plotly_chart(fig, width="stretch")
            st.caption("Laranja: limiares da contribuição (13 anos: cadastro em redes sociais; 16 anos: consentimento).")

    # --------------------------------------------------------- saúde do modelo
    st.subheader("Saúde do modelo a cada execução da DAG")
    mon = pd.DataFrame(m["monitoramento"])
    if not mon.empty:
        mon["registrado_em"] = pd.to_datetime(mon["registrado_em"])
        mon = mon.sort_values("registrado_em")
        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mon["registrado_em"], y=mon["taxa_falso_adulto"] * 100, name="falso adulto", line_color=LARANJA))
            fig.add_hline(y=1, line_dash="dot", annotation_text="meta 1%")
            fig.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="alerta 3%")
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="% (últimas 300 predições)")
            st.plotly_chart(fig, width="stretch")
        with b:
            fig = px.line(mon, x="registrado_em", y="mae", color_discrete_sequence=[AZUL])
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title="MAE (anos)")
            st.plotly_chart(fig, width="stretch")
        alertas = mon.dropna(subset=["alerta"])
        if not alertas.empty:
            st.warning(f"⚠️ {len(alertas)} alerta(s) de monitoramento. Último: {alertas.iloc[-1]['alerta']}")
    else:
        st.info("O monitoramento aparece depois da primeira execução da DAG no Airflow.")

    # --------------------------------------------------------- últimas predições
    t, s = st.columns([3, 2])
    with t:
        st.subheader("Últimas verificações")
        pr = pd.DataFrame(get("/predicoes", limite=15) or [])
        if not pr.empty:
            pr["decisão"] = pr["menor_18"].map({True: "🔒 bloqueado (< 18)", False: "✅ liberado"})
            pr["acerto"] = ((pr["idade_real"] < 18) == pr["menor_18"]).map({True: "✔", False: "✘"})
            pr["horário"] = pd.to_datetime(pr["capturado_em"]).dt.strftime("%H:%M:%S")
            st.dataframe(pr[["horário", "totem_id", "idade_estimada", "decisão", "idade_real", "acerto"]]
                         .rename(columns={"totem_id": "totem", "idade_estimada": "idade estimada", "idade_real": "idade real"}),
                         hide_index=True, width="stretch")
    with s:
        st.subheader("Por totem")
        pt = pd.DataFrame(m["por_totem"])
        if not pt.empty:
            pt["% bloqueados"] = (pt["pct_bloqueados_18"].astype(float) * 100).round(1)
            st.dataframe(pt[["totem_id", "processadas", "% bloqueados"]].rename(columns={"totem_id": "totem"}),
                         hide_index=True, width="stretch")
        st.caption("Nenhuma imagem é armazenada ou exibida: os totens enviam apenas o embedding (privacidade desde a concepção).")


painel()
