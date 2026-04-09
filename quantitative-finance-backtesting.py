# =============================================================================
# ATIVIDADE 02 — ESTRATÉGIAS QUANTITATIVAS: CRUZAMENTO DE MÉDIAS MÓVEIS
# Disciplina: Análise de Ativos e Finanças Quantitativas
# =============================================================================
# O gráfico será aberto automaticamente no navegador padrão.
# =============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =============================================================================
# NOTA TÉCNICA SOBRE A METODOLOGIA
# =============================================================================
# Este projeto utiliza Retornos Logarítmicos devido às suas propriedades 
# estatísticas superiores: aditividade temporal e simetria de escala.
# A estratégia implementada busca capturar momentum através do cruzamento
# de médias (SMA), utilizando o conceito de Golden/Death Cross.
# =============================================================================

# =============================================================================
# SEÇÃO 1 — PARÂMETROS DA ANÁLISE
# =============================================================================
# Centralizamos todos os parâmetros configuráveis aqui para facilitar
# a reutilização do código com outros ativos ou janelas de tempo.

TICKER        = "PETR4.SA"   # Ativo analisado (Yahoo Finance format)
DATA_INICIO   = "2022-01-01" # Data inicial da série histórica
DATA_FIM      = datetime.today().strftime("%Y-%m-%d")  # Data atual

SMA_CURTA     = 20           # Janela da Média Móvel Simples curta (dias úteis)
SMA_LONGA     = 50           # Janela da Média Móvel Simples longa (dias úteis)

CAPITAL_INICIAL = 10_000.0   # Capital inicial para o backtesting (R$)

# Custo de transação (0.1% por operação)
CUSTO_TRANSACAO = 0.001

# =============================================================================
# SEÇÃO 2 — COLETA DE DADOS HISTÓRICOS (yfinance)
# =============================================================================
# O yfinance é um wrapper da API não-oficial do Yahoo Finance.
# Utilizamos o preço de fechamento ajustado ("Close"), que já desconta
# splits, bonificações e dividendos — essencial para séries longas.

print(f"\n{'='*60}")
print(f"  ANÁLISE QUANTITATIVA — {TICKER}")
print(f"  Período: {DATA_INICIO} até {DATA_FIM}")
print(f"{'='*60}\n")

print("[1/5] Baixando dados históricos...")

raw = yf.download(TICKER, start=DATA_INICIO, end=DATA_FIM, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

# Garantimos que trabalhamos apenas com a coluna de fechamento
# e removemos quaisquer valores ausentes (NaN) da série.
df = raw[["Close"]].copy()
df.dropna(inplace=True)

# Renomeamos para facilitar a leitura ao longo do código
df.rename(columns={"Close": "Preco"}, inplace=True)

print(f"    ✓ {len(df)} pregões carregados  |  "
      f"Primeiro: {df.index[0].date()}  |  Último: {df.index[-1].date()}")

# =============================================================================
# SEÇÃO 3 — RETORNO LOGARÍTMICO
# =============================================================================
# O retorno logarítmico é utilizado em finanças quantitativas porque:
#   1. É aditivo no tempo: r(0→2) = r(0→1) + r(1→2)
#   2. Segue distribuição aproximadamente normal (pressuposto do MFE)
#   3. É simétrico: um ganho de 100% e uma perda de 50% têm |r| idêntico
#
# Fórmula: r_t = ln(P_t / P_{t-1})
#
# Equivalente vetorial com pandas: df["Preco"].apply(np.log).diff()
# Usamos np.log(df["Preco"] / df["Preco"].shift(1)) para explicitar a fórmula.

print("\n[2/5] Calculando retornos logarítmicos e médias móveis...")

df["Retorno_Log"] = np.log(df["Preco"] / df["Preco"].shift(1))

# =============================================================================
# SEÇÃO 4 — MÉDIAS MÓVEIS SIMPLES (SMA)
# =============================================================================
# A Média Móvel Simples de janela k no instante t é:
#   SMA_k(t) = (1/k) * Σ P_{t-i}  para i = 0, ..., k-1
#
# Interpretação econômica:
#   • SMA curta (20d) ≈ tendência de curto prazo (~1 mês de pregões)
#   • SMA longa (50d) ≈ tendência de médio prazo (~2,5 meses de pregões)
#
# Os primeiros (k-1) valores serão NaN — comportamento esperado do rolling().

df[f"SMA_{SMA_CURTA}"] = df["Preco"].rolling(window=SMA_CURTA).mean()
df[f"SMA_{SMA_LONGA}"] = df["Preco"].rolling(window=SMA_LONGA).mean()

print(f"    ✓ SMA {SMA_CURTA} dias e SMA {SMA_LONGA} dias calculadas")

# =============================================================================
# SEÇÃO 5 — GERAÇÃO DE SINAIS (CROSSOVER)
# =============================================================================
# A estratégia de Cruzamento de Médias (Moving Average Crossover) é um
# sistema de seguimento de tendência clássico da análise técnica quantitativa.
#
# LÓGICA:
#   • Golden Cross: SMA curta cruza ACIMA da SMA longa → sinal de COMPRA (+1)
#     Indica que a tendência de curto prazo superou a de médio prazo.
#
#   • Death Cross:  SMA curta cruza ABAIXO da SMA longa → sinal de VENDA (-1)
#     Indica reversão da tendência de curto prazo para baixo.
#
#   • Sem cruzamento: posição mantida → sinal NEUTRO (0)
#
# Implementação:
#   - "Posicao" indica qual SMA está acima em cada instante (1 = curta > longa)
#   - "Sinal" captura apenas as MUDANÇAS de posição (diff != 0)

col_curta = f"SMA_{SMA_CURTA}"
col_longa = f"SMA_{SMA_LONGA}"

# 1 quando SMA curta > SMA longa; 0 caso contrário
# IMPORTANTE:
# Esta é uma estratégia LONG ONLY:
# • 1 = comprado
# • 0 = fora do mercado
# Não consideramos posições vendidas (short)
df["Posicao"] = np.where(df[col_curta] > df[col_longa], 1, 0)

# diff() detecta as transições: +1 = cruzamento para cima, -1 = cruzamento para baixo
df["Sinal"] = df["Posicao"].diff()

# Filtramos os dias com sinais reais
sinais = df[df["Sinal"] != 0].dropna(subset=["Sinal"])
compras = df[df["Sinal"] == 1]
vendas  = df[df["Sinal"] == -1]

print(f"    ✓ Sinais gerados: {len(compras)} compras | {len(vendas)} vendas")

# =============================================================================
# SEÇÃO 6 — BACKTESTING
# =============================================================================
# O backtesting simula como a estratégia teria performado historicamente,
# aplicando os sinais gerados sobre o capital inicial.
#
# ESTRATÉGIA CROSSOVER:
#   • Ficamos comprados (+1) quando SMA curta > SMA longa
#   • Ficamos fora do mercado (0) quando SMA curta < SMA longa
#   • Retorno da estratégia em cada dia = Posicao(t-1) × Retorno_Log(t)
#     (usamos shift(1) para evitar look-ahead bias — só agimos no dia seguinte)
#
# BUY AND HOLD:
#   • Compramos no primeiro pregão e seguramos até o final
#   • Retorno acumulado = exp(Σ Retorno_Log) = Preco_Final / Preco_Inicial
#
# Capital final = Capital_Inicial × exp(Σ retornos_da_estratégia)

print("\n[3/5] Executando backtesting...")

# Aplicação de custo de transação:
# Sempre que há mudança de posição (compra ou venda), aplicamos custo
# Identifica quando houve operação (compra ou venda)
# Identifica quando houve operação (compra ou venda)
df["Trade"] = (df["Sinal"] != 0).shift(1).fillna(0).astype(int)

# Custo em termos logarítmicos (compatível com retorno log)
df["Custo"] = df["Trade"] * np.log(1 - CUSTO_TRANSACAO)

# Retorno da estratégia com custo embutido
df["Retorno_Estrategia"] = df["Posicao"].shift(1) * df["Retorno_Log"] + df["Custo"]

# Retorno acumulado: soma dos logs é equivalente ao produto das razões de preço
df["Acum_Estrategia"] = np.exp(df["Retorno_Estrategia"].cumsum())
df["Acum_BuyHold"]    = np.exp(df["Retorno_Log"].cumsum())

# Normalizamos pelo capital inicial
df["Capital_Estrategia"] = CAPITAL_INICIAL * df["Acum_Estrategia"]
df["Capital_BuyHold"]    = CAPITAL_INICIAL * df["Acum_BuyHold"]

# Removemos NaNs do início (período de aquecimento das SMAs)
df_valido = df.dropna(subset=[col_curta, col_longa]).copy()

capital_final_estrategia = df_valido["Capital_Estrategia"].iloc[-1]
capital_final_bh          = df_valido["Capital_BuyHold"].iloc[-1]

retorno_estrategia = (capital_final_estrategia / CAPITAL_INICIAL - 1) * 100
retorno_bh         = (capital_final_bh          / CAPITAL_INICIAL - 1) * 100

# =============================================================================
# SEÇÃO 7 — ESTATÍSTICAS DE PERFORMANCE
# =============================================================================
# Métricas adicionais para avaliar a qualidade da estratégia:
#
# Volatilidade Anualizada: desvio padrão dos retornos diários × √252
#   (252 = número médio de pregões por ano na B3)
#
# Sharpe Ratio (aproximado): (Retorno_Anualizado - Taxa_Livre_Risco) / Volatilidade
#   Mede o retorno ajustado ao risco. Sharpe > 1 é considerado bom.

vol_anual_estrategia = df_valido["Retorno_Estrategia"].std() * np.sqrt(252)
vol_anual_bh         = df_valido["Retorno_Log"].std() * np.sqrt(252)

# Retorno anualizado: (1 + retorno_total)^(1/anos) - 1
n_anos = len(df_valido) / 252
ret_anual_estrategia = (capital_final_estrategia / CAPITAL_INICIAL) ** (1 / n_anos) - 1
ret_anual_bh         = (capital_final_bh          / CAPITAL_INICIAL) ** (1 / n_anos) - 1

# Taxa Selic aproximada como proxy da taxa livre de risco
TAXA_LIVRE_RISCO = 0.1075

sharpe_estrategia = (ret_anual_estrategia - TAXA_LIVRE_RISCO) / vol_anual_estrategia if vol_anual_estrategia > 0 else 0
sharpe_bh         = (ret_anual_bh         - TAXA_LIVRE_RISCO) / vol_anual_bh         if vol_anual_bh > 0 else 0

# =============================================================================
# DRAWDOWN (RISCO DE QUEDA)
# =============================================================================
# O drawdown mede a maior queda do capital a partir de um pico histórico.

df_valido = df_valido.copy()

df_valido["Pico"] = df_valido["Capital_Estrategia"].cummax()
df_valido["Drawdown"] = (df_valido["Capital_Estrategia"] - df_valido["Pico"]) / df_valido["Pico"]
drawdown_max = df_valido["Drawdown"].min()


print(f"\n{'─'*60}")
print(f"  RESULTADO DO BACKTESTING — Capital Inicial: R$ {CAPITAL_INICIAL:,.2f}")
print(f"{'─'*60}")
print(f"  {'Estratégia Crossover':30s}  {'Buy & Hold':>12s}")
print(f"  {'Capital Final':30s}  R$ {capital_final_estrategia:>9,.2f}  |  R$ {capital_final_bh:>9,.2f}")
print(f"  {'Retorno Total':30s}  {retorno_estrategia:>+9.2f}%  |  {retorno_bh:>+9.2f}%")
print(f"  {'Retorno Anualizado':30s}  {ret_anual_estrategia*100:>+9.2f}%  |  {ret_anual_bh*100:>+9.2f}%")
print(f"  {'Volatilidade Anualizada':30s}  {vol_anual_estrategia*100:>9.2f}%  |  {vol_anual_bh*100:>9.2f}%")
print(f"  {'Sharpe Ratio (aprox.)':30s}  {sharpe_estrategia:>+9.4f}  |  {sharpe_bh:>+9.4f}")
print(f"  {'Nº de Sinais de Compra':30s}  {len(compras):>9d}")
print(f"  {'Nº de Sinais de Venda':30s}  {len(vendas):>9d}")
print(f"{'─'*60}\n")

# =============================================================================
# SEÇÃO 8 — VISUALIZAÇÃO COM PLOTLY
# =============================================================================
# Criamos um dashboard com 3 subplots empilhados verticalmente:
#   1. Gráfico principal: preço + SMAs + marcadores de compra/venda
#   2. Retornos logarítmicos diários (histograma de barras)
#   3. Curva de patrimônio: estratégia vs. buy and hold

print("[4/5] Gerando visualização interativa com Plotly...")

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[0.45, 0.18, 0.22, 0.15],
    subplot_titles=(
    f"<b>{TICKER}</b> — Preço, SMA {SMA_CURTA} e SMA {SMA_LONGA} com Sinais de Cruzamento",
    "Retorno Logarítmico Diário",
    "Curva de Patrimônio — Backtesting",
    "Drawdown (Risco de Queda)"
    )
)

# ── SUBPLOT 1: Preço e Médias Móveis ─────────────────────────────────────────

fig.add_trace(go.Scatter(
    x=df.index, y=df["Preco"],
    name="Preço Fechamento",
    line=dict(color="#4A90D9", width=1.5),
    hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Preço: R$ %{y:.2f}<extra></extra>"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df[col_curta],
    name=f"SMA {SMA_CURTA} (curta)",
    line=dict(color="#F5A623", width=1.8, dash="solid"),
    hovertemplate=f"SMA {SMA_CURTA}: R$ %{{y:.2f}}<extra></extra>"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df[col_longa],
    name=f"SMA {SMA_LONGA} (longa)",
    line=dict(color="#E05C4B", width=1.8, dash="dot"),
    hovertemplate=f"SMA {SMA_LONGA}: R$ %{{y:.2f}}<extra></extra>"
), row=1, col=1)

# Marcadores de COMPRA (Golden Cross) — triângulo verde apontando para cima
fig.add_trace(go.Scatter(
    x=compras.index, y=compras["Preco"],
    name="Sinal de COMPRA (Golden Cross)",
    mode="markers",
    marker=dict(symbol="triangle-up", color="#27AE60", size=12, line=dict(color="white", width=1)),
    hovertemplate="<b>COMPRA</b><br>%{x|%d/%m/%Y}<br>R$ %{y:.2f}<extra></extra>"
), row=1, col=1)

# Marcadores de VENDA (Death Cross) — triângulo vermelho apontando para baixo
fig.add_trace(go.Scatter(
    x=vendas.index, y=vendas["Preco"],
    name="Sinal de VENDA (Death Cross)",
    mode="markers",
    marker=dict(symbol="triangle-down", color="#E74C3C", size=12, line=dict(color="white", width=1)),
    hovertemplate="<b>VENDA</b><br>%{x|%d/%m/%Y}<br>R$ %{y:.2f}<extra></extra>"
), row=1, col=1)

# ── SUBPLOT 2: Retornos Logarítmicos ─────────────────────────────────────────
# Colorimos as barras: verde para retornos positivos, vermelho para negativos
# Isso facilita a visualização da distribuição de ganhos e perdas diárias.

cores_retorno = ["#27AE60" if v >= 0 else "#E74C3C"
                 for v in df["Retorno_Log"].fillna(0)]

fig.add_trace(go.Bar(
    x=df.index, y=df["Retorno_Log"],
    name="Retorno Log Diário",
    marker_color=cores_retorno,
    opacity=0.75,
    hovertemplate="%{x|%d/%m/%Y}<br>r_t = %{y:.4f}<extra></extra>"
), row=2, col=1)

# Linha de referência em zero
fig.add_hline(y=0, line_color="rgba(200,200,200,0.5)", line_width=1, row=2, col=1)

# ── SUBPLOT 3: Curva de Patrimônio ────────────────────────────────────────────
# Compara diretamente o desempenho acumulado das duas estratégias,
# mostrando o valor do portfólio em reais ao longo do tempo.

fig.add_trace(go.Scatter(
    x=df_valido.index, y=df_valido["Capital_Estrategia"],
    name=f"Crossover SMA ({retorno_estrategia:+.1f}%)",
    line=dict(color="#8E44AD", width=2),
    fill="tozeroy", fillcolor="rgba(142,68,173,0.08)",
    hovertemplate="<b>Crossover</b>: R$ %{y:,.2f}<extra></extra>"
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=df_valido.index, y=df_valido["Capital_BuyHold"],
    name=f"Buy & Hold ({retorno_bh:+.1f}%)",
    line=dict(color="#1ABC9C", width=2, dash="dash"),
    hovertemplate="<b>Buy & Hold</b>: R$ %{y:,.2f}<extra></extra>"
), row=3, col=1)

# Linha de referência no capital inicial
fig.add_hline(
    y=CAPITAL_INICIAL,
    line_color="rgba(200,200,200,0.6)", line_width=1, line_dash="dot",
    row=3, col=1
)

# ── SUBPLOT 4: Drawdown ─────────────────────────────────────────────

fig.add_trace(go.Scatter(
    x=df_valido.index,
    y=df_valido["Drawdown"],
    name="Drawdown",
    line=dict(color="#FF4D4D", width=1.5),
    fill="tozeroy",
    fillcolor="rgba(255,77,77,0.2)",
    hovertemplate="%{x|%d/%m/%Y}<br>Drawdown: %{y:.2%}<extra></extra>"
), row=4, col=1)

df_valido["Pico_BH"] = df_valido["Capital_BuyHold"].cummax()
df_valido["Drawdown_BH"] = (df_valido["Capital_BuyHold"] - df_valido["Pico_BH"]) / df_valido["Pico_BH"]

fig.add_trace(go.Scatter(
    x=df_valido.index,
    y=df_valido["Drawdown_BH"],
    name="Drawdown Buy & Hold",
    line=dict(color="#1ABC9C", width=1.5, dash="dash"),
    fill="tozeroy",
    fillcolor="rgba(26,188,156,0.08)",
    hovertemplate="%{x|%d/%m/%Y}<br>Drawdown B&H: %{y:.2%}<extra></extra>"
), row=4, col=1)

# Linha de referência em 0
fig.add_hline(y=0, line_color="rgba(200,200,200,0.5)", row=4, col=1)

# ── LAYOUT GERAL ─────────────────────────────────────────────────────────────

fig.update_layout(
    title=dict(
        text=(f"Estratégia Quantitativa — Cruzamento de Médias Móveis | {TICKER} | "
              f"{DATA_INICIO} a {DATA_FIM}"),
        font=dict(size=15, color="#ECEFF1"),
        x=0.01
    ),
    template="plotly_dark",
    height=1100,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01,
        xanchor="left", x=0,
        font=dict(size=11),
        bgcolor="rgba(0,0,0,0)"
    ),
    hovermode="x unified",
    paper_bgcolor="#0D1117",
    plot_bgcolor="#161B22",
    margin=dict(l=60, r=30, t=100, b=40),
    font=dict(color="#C9D1D9", family="Courier New, monospace")
)

# Configuração dos eixos Y com prefixo R$
# eixo 1 → preço
fig.update_yaxes(
    title_text="Preço (R$)", row=1, col=1,
    gridcolor="rgba(255,255,255,0.05)", tickprefix="R$ "
)
# eixo 2 → retorno
fig.update_yaxes(
    title_text="Retorno Log (%)", row=2, col=1,
    gridcolor="rgba(255,255,255,0.05)", tickformat=".3f"
)
# eixo 3 → capital
fig.update_yaxes(
    title_text="Capital (R$)", row=3, col=1,
    gridcolor="rgba(255,255,255,0.05)", tickprefix="R$ "
)
# eixo 4 → drawdown
fig.update_yaxes(
    title_text="Drawdown (%)",
    row=4, col=1,
    tickformat=".0%",
    gridcolor="rgba(255,255,255,0.05)"
)

# Eixo X somente no gráfico inferior (shared_xaxes=True)
fig.update_xaxes(
    title_text="Data", row=3, col=1,
    gridcolor="rgba(255,255,255,0.05)",
    rangeslider=dict(visible=False)
)

# Anotação com resumo das métricas no gráfico
anotacao = (
    f"Capital Final — Crossover: R$ {capital_final_estrategia:,.2f}  |  "
    f"Buy & Hold: R$ {capital_final_bh:,.2f}  |  "
    f"Sharpe Crossover: {sharpe_estrategia:.2f}  |  "
    f"Drawdown Máx.: {drawdown_max:.2%}"
)
fig.add_annotation(
    text=anotacao,
    xref="paper", yref="paper", x=0.01, y=-0.03,
    showarrow=False, font=dict(size=10, color="#8B949E"),
    align="left"
)

# =============================================================================
# SEÇÃO 9 — EXIBIÇÃO DO GRÁFICO
# =============================================================================
# fig.show() abre o gráfico no navegador padrão do sistema.
# Ideal para execução via terminal no VS Code.

print("[5/5] Abrindo gráfico no navegador...\n")
fig.show()

print("✓ Análise concluída com sucesso!")
print(f"  Ticker analisado : {TICKER}")
print(f"  Período          : {DATA_INICIO} a {DATA_FIM}")
print(f"  Crossover final  : R$ {capital_final_estrategia:,.2f}  ({retorno_estrategia:+.2f}%)")
print(f"  Buy & Hold final : R$ {capital_final_bh:,.2f}  ({retorno_bh:+.2f}%)\n")
print(f"  {'Drawdown Máximo':30s}  {drawdown_max:>9.2%}")