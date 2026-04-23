# =============================================================================
# ATIVIDADE 02 — SIMULADOR DE BACKTESTING QUANTITATIVO
# Estratégia: Cruzamento de Médias Móveis Simples (SMA Crossover)
# Disciplina: Análise de Ativos e Finanças Quantitativas
# =============================================================================
#
# DESCRIÇÃO GERAL DO PROJETO:
#   Este script implementa um simulador completo de backtesting para a
#   estratégia de Cruzamento de Médias Móveis (Moving Average Crossover),
#   amplamente utilizada em análise técnica quantitativa. O sistema evoluiu
#   de um script estático para uma ferramenta interativa global, cobrindo
#   ações de qualquer bolsa mundial, ETFs e criptomoedas.
#
# ENTRADAS DO USUÁRIO (via terminal):
#   1. Ativo    — nome da empresa ou ticker (ex: "Petrobras", "AAPL", "BTC-USD")
#   2. Capital  — valor inicial da simulação em reais (ex: 5000)
#   3. Data     — ponto de partida do backtesting no formato ISO 8601
#   4. Taxa     — taxa livre de risco anual em % (ex: 10.75 para Selic)
#
# SAÍDAS GERADAS:
#   • Tabela de performance no terminal (retorno, Sharpe, drawdown, etc.)
#   • Dashboard interativo em HTML (4 painéis via Plotly)
#
# DEPENDÊNCIAS:
#   pip install pandas numpy plotly yfinance
#
# EXECUÇÃO:
#   python atividade02_estrategia_quantitativa.py
#   → O gráfico será aberto automaticamente no navegador padrão.
#
# =============================================================================
# REGISTRO DE MELHORIAS (v2) — RESUMO TÉCNICO
# =============================================================================
#
#   [M1] TAXA LIVRE DE RISCO INTERATIVA
#        A TAXA_LIVRE_RISCO era hardcoded (0.1075). Agora é coletada como
#        quarta entrada do usuário via coletar_taxa_livre_risco(), com
#        validação de range e padrão Selic (10.75%).
#
#   [M2] FATOR DE ANUALIZAÇÃO INTELIGENTE (252 vs 365)
#        detectar_fator_anual(ticker) inspeciona o sufixo do ticker para
#        identificar criptoativos (sufixos -USD, -BTC, -ETH, -BRL, -EUR,
#        -USDT). Cripto → √365; ações/ETFs → √252. Isso elimina o erro
#        sistemático de ~20% na volatilidade de criptoativos.
#
#   [M3] DRAWDOWN DO BUY & HOLD CALCULADO NA SEÇÃO 7
#        O cálculo de Pico_BH e Drawdown_BH foi movido da Seção 8
#        (visualização) para a Seção 7 (métricas), junto ao MDD da
#        estratégia. Ambos os drawdowns agora aparecem na tabela do terminal.
#
#   [M4] CÁLCULO DE RETORNOS ACUMULADOS SOMENTE SOBRE df_valido
#        O cumsum()/exp() era calculado sobre o df completo (incluindo o
#        período de aquecimento das SMAs, que contém NaN em Retorno_Log).
#        NaN propagado no cumsum() distorce os primeiros valores da curva
#        de capital. A correção isola df_valido (pós-dropna) ANTES de
#        calcular os acumulados, garantindo que Capital(t=0) == CAPITAL_INICIAL.
#
#   [M5] ELIMINAÇÃO DE SettingWithCopyWarning
#        Todas as atribuições de novas colunas em DataFrames derivados
#        (df_valido) usam .loc[:, coluna] = valor para sinalizar ao pandas
#        que a escrita é intencional sobre a cópia, não sobre uma fatia
#        temporária. O .copy() permanece nos pontos de criação do df_valido.
#
#   [M6] ORDEM DE DEFINIÇÃO DE VARIÁVEIS CORRIGIDA
#        TAXA_LIVRE_RISCO e FATOR_ANUAL agora são definidas na Seção 1
#        (bloco de configuração interativo), antes de qualquer cálculo que
#        as utilize. Isso elimina os NameError observados na versão anterior.
#
# =============================================================================


# =============================================================================
# NOTA TÉCNICA SOBRE A METODOLOGIA — FUNDAMENTOS TEÓRICOS
# =============================================================================
#
# ── 1. RETORNO LOGARÍTMICO ───────────────────────────────────────────────────
#
#   Em finanças quantitativas, existem dois tipos principais de retorno:
#
#   a) Retorno Aritmético (simples):  r_t = (P_t - P_{t-1}) / P_{t-1}
#   b) Retorno Logarítmico (log):     r_t = ln(P_t / P_{t-1})
#
#   Este projeto adota o retorno logarítmico por três razões fundamentais:
#
#   ADITIVIDADE TEMPORAL:
#     O retorno total de um período de múltiplos dias é simplesmente a SOMA
#     dos retornos diários, o que não ocorre com o retorno aritmético:
#       r(0→2) = r(0→1) + r(1→2)   [log: sempre válido]
#     Isso simplifica o cálculo de retornos acumulados via cumsum() + exp().
#
#   SIMETRIA DE ESCALA:
#     Um ganho de 100% (dobrar o preço) e uma perda de 50% (metade do preço)
#     têm magnitudes logarítmicas iguais e opostas: |ln(2)| = |ln(0.5)|.
#     O retorno aritmético não possui essa simetria, o que distorceria análises
#     de risco ao tratar assimetricamente ganhos e perdas.
#
#   NORMALIDADE APROXIMADA:
#     Pelos resultados do Teorema Central do Limite, retornos logarítmicos
#     tendem a seguir uma distribuição aproximadamente normal para séries
#     suficientemente longas — pressuposto central do Modelo de Finanças
#     Estocásticas (MFE) e de métricas como o Índice Sharpe.
#
# ── 2. PREÇO DE FECHAMENTO AJUSTADO ─────────────────────────────────────────
#
#   O Yahoo Finance disponibiliza o preço ajustado ("Adj Close"), que desconta
#   retroativamente todos os eventos corporativos ocorridos no histórico:
#
#   • Dividendos: sem ajuste, o preço cai no dia "ex-dividendo", gerando
#     um retorno negativo artificial que não representa perda real de valor
#     para o acionista que recebe o provento.
#   • Splits (desdobramentos): uma ação que vale R$ 100 passa a valer R$ 25
#     após um split 1:4; sem ajuste, isso apareceria como queda de 75%.
#   • Grupamentos (inplit): o inverso do split, com efeito oposto.
#
#   O uso do preço ajustado garante que a série histórica reflita apenas a
#   variação real do valor econômico do ativo — essencial para backtesting
#   com horizonte de múltiplos anos.
#
# ── 3. ESTRATÉGIA DE CRUZAMENTO DE MÉDIAS (SMA CROSSOVER) ───────────────────
#
#   A Média Móvel Simples (SMA) suaviza o ruído de curto prazo da série de
#   preços, revelando a tendência subjacente do ativo. O cruzamento entre
#   duas SMAs de janelas diferentes gera sinais de entrada e saída:
#
#   GOLDEN CROSS (Cruzamento Dourado):
#     SMA curta (20d) cruza ACIMA da SMA longa (50d) → sinal de COMPRA
#     Interpretação: a tendência de curto prazo superou a de médio prazo,
#     sinalizando aceleração positiva (momentum de alta).
#
#   DEATH CROSS (Cruzamento da Morte):
#     SMA curta (20d) cruza ABAIXO da SMA longa (50d) → sinal de VENDA
#     Interpretação: o momentum de curto prazo reverteu para baixo,
#     sugerindo o início de uma tendência de queda sustentada.
#
#   Esta é uma estratégia de SEGUIMENTO DE TENDÊNCIA (trend-following),
#   adequada para mercados com tendências persistentes. Em mercados laterais
#   (sideways/ranging), ela tende a gerar falsos sinais (whipsaws).
#
# =============================================================================


import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys


# =============================================================================
# SEÇÃO 1 — MOTOR DE BUSCA E PARÂMETROS INTERATIVOS DA ANÁLISE
# =============================================================================
#
# FILOSOFIA DO DESIGN INTERATIVO:
#   A centralização das entradas do usuário no início do script segue o
#   princípio de "configuração antes da execução": todos os parâmetros
#   variáveis são coletados e validados ANTES de qualquer processamento
#   de dados. Isso evita que o script consuma tempo baixando dados apenas
#   para falhar por um parâmetro inválido no final.
#
# ── MOTOR DE BUSCA SEMÂNTICA GLOBAL ─────────────────────────────────────────
#
#   O sistema de resolução de tickers opera em dois estágios complementares:
#
#   ESTÁGIO 1 — TICKER DIRETO (rápido, ~1s):
#     Testa a entrada como ticker exato via yf.Ticker().history(period="5d").
#     Usa history() em vez de .info() porque:
#       • Retorna apenas 1 linha de OHLCV: mais eficiente na rede
#       • Não dispara logs de erro 404 no console para tickers inválidos
#       • Funciona uniformemente para ações, cripto, índices e ETFs
#     Testa duas variantes em sequência: a entrada exata e com sufixo ".SA"
#     (para ativos da B3 digitados sem sufixo, como "PETR4" → "PETR4.SA").
#
#   ESTÁGIO 2 — BUSCA SEMÂNTICA (fallback, ~2-4s):
#     Ativado quando o estágio 1 falha. Consulta o endpoint de busca do
#     Yahoo Finance via yf.Search(), que aceita nomes em linguagem natural
#     ("Petrobras", "Apple", "Bitcoin") e retorna uma lista de candidatos
#     ordenados por relevância semântica e volume de negociação.
#     O sistema itera pelos candidatos e seleciona o PRIMEIRO com histórico
#     de preços válido — geralmente o ativo de maior liquidez.
#
#   COBERTURA GLOBAL DE ATIVOS:
#     Ações BR (B3)    → "Petrobras", "PETR4", "Vale", "WEGE3"
#     Ações EUA        → "Apple", "AAPL", "Tesla", "MSFT"
#     Ações Intl.      → "Toyota" (7203.T), "ASML" (ASML.AS)
#     ETFs             → "SPY", "QQQ", "BOVA11"
#     Criptomoedas     → "Bitcoin", "BTC-USD", "ETH-USD"
#     Índices          → "^GSPC" (S&P 500), "^BVSP" (Ibovespa)
#
#   CORREÇÃO AUTOMÁTICA DE TICKERS FRACIONÁRIOS (B3):
#     Na B3, ações negociadas em lote fracionário têm o sufixo "F" no ticker
#     (ex: MGLU3F.SA, PETR4F.SA). Esses tickers possuem histórico de dados
#     muito reduzido no Yahoo Finance, inviabilizando o cálculo das SMAs.
#     O sistema detecta o padrão "F.SA" e remove o "F" automaticamente,
#     redirecionando para o lote padrão (MGLU3F.SA → MGLU3.SA).
#
# ── [M1] TAXA LIVRE DE RISCO INTERATIVA ─────────────────────────────────────
#
#   PROBLEMA ANTERIOR:
#     TAXA_LIVRE_RISCO = 0.1075 era um valor hardcoded fixo na Seção 7,
#     após o bloco de configuração. Para ativos americanos (AAPL, SPY),
#     a Selic brasileira é um benchmark inadequado — o correto seria usar
#     a Fed Funds Rate ou o yield do T-bill de 3 meses (~5% em 2024).
#     Para cripto, muitos analistas usam 0% por não haver ativo livre de
#     risco com liquidez equivalente.
#
#   SOLUÇÃO:
#     A função coletar_taxa_livre_risco() coleta o valor como quarta
#     entrada do usuário, com sugestões contextuais por tipo de mercado.
#     O valor é expresso em % (ex: "10.75") e convertido para decimal
#     internamente (÷ 100). A definição acontece no bloco de configuração
#     (Seção 1), ANTES de qualquer cálculo que use a variável — eliminando
#     o NameError que ocorria quando a variável era referenciada antes
#     de ser definida.
#
# ── [M2] FATOR DE ANUALIZAÇÃO INTELIGENTE ───────────────────────────────────
#
#   PROBLEMA ANTERIOR:
#     O código usava sempre np.sqrt(252) para anualizar volatilidade e
#     retornos, mesmo para criptoativos. Criptos negociam 365 dias/ano
#     (incluindo fins de semana e feriados), enquanto bolsas de ações
#     operam ~252 dias úteis/ano. Usar √252 para cripto subestima a
#     volatilidade em ~20% (√252 ≈ 15.87 vs √365 ≈ 19.10), distorcendo
#     tanto o Sharpe quanto a volatilidade anualizada reportada.
#
#   SOLUÇÃO:
#     detectar_fator_anual(ticker) verifica se o ticker termina com
#     sufixos típicos de pares cripto (-USD, -BTC, -ETH, -BRL, -EUR,
#     -USDT). Se sim, retorna 365 e exibe um aviso ao usuário. Caso
#     contrário, retorna 252 (padrão para ações e ETFs). O fator é
#     calculado na Seção 1 e usado como FATOR_ANUAL em toda a Seção 7.

def normalizar_ticker_fracionario(ticker: str) -> tuple[str, bool]:
    """
    Detecta e corrige tickers fracionários da B3.

    Na B3, ações fracionárias recebem o sufixo "F" imediatamente antes do
    ".SA" (ex: MGLU3F.SA, PETR4F.SA). Esses tickers têm histórico de dados
    muito limitado no Yahoo Finance e causariam falha no backtesting por
    insuficiência de pregões. Esta função remove o "F" automaticamente,
    redirecionando a análise para o lote padrão correspondente.

    Exemplos:
        "MGLU3F.SA"  → ("MGLU3.SA",  True)   # fracionário corrigido
        "PETR4F.SA"  → ("PETR4.SA",  True)   # fracionário corrigido
        "VALE3.SA"   → ("VALE3.SA",  False)  # lote padrão, sem alteração
        "BTC-USD"    → ("BTC-USD",   False)  # cripto, sem alteração

    Args:
        ticker: Ticker bruto retornado pela busca ou digitado pelo usuário.

    Returns:
        tuple: (ticker_normalizado, foi_corrigido)
    """
    if ticker.upper().endswith("F.SA"):
        ticker_corrigido = ticker[:-4] + ".SA"
        return ticker_corrigido, True
    return ticker, False


def buscar_ticker_semantico() -> tuple[str, str]:
    """
    Motor de busca semântica GLOBAL para resolução de tickers.

    Aceita qualquer ativo disponível no Yahoo Finance: ações de qualquer
    bolsa mundial, ETFs, criptomoedas e índices. Implementa um sistema de
    fallback em dois estágios:
        1. Validação como ticker direto (rápido, sem chamada extra à API)
        2. Busca semântica global via yf.Search (para nomes em linguagem natural)

    Aplica correção automática de tickers fracionários da B3 em ambos
    os estágios.

    Returns:
        tuple: (ticker_final, nome_empresa)
               Ex: ("AAPL",     "Apple Inc.")
                   ("BTC-USD",  "Bitcoin USD")
                   ("PETR4.SA", "Petróleo Brasileiro S.A. - Petrobras")

    Raises:
        SystemExit: Em caso de entrada vazia, ausência de resultados válidos
                    ou falha de conexão com a API do Yahoo Finance.
    """
    print(f"\n{'='*60}")
    print("  SIMULADOR DE BACKTESTING QUANTITATIVO")
    print("  Estratégia: Cruzamento de Médias Móveis (SMA Crossover)")
    print(f"{'='*60}")
    print("\n  Busca global: ações, ETFs, criptomoedas e índices.")
    print("  Exemplos:")
    print("    'Petrobras' / 'PETR4'  →  PETR4.SA  (B3)")
    print("    'Apple' / 'AAPL'       →  AAPL      (NASDAQ)")
    print("    'Bitcoin' / 'BTC-USD'  →  BTC-USD   (Cripto)\n")

    entrada = input("  [1/4] Digite o nome do ativo ou ticker: ").strip()

    if not entrada:
        print("\n  [ERRO] Nenhuma entrada informada. Encerrando.")
        sys.exit(1)

    entrada_norm = entrada.upper().strip()
    candidatos_diretos: list[str] = [entrada_norm]

    if "." not in entrada_norm and "-" not in entrada_norm and "^" not in entrada_norm:
        candidatos_diretos.append(entrada_norm + ".SA")

    for candidato in candidatos_diretos:
        ticker_testado, foi_corrigido = normalizar_ticker_fracionario(candidato)

        if foi_corrigido:
            print(f"\n  [AVISO] Ticker fracionário detectado: '{candidato}'")
            print(f"  [AVISO] Convertido automaticamente para lote padrão: '{ticker_testado}'")

        try:
            t_obj = yf.Ticker(ticker_testado)
            hist  = t_obj.history(period="5d", raise_errors=False)

            if hist is not None and not hist.empty:
                try:
                    info         = t_obj.info
                    nome_oficial = (info.get("longName")
                                    or info.get("shortName")
                                    or ticker_testado)
                except Exception:
                    nome_oficial = ticker_testado

                print(f"\n  [BUSCA] Ticker direto confirmado: {ticker_testado}")
                print(f"  [BUSCA] Ativo: {nome_oficial}\n")
                return ticker_testado, nome_oficial

        except Exception:
            pass

    print(f"\n  [BUSCA] '{entrada}' não confirmado como ticker direto.")
    print(f"  [BUSCA] Realizando busca semântica global...")

    try:
        resultado = yf.Search(entrada, max_results=10, news_count=0)
        quotes    = resultado.quotes
    except Exception as e:
        print(f"\n  [ERRO] Falha na conexão com a API de busca do Yahoo Finance.")
        print(f"  Detalhe técnico: {e}")
        print("  Dica: verifique sua conexão ou tente digitar o ticker diretamente.\n")
        sys.exit(1)

    if not quotes:
        print(f"\n  [ERRO] Nenhum resultado encontrado para '{entrada}'.")
        print("  Tente um nome diferente ou o ticker exato (ex: AAPL, BTC-USD, PETR4).\n")
        sys.exit(1)

    print(f"  [BUSCA] Validando {len(quotes)} candidatos encontrados...\n")

    for quote in quotes:
        simbolo_raw = str(quote.get("symbol", ""))
        nome_raw    = quote.get("longname") or quote.get("shortname") or simbolo_raw
        quote_type  = quote.get("quoteType", "UNKNOWN")

        if not simbolo_raw:
            continue

        simbolo, foi_corrigido = normalizar_ticker_fracionario(simbolo_raw)

        if foi_corrigido:
            print(f"  [AVISO] Fracionário na busca: '{simbolo_raw}' → corrigido para '{simbolo}'")

        try:
            t_obj = yf.Ticker(simbolo)
            hist  = t_obj.history(period="5d", raise_errors=False)

            if hist is not None and not hist.empty:
                outros = [q for q in quotes if q.get("symbol") != simbolo_raw]
                if outros:
                    print(f"  [BUSCA] Outros candidatos (use o ticker diretamente para escolher):")
                    for q in outros[:4]:
                        sym  = q.get("symbol", "?")
                        nome = q.get("longname") or q.get("shortname", "?")
                        tipo = q.get("quoteType", "?")
                        print(f"           {sym:18s} | {tipo:14s} | {nome}")
                    print()

                print(f"  [BUSCA] Selecionado: {nome_raw} ({simbolo}) — {quote_type}")
                print()
                return simbolo, nome_raw

        except Exception:
            continue

    print(f"\n  [ERRO] Nenhum resultado válido com histórico de preços para '{entrada}'.")
    print("  Dica: tente o ticker exato (ex: AAPL, BTC-USD, PETR4.SA, 7203.T).\n")
    sys.exit(1)


def coletar_capital() -> float:
    """
    Coleta e valida o capital inicial da simulação via terminal.

    Trata strings com formatação local (pontos e vírgulas) antes de converter
    para float, garantindo compatibilidade com padrões numéricos brasileiros
    (10.000,00) e americanos (10,000.00).

    Returns:
        float: Capital inicial validado. Padrão R$ 10.000,00 se inválido.
    """
    # -------------------------------------------------------------------------
    # LIMPEZA DA STRING DE ENTRADA:
    #   Padrão BR: "10.000,50" → remove "." → "10000,50" → troca "," por "." → 10000.50
    #   Padrão US: "10,000.50" → remove "," via replace → precisa de lógica adicional
    #   Simplificação adotada: remove pontos E substitui vírgulas por ponto,
    #   o que cobre corretamente o padrão brasileiro (o mais comum neste contexto).
    # -------------------------------------------------------------------------
    print(f"{'─'*60}")
    entrada_cap = input("  [2/4] Capital inicial (ex: 5000) [Padrão R$ 10.000]: ").strip()

    if not entrada_cap:
        return 10_000.0

    try:
        capital = float(entrada_cap.replace('.', '').replace(',', '.'))
        if capital <= 0:
            raise ValueError("Capital deve ser positivo.")
        return capital
    except ValueError:
        print("  [AVISO] Valor inválido. Usando R$ 10.000,00 como padrão.")
        return 10_000.0


def coletar_data_inicio() -> str:
    """
    Coleta e valida a data de início do backtesting via terminal.

    Aplica validação rigorosa no formato ISO 8601 (AAAA-MM-DD) usando
    datetime.strptime, que verifica tanto o formato quanto a existência
    da data (rejeita, por exemplo, 2022-02-30).

    Returns:
        str: Data validada no formato "AAAA-MM-DD". Padrão "2022-01-01".
    """
    # -------------------------------------------------------------------------
    # A data de início define o REGIME DE MERCADO incluído no backtesting.
    # Diferentes períodos expõem a estratégia a condições distintas:
    #
    #   2020-01-01: inclui o crash da COVID-19 (volatilidade extrema) e a
    #               recuperação em "V" — testa a resiliência da estratégia.
    #   2022-01-01: ciclo de alta de juros global (Fed e Banco Central).
    #   2023-01-01: período de recuperação com menor volatilidade.
    #
    # Um sistema quantitativo robusto deve apresentar resultados aceitáveis
    # em MÚLTIPLOS recortes temporais. Se a estratégia só funciona em um
    # período específico, pode haver overfitting ao histórico observado —
    # fenômeno conhecido como "data snooping bias" na literatura.
    # -------------------------------------------------------------------------
    while True:
        print(f"{'─'*60}")
        print("  Cenários sugeridos:")
        print("    2020-01-01 → inclui crash COVID-19 (alta volatilidade)")
        print("    2022-01-01 → ciclo de alta de juros global [padrão]")
        print("    2023-01-01 → recuperação dos mercados (menor volatilidade)")
        entrada_data = input("  [3/4] Data de início AAAA-MM-DD [Padrão 2022-01-01]: ").strip()

        if not entrada_data:
            return "2022-01-01"

        try:
            datetime.strptime(entrada_data, "%Y-%m-%d")
            return entrada_data
        except ValueError:
            print("  [ERRO] Data inválida ou fora do formato. Use AAAA-MM-DD (ex: 2022-01-01).")


# -----------------------------------------------------------------------------
# [M1] NOVA FUNÇÃO — TAXA LIVRE DE RISCO INTERATIVA
# -----------------------------------------------------------------------------
def coletar_taxa_livre_risco() -> float:
    """
    [MELHORIA M1] Coleta a taxa livre de risco anual via terminal.

    MOTIVAÇÃO:
        A taxa livre de risco adequada varia por tipo de ativo e região:
          • Ações BR (B3): Selic (~10.75% em 2024)
          • Ações EUA    : Fed Funds Rate / T-bill 3M (~5.25% em 2024)
          • Cripto       : 0% (consenso comum, sem ativo livre de risco
                           com liquidez equivalente no ecossistema cripto)

        Hardcodar a Selic para todos os ativos distorcia o Sharpe de
        ativos americanos e cripto — um ativo com retorno de 8% a.a.
        pareceria ter Sharpe negativo frente à Selic de 10.75%, quando
        na verdade supera o T-bill americano de 5.25%.

    Returns:
        float: Taxa livre de risco em decimal (ex: 0.1075 para 10.75%).
               Padrão: 10.75% (Selic aproximada) se entrada inválida.
    """
    print(f"{'─'*60}")
    print("  Taxa livre de risco sugerida por mercado:")
    print("    10.75 → Selic (ações brasileiras, B3)  [padrão]")
    print("     5.25 → Fed Funds Rate (ações americanas, EUA)")
    print("     0.00 → Zero (criptoativos — sem benchmark equivalente)")
    entrada_taxa = input("  [4/4] Taxa livre de risco % a.a. [Padrão 10.75]: ").strip()

    if not entrada_taxa:
        print("  [INFO] Usando Selic padrão: 10.75% a.a.")
        return 0.1075

    try:
        taxa = float(entrada_taxa.replace(',', '.'))
        if taxa < 0 or taxa > 100:
            raise ValueError("Taxa fora do intervalo esperado [0, 100].")
        return taxa / 100.0
    except ValueError:
        print("  [AVISO] Valor inválido. Usando 10.75% como padrão.")
        return 0.1075


# -----------------------------------------------------------------------------
# [M2] NOVA FUNÇÃO — DETECÇÃO AUTOMÁTICA DO FATOR DE ANUALIZAÇÃO
# -----------------------------------------------------------------------------
def detectar_fator_anual(ticker: str) -> tuple[int, str]:
    """
    [MELHORIA M2] Detecta o fator de anualização correto com base no tipo de ativo.

    MOTIVAÇÃO:
        O fator de anualização converte volatilidade diária em anual via
        multiplicação por √N, onde N é o número de dias de negociação por ano:

          • Ações / ETFs / Índices: N = 252 (dias úteis em bolsas tradicionais)
          • Criptoativos           : N = 365 (negociam 24/7, sem feriados)

        Usar √252 para cripto subestima a volatilidade em ~20%:
          √252 ≈ 15.87  vs  √365 ≈ 19.10  → diferença de 20.4%

        Esse erro se propaga diretamente para o Índice Sharpe:
          Sharpe = (R_anual - R_livre) / (σ_diária × √N)
          Um σ_anual artificialmente menor gera um Sharpe artificialmente maior,
          fazendo o cripto parecer mais eficiente do que de fato é.

    Critério de detecção:
        Verifica se o ticker contém um hífen ("-") seguido de um sufixo
        monetário reconhecido: USD, BTC, ETH, BRL, EUR, USDT, USDC, GBP.
        Exemplos: BTC-USD, ETH-BRL, SOL-USDT → cripto (365).
        PETR4.SA, AAPL, SPY → ações/ETF (252).

    Args:
        ticker: Ticker resolvido pelo motor de busca.

    Returns:
        tuple: (fator_dias, descricao)
               Ex: (365, "Criptoativo (365 dias/ano)")
                   (252, "Ação/ETF (252 pregões/ano)")
    """
    # Sufixos monetários típicos de pares de negociação cripto
    sufixos_cripto = {"-USD", "-BTC", "-ETH", "-BRL", "-EUR", "-USDT", "-USDC", "-GBP"}
    ticker_upper   = ticker.upper()

    for sufixo in sufixos_cripto:
        if ticker_upper.endswith(sufixo):
            return 365, "Criptoativo (365 dias/ano)"

    return 252, "Ação/ETF (252 pregões/ano)"


def validar_dados(df: pd.DataFrame, ticker: str, sma_longa: int) -> None:
    """
    Valida se os dados baixados são suficientes para a análise.
    Encerra o script com mensagem amigável em caso de falha.

    Args:
        df       : DataFrame com a série de preços já filtrada.
        ticker   : Ticker do ativo (para mensagens de erro).
        sma_longa: Janela da SMA longa — mínimo de pregões necessários.
    """
    if df is None or df.empty:
        print(f"\n  [ERRO] Nenhum dado encontrado para '{ticker}'.")
        print("  Verifique se o ticker está correto e tente novamente.")
        print("  Dica: certifique-se de usar o código correto (ex: PETR4.SA, AAPL, BTC-USD).\n")
        sys.exit(1)

    minimo_pregoes = sma_longa * 2
    if len(df) < minimo_pregoes:
        print(f"\n  [AVISO] Apenas {len(df)} pregões encontrados para '{ticker}'.")
        print(f"  Dados insuficientes (mínimo: {minimo_pregoes} pregões para SMA {sma_longa}d).")
        print("  Tente uma data de início mais antiga ou um ticker com mais histórico.\n")
        sys.exit(1)


# =============================================================================
# EXECUÇÃO DO BLOCO DE CONFIGURAÇÃO INTERATIVO
# =============================================================================
# As quatro funções abaixo coletam os parâmetros do usuário em sequência,
# validando cada entrada antes de prosseguir. Se qualquer função chamar
# sys.exit(1), o script encerra antes de baixar qualquer dado.
#
# [M6] ORDEM DE DEFINIÇÃO CORRIGIDA:
#   TAXA_LIVRE_RISCO e FATOR_ANUAL são definidas AQUI, no bloco de
#   configuração, antes de qualquer cálculo que as utilize (Seção 7).
#   Na versão anterior, TAXA_LIVRE_RISCO era definida dentro da Seção 7
#   (após print() da tabela), causando NameError quando algum cálculo
#   intermediário tentava acessá-la antes desse ponto.

TICKER, NOME_EMPRESA = buscar_ticker_semantico()
CAPITAL_INICIAL      = coletar_capital()
DATA_INICIO          = coletar_data_inicio()
TAXA_LIVRE_RISCO     = coletar_taxa_livre_risco()   # [M1] agora interativo e definido aqui

# [M2] Fator de anualização detectado ANTES dos cálculos de Seção 7
FATOR_ANUAL, DESCRICAO_FATOR = detectar_fator_anual(TICKER)

print(f"\n  [CONFIG] Ativo      : {NOME_EMPRESA} ({TICKER})")
print(f"  [CONFIG] Capital    : R$ {CAPITAL_INICIAL:,.2f}")
print(f"  [CONFIG] Início     : {DATA_INICIO}")
print(f"  [CONFIG] Taxa livre : {TAXA_LIVRE_RISCO*100:.2f}% a.a.")
print(f"  [CONFIG] Anualização: √{FATOR_ANUAL} ({DESCRICAO_FATOR})")

# Alerta explícito ao usuário quando o fator é 365
if FATOR_ANUAL == 365:
    print(f"\n  [AVISO] Criptoativo detectado → usando √365 para anualização.")
    print(f"  [AVISO] Volatilidade e Sharpe calculados com base em 365 dias/ano,")
    print(f"  [AVISO] diferentemente de ações (252 dias). Valores não são diretamente")
    print(f"  [AVISO] comparáveis com benchmarks tradicionais de renda variável.")

print(f"{'─'*60}\n")

# Parâmetros fixos da estratégia
DATA_FIM        = datetime.today().strftime("%Y-%m-%d")
SMA_CURTA       = 20
SMA_LONGA       = 50
CUSTO_TRANSACAO = 0.001  # 0,1% por operação (compra ou venda)


# =============================================================================
# SEÇÃO 2 — COLETA DE DADOS HISTÓRICOS (yfinance)
# =============================================================================
#
# SOBRE O yfinance:
#   O yfinance é um wrapper Python da API não-oficial do Yahoo Finance,
#   amplamente utilizado em projetos acadêmicos e de pesquisa por sua
#   simplicidade e cobertura global de ativos.
#
# PREÇO DE FECHAMENTO AJUSTADO:
#   Utilizamos a coluna "Close" do yfinance, que em versões recentes já
#   retorna o preço ajustado por dividendos e splits por padrão. Isso
#   garante que a série histórica reflita apenas a variação real do valor
#   econômico do ativo, sem distorções por eventos corporativos.
#
# TRATAMENTO DE NaN:
#   Valores ausentes surgem naturalmente em séries financeiras: feriados
#   em bolsas internacionais, suspensões de negociação, dias sem volume.
#   dropna() os remove antes de qualquer cálculo para evitar propagação
#   de NaN nas SMAs e nos retornos logarítmicos.

print(f"{'='*60}")
print(f"  ANÁLISE QUANTITATIVA — {TICKER}")
print(f"  Empresa : {NOME_EMPRESA}")
print(f"  Período : {DATA_INICIO} até {DATA_FIM}")
print(f"{'='*60}\n")

print("[1/5] Baixando dados históricos...")

try:
    raw = yf.download(TICKER, start=DATA_INICIO, end=DATA_FIM, progress=False)
except Exception as e:
    print(f"\n  [ERRO] Falha na conexão com o servidor do Yahoo Finance.")
    print(f"  Detalhe técnico: {e}")
    print("  Verifique sua conexão com a internet e tente novamente.\n")
    sys.exit(1)

if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[["Close"]].copy()
df.dropna(inplace=True)

validar_dados(df, TICKER, SMA_LONGA)

df.rename(columns={"Close": "Preco"}, inplace=True)

print(f"    ✓ {len(df)} pregões carregados  |  "
      f"Primeiro: {df.index[0].date()}  |  Último: {df.index[-1].date()}")


# =============================================================================
# SEÇÃO 3 — RETORNO LOGARÍTMICO
# =============================================================================
#
# FÓRMULA IMPLEMENTADA:
#   r_t = ln(P_t / P_{t-1})
#
#   Equivalência vetorial:
#     df["Preco"].apply(np.log).diff()               ← mais legível
#     np.log(df["Preco"] / df["Preco"].shift(1))     ← fórmula explícita (adotada)
#
#   Ambas produzem resultados idênticos. A segunda forma evidencia a
#   estrutura matemática — útil para fins pedagógicos e revisão acadêmica.
#
# NOTA SOBRE O PRIMEIRO VALOR:
#   O primeiro retorno logarítmico será NaN, pois P_{t-1} não existe para
#   t=0. Esse NaN é controlado: permanece no df completo para manter
#   alinhamento de índices, e é eliminado apenas em df_valido (Seção 6).

print("\n[2/5] Calculando retornos logarítmicos e médias móveis...")

df["Retorno_Log"] = np.log(df["Preco"] / df["Preco"].shift(1))


# =============================================================================
# SEÇÃO 4 — MÉDIAS MÓVEIS SIMPLES (SMA)
# =============================================================================
#
# FÓRMULA:
#   SMA_k(t) = (1/k) × Σ_{i=0}^{k-1} P_{t-i}
#
# PERÍODO DE AQUECIMENTO (WARM-UP PERIOD):
#   Os primeiros (k-1) valores de cada SMA serão NaN:
#     • SMA 20d: primeiros 19 valores são NaN
#     • SMA 50d: primeiros 49 valores são NaN
#   O período efetivo de análise começa apenas após o 50º pregão.
#
# INTERPRETAÇÃO ECONÔMICA:
#   • SMA 20d ≈ tendência de curto prazo (~1 mês útil)
#   • SMA 50d ≈ tendência de médio prazo (~2,5 meses úteis)
#
# ESCOLHA DAS JANELAS:
#   As janelas de 20 e 50 dias são amplamente monitoradas por traders
#   institucionais — gerando uma "profecia auto-realizável" que reforça
#   a validade dos sinais de cruzamento.

col_curta = f"SMA_{SMA_CURTA}"
col_longa = f"SMA_{SMA_LONGA}"

df[col_curta] = df["Preco"].rolling(window=SMA_CURTA).mean()
df[col_longa] = df["Preco"].rolling(window=SMA_LONGA).mean()

print(f"    ✓ SMA {SMA_CURTA} dias e SMA {SMA_LONGA} dias calculadas")


# =============================================================================
# SEÇÃO 5 — GERAÇÃO DE SINAIS (CROSSOVER)
# =============================================================================
#
# LÓGICA DE POSIÇÃO (LONG-ONLY):
#   • Posicao = 1 → comprado (SMA curta > SMA longa)
#   • Posicao = 0 → fora do mercado (SMA curta ≤ SMA longa)
#   Não consideramos posições vendidas (short), pois exigem mecanismos
#   adicionais (aluguel, margem) não modelados neste backtesting.
#
# DETECÇÃO DE CRUZAMENTOS via diff():
#   df["Posicao"].diff() calcula a variação diária:
#     • +1 → Golden Cross (COMPRA)
#     • -1 → Death Cross  (VENDA)
#     •  0 → posição mantida
#
# PREVENÇÃO DE LOOK-AHEAD BIAS:
#   O sinal é gerado no fechamento do dia t. A execução só ocorre em t+1.
#   O deslocamento via .shift(1) na Seção 6 garante que o retorno capturado
#   corresponda ao dia seguinte ao sinal — simulação realista.

df["Posicao"] = np.where(df[col_curta] > df[col_longa], 1, 0)
df["Sinal"]   = df["Posicao"].diff()

sinais  = df[df["Sinal"] != 0].dropna(subset=["Sinal"])
compras = df[df["Sinal"] == 1]
vendas  = df[df["Sinal"] == -1]

print(f"    ✓ Sinais gerados: {len(compras)} compras | {len(vendas)} vendas")


# =============================================================================
# SEÇÃO 6 — BACKTESTING
# =============================================================================
#
# LOOK-AHEAD BIAS — O PRINCIPAL INIMIGO DO BACKTESTING HONESTO:
#   O Look-ahead Bias ocorre quando o sistema usa informações que só
#   estariam disponíveis APÓS o momento da decisão, gerando resultados
#   artificialmente otimistas. O .shift(1) na posição e no custo garante
#   que o retorno e o custo sejam computados no dia da execução (t+1),
#   não no dia do sinal (t).
#
# CUSTO DE TRANSAÇÃO EM ESCALA LOGARÍTMICA:
#   Custo = ln(1 - 0.001) ≈ -0.001
#   Matematicamente consistente: subtrair ln(1-c) é equivalente a
#   multiplicar o capital por (1-c) no espaço aritmético.
#
# [M4] CÁLCULO DE RETORNOS ACUMULADOS SOMENTE SOBRE df_valido:
#
#   PROBLEMA ANTERIOR:
#     cumsum() e exp() eram aplicados sobre df completo, que contém NaN
#     nas primeiras linhas (período de aquecimento das SMAs e primeiro
#     Retorno_Log). O comportamento padrão do pandas com NaN no cumsum()
#     é: NaN + qualquer_valor = NaN → a propagação "zera" o acumulado
#     nos primeiros pregões, fazendo o Capital(t_início_efetivo) ≠
#     CAPITAL_INICIAL até que os NaN sejam superados, introduzindo uma
#     distorção sutil mas real na curva de patrimônio.
#
#   SOLUÇÃO:
#     Criamos df_valido com dropna() ANTES de calcular qualquer acumulado.
#     Assim, o primeiro valor de Retorno_Estrategia e Retorno_Log usado
#     no cumsum() já é numérico, garantindo que:
#       Capital_Estrategia.iloc[0] ≈ CAPITAL_INICIAL (dentro da precisão float)
#     O .copy() explícito evita SettingWithCopyWarning nas atribuições
#     subsequentes via .loc[] (ver [M5]).

print("\n[3/5] Executando backtesting...")

# .shift(1): usa a posição de ontem → elimina look-ahead bias
df["Trade"] = (df["Sinal"] != 0).shift(1).fillna(0).astype(int)

# Custo logarítmico no dia da execução (t+1)
df["Custo"] = df["Trade"] * np.log(1 - CUSTO_TRANSACAO)

# Retorno diário: posição de ontem × retorno de hoje + custo de execução
df["Retorno_Estrategia"] = df["Posicao"].shift(1) * df["Retorno_Log"] + df["Custo"]

# [M4] Isola o período válido (pós warm-up das SMAs) ANTES do cumsum()
# Isso garante que o acumulado comece de um valor 100% limpo de NaN,
# evitando distorções na curva de capital nos primeiros pregões efetivos.
df_valido = df.dropna(subset=[col_curta, col_longa, "Retorno_Log"]).copy()

# Acumulados calculados exclusivamente sobre df_valido
# [M5] .loc[:, col] = valor → notação explícita de escrita sobre cópia
#       sinaliza intenção ao pandas, eliminando SettingWithCopyWarning
df_valido.loc[:, "Acum_Estrategia"] = np.exp(df_valido["Retorno_Estrategia"].cumsum())
df_valido.loc[:, "Acum_BuyHold"]    = np.exp(df_valido["Retorno_Log"].cumsum())

df_valido.loc[:, "Capital_Estrategia"] = CAPITAL_INICIAL * df_valido["Acum_Estrategia"]
df_valido.loc[:, "Capital_BuyHold"]    = CAPITAL_INICIAL * df_valido["Acum_BuyHold"]

capital_final_estrategia = df_valido["Capital_Estrategia"].iloc[-1]
capital_final_bh         = df_valido["Capital_BuyHold"].iloc[-1]

retorno_estrategia = (capital_final_estrategia / CAPITAL_INICIAL - 1) * 100
retorno_bh         = (capital_final_bh          / CAPITAL_INICIAL - 1) * 100


# =============================================================================
# SEÇÃO 7 — MÉTRICAS DE RISCO E PERFORMANCE
# =============================================================================
#
# PRINCÍPIO FUNDAMENTAL:
#   Retorno sem risco é uma informação incompleta. As métricas desta seção
#   capturam a dimensão de risco que o retorno simples ignora.
#
# ── VOLATILIDADE ANUALIZADA ──────────────────────────────────────────────────
#
#   Fórmula: σ_anual = σ_diária × √FATOR_ANUAL
#
#   [M2] FATOR_ANUAL é 252 para ações/ETFs e 365 para criptoativos.
#   O fator correto é essencial: usar √252 em cripto subestima σ em ~20%,
#   inflando artificialmente o Índice Sharpe e tornando a análise de risco
#   irreal. Ver função detectar_fator_anual() para detalhes.
#
# ── ÍNDICE SHARPE ────────────────────────────────────────────────────────────
#
#   Fórmula: Sharpe = (R_anualizado - R_livre) / σ_anual
#
#   [M1] R_livre = TAXA_LIVRE_RISCO, agora fornecida pelo usuário.
#   O benchmark adequado varia por ativo: Selic (B3), Fed Funds (EUA), 0% (cripto).
#   Usar benchmark incorreto distorce a interpretação do Sharpe, mascarando
#   ou exagerando a eficiência real da estratégia.
#
#   INTERPRETAÇÃO:
#     Sharpe < 0     → retorno abaixo da taxa livre de risco
#     0 < Sharpe < 1 → positivo, mas risco alto em relação ao excesso de retorno
#     Sharpe > 1     → bom retorno ajustado ao risco
#     Sharpe > 2     → excelente (raro em mercados eficientes)
#
# ── MAXIMUM DRAWDOWN (MDD) ───────────────────────────────────────────────────
#
#   Fórmula: Drawdown(t) = (Capital(t) - Pico(t)) / Pico(t)
#            MDD = min(Drawdown(t))  para todo t
#
#   O Maximum Drawdown representa o PIOR CENÁRIO DE QUEDA: a maior perda
#   de capital desde um pico histórico até o vale subsequente.
#
#   [M3] DRAWDOWN DO BUY & HOLD CALCULADO AQUI:
#     Na versão anterior, o MDD do B&H era calculado dentro da Seção 8
#     (visualização), após a impressão da tabela de resultados. Isso impedia
#     sua exibição na tabela do terminal e criava uma assimetria: a estratégia
#     tinha MDD na tabela, o B&H não. Ambos agora são calculados juntos aqui,
#     e ambos aparecem na tabela comparativa abaixo.
#
# ── COMPLEMENTARIDADE DAS MÉTRICAS ───────────────────────────────────────────
#   Sharpe captura risco via desvio padrão (volatilidade simétrica).
#   Drawdown captura risco via perdas reais (assimetria de quedas).
#   Juntos, formam uma visão mais completa do perfil de risco da estratégia.

print("\n[4/5] Calculando métricas de risco e performance...")

# [M2] np.sqrt(FATOR_ANUAL): fator correto por tipo de ativo
vol_anual_estrategia = df_valido["Retorno_Estrategia"].std() * np.sqrt(FATOR_ANUAL)
vol_anual_bh         = df_valido["Retorno_Log"].std()         * np.sqrt(FATOR_ANUAL)

n_anos               = len(df_valido) / FATOR_ANUAL
ret_anual_estrategia = (capital_final_estrategia / CAPITAL_INICIAL) ** (1 / n_anos) - 1
ret_anual_bh         = (capital_final_bh          / CAPITAL_INICIAL) ** (1 / n_anos) - 1

# [M1] TAXA_LIVRE_RISCO: definida na Seção 1, fornecida pelo usuário
sharpe_estrategia = ((ret_anual_estrategia - TAXA_LIVRE_RISCO) / vol_anual_estrategia
                     if vol_anual_estrategia > 0 else 0.0)
sharpe_bh         = ((ret_anual_bh - TAXA_LIVRE_RISCO) / vol_anual_bh
                     if vol_anual_bh > 0 else 0.0)

# [M3] DRAWDOWN DA ESTRATÉGIA — calculado sobre df_valido (correto)
# [M5] .loc[:, col] = valor — escrita explícita sobre cópia (sem warning)
df_valido.loc[:, "Pico"]     = df_valido["Capital_Estrategia"].cummax()
df_valido.loc[:, "Drawdown"] = ((df_valido["Capital_Estrategia"] - df_valido["Pico"])
                                 / df_valido["Pico"])
drawdown_max_estrategia      = df_valido["Drawdown"].min()

# [M3] DRAWDOWN DO BUY & HOLD — movido da Seção 8 para cá
# Calculado na mesma seção de métricas para consistência e para permitir
# exibição na tabela do terminal (antes era calculado só para o gráfico).
df_valido.loc[:, "Pico_BH"]     = df_valido["Capital_BuyHold"].cummax()
df_valido.loc[:, "Drawdown_BH"] = ((df_valido["Capital_BuyHold"] - df_valido["Pico_BH"])
                                    / df_valido["Pico_BH"])
drawdown_max_bh                  = df_valido["Drawdown_BH"].min()

# Tabela de resultados no terminal — agora com MDD do B&H incluído [M3]
print(f"\n{'─'*60}")
print(f"  RESULTADO DO BACKTESTING")
print(f"  Ativo  : {NOME_EMPRESA} ({TICKER})")
print(f"  Capital: R$ {CAPITAL_INICIAL:,.2f}  |  Período: {DATA_INICIO} a {DATA_FIM}")
print(f"  Taxa livre de risco: {TAXA_LIVRE_RISCO*100:.2f}% a.a.  |  Anualização: √{FATOR_ANUAL} ({DESCRICAO_FATOR})")
print(f"{'─'*60}")
print(f"  {'Métrica':30s}  {'Crossover SMA':>13s}  {'Buy & Hold':>10s}")
print(f"  {'─'*56}")
print(f"  {'Capital Final':30s}  R$ {capital_final_estrategia:>9,.2f}  |  R$ {capital_final_bh:>9,.2f}")
print(f"  {'Retorno Total':30s}  {retorno_estrategia:>+12.2f}%  |  {retorno_bh:>+9.2f}%")
print(f"  {'Retorno Anualizado':30s}  {ret_anual_estrategia*100:>+12.2f}%  |  {ret_anual_bh*100:>+9.2f}%")
print(f"  {'Volatilidade Anualizada':30s}  {vol_anual_estrategia*100:>12.2f}%  |  {vol_anual_bh*100:>9.2f}%")
print(f"  {'Índice Sharpe':30s}  {sharpe_estrategia:>+13.4f}  |  {sharpe_bh:>+9.4f}")
print(f"  {'Drawdown Máximo':30s}  {drawdown_max_estrategia:>13.2%}  |  {drawdown_max_bh:>9.2%}")
print(f"  {'Nº de Sinais de Compra':30s}  {len(compras):>13d}")
print(f"  {'Nº de Sinais de Venda':30s}  {len(vendas):>13d}")
print(f"{'─'*60}\n")


# =============================================================================
# SEÇÃO 8 — DASHBOARD INTERATIVO COM PLOTLY
# =============================================================================
#
# NARRATIVA VISUAL DO BACKTESTING:
#   O dashboard integra todas as variáveis dinâmicas da simulação em uma
#   narrativa visual coesa, dividida em 4 painéis complementares:
#
#   PAINEL 1 — PREÇO + SMAs + SINAIS:
#     Conta a história do ativo: preço, médias e pontos de decisão.
#     Triângulos verdes (Golden Cross) e vermelhos (Death Cross) marcam
#     os momentos exatos de entrada e saída da estratégia.
#
#   PAINEL 2 — RETORNOS LOGARÍTMICOS DIÁRIOS:
#     Barras coloridas revelam a distribuição de ganhos e perdas diárias.
#     Períodos de alta volatilidade aparecem como barras longas e alternadas;
#     tendências fortes mostram sequências de mesma cor.
#
#   PAINEL 3 — CURVA DE PATRIMÔNIO (EQUITY CURVE):
#     Comparação direta entre Crossover e Buy & Hold. A linha pontilhada
#     horizontal marca o capital inicial como referência de break-even.
#
#   PAINEL 4 — DRAWDOWN COMPARATIVO:
#     [M3] Ambos os drawdowns (estratégia e B&H) são plotados a partir dos
#     valores calculados na Seção 7. A redundância de recalcular Pico_BH
#     e Drawdown_BH foi eliminada — os dados já estão em df_valido.
#
# INTERATIVIDADE DO PLOTLY:
#   • Zoom por seleção ou scroll
#   • Hover unificado (hovermode="x unified")
#   • Legendas clicáveis para ocultar/exibir séries
#   • Download como PNG pela barra de ferramentas

print("[4/5] Gerando visualização interativa com Plotly...")

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[0.45, 0.18, 0.22, 0.15],
    subplot_titles=(
        f"<b>{TICKER}</b> — {NOME_EMPRESA} | SMA {SMA_CURTA} e SMA {SMA_LONGA} | Sinais de Cruzamento",
        "Retorno Logarítmico Diário  [ r_t = ln(P_t / P_{t-1}) ]",
        f"Curva de Patrimônio — Capital Inicial: {CAPITAL_INICIAL:,.0f}",
        "Maximum Drawdown — Queda Máxima em Relação ao Pico Histórico"
    )
)

# ── PAINEL 1: Preço + Médias Móveis + Sinais ─────────────────────────────────

fig.add_trace(go.Scatter(
    x=df.index, y=df["Preco"],
    name="Preço Fechamento",
    line=dict(color="#4A90D9", width=1.5),
    hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Preço: %{y:.2f}<extra></extra>"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df[col_curta],
    name=f"SMA {SMA_CURTA}d (curta)",
    line=dict(color="#F5A623", width=1.8),
    hovertemplate=f"SMA {SMA_CURTA}d: %{{y:.2f}}<extra></extra>"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df[col_longa],
    name=f"SMA {SMA_LONGA}d (longa)",
    line=dict(color="#E05C4B", width=1.8, dash="dot"),
    hovertemplate=f"SMA {SMA_LONGA}d: %{{y:.2f}}<extra></extra>"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=compras.index, y=compras["Preco"],
    name="COMPRA — Golden Cross",
    mode="markers",
    marker=dict(symbol="triangle-up", color="#27AE60", size=12,
                line=dict(color="white", width=1)),
    hovertemplate="<b>COMPRA</b><br>%{x|%d/%m/%Y}<br>%{y:.2f}<extra></extra>"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=vendas.index, y=vendas["Preco"],
    name="VENDA — Death Cross",
    mode="markers",
    marker=dict(symbol="triangle-down", color="#E74C3C", size=12,
                line=dict(color="white", width=1)),
    hovertemplate="<b>VENDA</b><br>%{x|%d/%m/%Y}<br>%{y:.2f}<extra></extra>"
), row=1, col=1)

# ── PAINEL 2: Retornos Logarítmicos Diários ──────────────────────────────────

cores_retorno = ["#27AE60" if v >= 0 else "#E74C3C"
                 for v in df["Retorno_Log"].fillna(0)]

fig.add_trace(go.Bar(
    x=df.index, y=df["Retorno_Log"],
    name="Retorno Log Diário",
    marker_color=cores_retorno,
    opacity=0.75,
    hovertemplate="%{x|%d/%m/%Y}<br>r_t = %{y:.4f}<extra></extra>"
), row=2, col=1)

fig.add_hline(y=0, line_color="rgba(200,200,200,0.5)", line_width=1, row=2, col=1)

# ── PAINEL 3: Curva de Patrimônio ────────────────────────────────────────────

fig.add_trace(go.Scatter(
    x=df_valido.index, y=df_valido["Capital_Estrategia"],
    name=f"Crossover SMA ({retorno_estrategia:+.1f}%)",
    line=dict(color="#8E44AD", width=2),
    fill="tozeroy", fillcolor="rgba(142,68,173,0.08)",
    hovertemplate="<b>Crossover</b>: %{y:,.2f}<extra></extra>"
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=df_valido.index, y=df_valido["Capital_BuyHold"],
    name=f"Buy & Hold ({retorno_bh:+.1f}%)",
    line=dict(color="#1ABC9C", width=2, dash="dash"),
    hovertemplate="<b>Buy & Hold</b>: %{y:,.2f}<extra></extra>"
), row=3, col=1)

fig.add_hline(y=CAPITAL_INICIAL, line_color="rgba(200,200,200,0.6)",
              line_width=1, line_dash="dot", row=3, col=1)

# ── PAINEL 4: Maximum Drawdown ───────────────────────────────────────────────
# [M3] Drawdown_BH já calculado na Seção 7 — sem recalcular aqui.
# Isso elimina a duplicação de código e garante consistência entre
# os valores exibidos no gráfico e os reportados na tabela do terminal.

fig.add_trace(go.Scatter(
    x=df_valido.index, y=df_valido["Drawdown"],
    name=f"Drawdown Crossover (MDD: {drawdown_max_estrategia:.1%})",
    line=dict(color="#FF4D4D", width=1.5),
    fill="tozeroy", fillcolor="rgba(255,77,77,0.2)",
    hovertemplate="%{x|%d/%m/%Y}<br>Drawdown: %{y:.2%}<extra></extra>"
), row=4, col=1)

fig.add_trace(go.Scatter(
    x=df_valido.index, y=df_valido["Drawdown_BH"],
    name=f"Drawdown Buy & Hold (MDD: {drawdown_max_bh:.1%})",
    line=dict(color="#1ABC9C", width=1.5, dash="dash"),
    fill="tozeroy", fillcolor="rgba(26,188,156,0.08)",
    hovertemplate="%{x|%d/%m/%Y}<br>Drawdown B&H: %{y:.2%}<extra></extra>"
), row=4, col=1)

fig.add_hline(y=0, line_color="rgba(200,200,200,0.5)", row=4, col=1)

# ── LAYOUT GERAL ─────────────────────────────────────────────────────────────

fig.update_layout(
    title=dict(
        text=(f"Backtesting Quantitativo — {NOME_EMPRESA} ({TICKER}) | "
              f"{DATA_INICIO} a {DATA_FIM}  |  "
              f"Taxa livre: {TAXA_LIVRE_RISCO*100:.2f}%  |  √{FATOR_ANUAL}"),
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

fig.update_yaxes(title_text="Preço",         row=1, col=1, gridcolor="rgba(255,255,255,0.05)")
fig.update_yaxes(title_text="Retorno Log",   row=2, col=1, gridcolor="rgba(255,255,255,0.05)", tickformat=".3f")
fig.update_yaxes(title_text="Capital",       row=3, col=1, gridcolor="rgba(255,255,255,0.05)")
fig.update_yaxes(title_text="Drawdown (%)",  row=4, col=1, gridcolor="rgba(255,255,255,0.05)", tickformat=".0%")

fig.update_xaxes(title_text="Data", row=4, col=1,
                 gridcolor="rgba(255,255,255,0.05)",
                 rangeslider=dict(visible=False))

# Rodapé com MDD comparativo (estratégia vs B&H) — [M3]
anotacao = (
    f"Capital Final — Crossover: {capital_final_estrategia:,.2f}  |  "
    f"Buy & Hold: {capital_final_bh:,.2f}  |  "
    f"Sharpe Crossover: {sharpe_estrategia:.2f}  |  "
    f"MDD Crossover: {drawdown_max_estrategia:.2%}  |  "
    f"MDD B&H: {drawdown_max_bh:.2%}"
)
fig.add_annotation(
    text=anotacao,
    xref="paper", yref="paper", x=0.01, y=-0.03,
    showarrow=False, font=dict(size=10, color="#8B949E"),
    align="left"
)


# =============================================================================
# SEÇÃO 9 — EXIBIÇÃO DO GRÁFICO E ENCERRAMENTO
# =============================================================================
#
# fig.show() serializa o dashboard em HTML e abre no navegador padrão do
# sistema via o módulo webbrowser do Python.
#
# O arquivo HTML gerado é completamente autossuficiente (inline JavaScript)
# e pode ser salvo e compartilhado sem dependências externas.

print("[5/5] Abrindo gráfico no navegador...\n")
fig.show()

print("✓ Simulação concluída com sucesso!")
print(f"  Empresa analisada : {NOME_EMPRESA}")
print(f"  Ticker            : {TICKER}")
print(f"  Período simulado  : {DATA_INICIO} a {DATA_FIM}")
print(f"  Capital inicial   : R$ {CAPITAL_INICIAL:,.2f}")
print(f"  Taxa livre de risco: {TAXA_LIVRE_RISCO*100:.2f}% a.a.  ({DESCRICAO_FATOR})")
print(f"  Crossover final   : {capital_final_estrategia:,.2f}  ({retorno_estrategia:+.2f}%)")
print(f"  Buy & Hold final  : {capital_final_bh:,.2f}  ({retorno_bh:+.2f}%)")
print(f"  Índice Sharpe     : {sharpe_estrategia:+.4f}  (Crossover)  |  {sharpe_bh:+.4f}  (B&H)")
print(f"  MDD Crossover     : {drawdown_max_estrategia:.2%}")
print(f"  MDD Buy & Hold    : {drawdown_max_bh:.2%}\n")
