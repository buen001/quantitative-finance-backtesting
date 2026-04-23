# 📈 Simulador de Backtesting Quantitativo (SMA Crossover)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen?style=for-the-badge)

Este repositório contém um simulador profissional de **Backtesting** para estratégias de investimento, desenvolvido em Python. O sistema utiliza dados históricos reais via Yahoo Finance para validar a estratégia de Cruzamento de Médias Móveis Simples (SMA).

## 🎯 Objetivo do Projeto
Transformar dados brutos do mercado financeiro em insights visuais e estatísticos, permitindo que o investidor teste suas hipóteses antes de arriscar capital real.

## 🚀 Diferenciais Técnicos (V2.0)
Nesta versão, foram implementados rigoros técnicos para evitar erros comuns de simulação:
- **Ajuste Cripto:** Identificação automática de ativos 24/7 (Cripto) ajustando a base de cálculo de 252 para 365 dias.
- **Prevenção de Bias:** Implementação de *shift* operacional para evitar o viés de antecipação (compras feitas no dia seguinte ao sinal).
- **Análise de Drawdown:** Comparação em tempo real do "sofrimento" da estratégia vs. a estratégia de Buy & Hold.
- **Taxa Livre de Risco:** Cálculo de Índice Sharpe dinâmico baseado na taxa de juros informada pelo usuário.

## 🛠️ Tecnologias Utilizadas
- **Python 3.12**
- **Pandas & NumPy:** Manipulação e cálculo de séries temporais.
- **YFinance:** Ingestão de dados do mercado financeiro global.
- **Plotly:** Visualização de dados através de dashboards interativos em HTML.

## 📋 Como utilizar
1. Clone este repositório.
2. Instale as dependências: `pip install pandas numpy plotly yfinance`.
3. Execute o script: `python atividade02_estrategia_quantitativa.py`.
4. Siga as instruções no terminal:
   - Digite o ticker (ex: `PETR4.SA`, `AAPL`, `BTC-USD`).
   - Informe o capital inicial.
   - Defina a data de início (AAAA-MM-DD).
   - Informe a taxa de juros (ex: `0.1075` para 10,75% ao ano).

## 📊 Exemplo de Saída
O sistema gera um dashboard com 4 painéis:
1. **Preço e Sinais:** Onde as médias se cruzam e os gatilhos são disparados.
2. **Evolução do Capital:** Comparativo entre a Estratégia e o Buy & Hold.
3. **Drawdown:** Visualização das quedas máximas durante o período.
4. **Retornos Diários:** Histograma de volatilidade.

---
*Projeto desenvolvido para a disciplina de Análise de Ativos e Finanças Quantitativas.*
