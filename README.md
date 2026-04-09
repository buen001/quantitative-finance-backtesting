# 🎓 Análise de Ativos e Estratégias Quantitativas com Python

[cite_start]Este projeto desenvolve uma análise completa de ativos financeiros, utilizando o cruzamento de médias móveis para validar estratégias de investimento[cite: 7, 26].

## 📌 Sobre o Projeto
[cite_start]O objetivo é modelar séries temporais financeiras para identificar tendências de mercado[cite: 14, 15]. [cite_start]A aplicação utiliza dados reais para realizar cálculos estatísticos e visualizações gráficas de alta fidelidade[cite: 8, 9, 10].

## 🛠️ Metodologia Aplicada
[cite_start]Seguindo o plano metodológico estabelecido[cite: 30]:
1. [cite_start]**Coleta de Dados**: Integração com a API `yfinance`[cite: 31].
2. [cite_start]**Cálculo de Retornos**: Utilização de **Retorno Logarítmico** para garantir aditividade temporal[cite: 20, 21].
3. [cite_start]**Indicadores**: Implementação de **Médias Móveis Simples (SMA)** de curto e longo prazo[cite: 23, 24].
4. [cite_start]**Estratégia**: Lógica de cruzamento (*Golden Cross* e *Death Cross*) para sinais de compra e venda[cite: 27, 28, 34].
5. [cite_start]**Backtesting**: Simulação realística com custos de transação e análise de risco (**Drawdown**)[cite: 35].

## 🚀 Como Executar
1. Clone este repositório.
2. Instale as bibliotecas necessárias:
   ```bash
   pip install pandas numpy plotly yfinance
