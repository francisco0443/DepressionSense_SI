# Feature Extraction

## Context

Na INESC TEC, no âmbito do projeto DepressionSense e durante o estágio de verão entre julho e setembro de 2025, foi desenvolvida uma pipeline em Python para processar e analisar dados comportamentais e de mouse interaction (clickstream).

O foco do trabalho foi metodológico: construir um fluxo leakage-safe e participant-aware para preparar os dados, treinar o modelo e apoiar a análise posterior sem misturar informação entre participantes.

## Methodology

O conjunto de dados foi organizado por participante, com split ao nível do participante para garantir separação estrita entre treino e teste. A validação interna foi feita com `GroupKFold`, evitando que sessões do mesmo participante aparecessem em folds diferentes.

A preparação dos dados inclui:

- extração do identificador do participante a partir da chave da sessão;
- criação de splits train/test ao nível do participante;
- preprocessing leakage-safe aplicado apenas a partir do treino;
- construção de sequências por sessão com padding/truncation controlados;
- tratamento de variáveis contínuas e categóricas, incluindo embeddings para features categóricas;
- máscara de timesteps válidos para ignorar padding na otimização e nas métricas.

O núcleo do pipeline e um LSTM autoencoder que aprende representações sequenciais a partir das dinâmicas de interação do rato. O encoder comprime a sequência numa bottleneck representation e o decoder tenta reconstruir as features contínuas de entrada. A arquitetura foi desenhada para capturar estrutura temporal e espacial do comportamento, em vez de depender apenas de estatísticas agregadas simples.

Para referência metodológica, o pipeline inclui ainda:

- comparação com um baseline simples baseado na média das features de treino;
- cálculo de erros de reconstrução por sessão e por feature;
- extração de embeddings na camada bottleneck para análise posterior;
- análise estatística complementar sobre os artefactos produzidos pelo pipeline.
