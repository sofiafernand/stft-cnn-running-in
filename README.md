# Identificação do Amaciamento de Compressores Alternativos

Arquitetura de software composta por um pipeline de extração de características tempo-frequência (STFT), classificação via Redes Neurais Convolucionais (CNN) e um agente de Inteligência Artificial Generativa para emissão de diagnósticos estruturados.

## Restrição de Dados e Reprodutibilidade

Os diretórios `./dados_brutos` e `./resultados` referenciados no código não estão incluídos neste repositório. O dataset original de sinais acústicos e de vibração pertence ao escopo do programa PRH-ANP 02/UFSC e Labmetro, possuindo restrições estritas de propriedade intelectual e confidencialidade acadêmica/industrial. 

Desta forma, a execução local (end-to-end) por terceiros não é possível. O código fonte está disponibilizado com o objetivo de demonstrar a arquitetura de software, a estruturação de pipelines de Machine Learning e a integração determinística de LLMs em ambiente de produção (Structured Outputs).

## Pré-requisitos

- Python 3.10+
- h5py==3.14.0
- instructor==1.2.5
- matplotlib==3.11.1
- numpy==2.5.1
- openai==1.20.0
- optuna==4.9.0
- optuna-dashboard==0.20.0
- pandas==3.0.5
- pydantic==2.7.0
- pydantic-settings==2.2.1
- scikit-learn==1.9.0
- scipy==1.18.0
- tensorflow==2.21.0

## Instalação

1. Clone o repositório:
```bash
git clone [https://github.com/sofiafernand/stft-cnn-running-in.git](https://github.com/sofiafernand/stft-cnn-running-in.git)
```

2. Acesse o diretório do projeto:
```bash
cd stft-cnn-running-in
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Configuração

Crie um arquivo `.env` na raiz do projeto com base no arquivo `.env.example`. As seguintes variáveis de ambiente determinam o comportamento do pipeline e da camada de inferência:

| Variável | Descrição | Valor Padrão / Exemplo |
| --- | --- | --- |
| `FREQUENCIA_AMOSTRAGEM` | Frequência de aquisição dos sinais em Hz | 25600 |
| `CAMINHO_DADOS` | Diretório de leitura dos dados brutos | `./dados_brutos` |
| `CAMINHO_RESULTADOS` | Diretório de gravação dos tensores HDF5 e modelos | `./resultados` |
| `OPENAI_API_KEY` | Chave de autenticação da API da OpenAI | `sk-...` |
| `LLM_MODEL` | Identificador do modelo de linguagem utilizado | `gpt-4-turbo` |
| `LLM_TEMPERATURE` | Parâmetro de temperatura do modelo de linguagem | `0.1` |

## Uso (Referência Arquitetural)

O repositório é dividido em ambiente de experimentação (`research_pipeline`) e ambiente de produção (`src`). Os comandos abaixo ilustram o fluxo de execução projetado, exigindo a presença dos datasets originais.

Pipeline de extração de dados e treinamento do modelo preditivo (Pesquisa):
```bash
python research_pipeline/etapa1_stft_fabrica_datasets.py
python research_pipeline/etapa3_otimizacao_continua.py
python research_pipeline/etapa4_treinamento_final.py
```

Monitoramento da otimização de hiperparâmetros (Optuna):
```text
http://localhost:8080
```

Execução do agente de Inteligência Artificial Generativa (Produção):
```bash
python src/etapa5_agente_genai.py
```

## Estrutura de Pastas

```text
.
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── research_pipeline/
│   ├── etapa1_stft_fabrica_datasets.py
│   ├── etapa2_checagem_hdf5.py
│   ├── etapa3_otimizacao_continua.py
│   └── etapa4_treinamento_final.py
└── src/
    ├── config.py
    ├── etapa1_extracao_stft.py
    ├── etapa2_pre_processamento.py
    ├── etapa3_otimizacao_optuna.py
    ├── etapa4_avaliacao_cnn.py
    └── etapa5_agente_genai.py
```

## Licença

Uso Acadêmico e Pesquisa. Código desenvolvido sob o escopo do Programa PRH-ANP 02/UFSC e Labmetro.
