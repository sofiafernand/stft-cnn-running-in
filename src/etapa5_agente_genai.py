"""
Módulo de Integração com Inteligência Artificial Generativa
Objetivo: Encapsula um Agente LLM para interpretar os outputs probabilísticos da CNN 
e gerar relatórios estruturados de manutenção via Retrieval-Augmented Generation (RAG).

Decisão de Engenharia: Utiliza a biblioteca 'instructor' acoplada ao 'pydantic' para 
forçar o retorno não determinístico do LLM em um JSON estrito, além de configurar 
'max_retries' para resiliência de API.
"""

import logging
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Configuração do cliente OpenAI envolvido pelo Instructor para gerar Structured Outputs
client = instructor.from_openai(OpenAI(api_key=settings.openai_api_key.get_secret_value()))

# 1. Definição do Schema de Saída (JSON Estruturado)
class RelatorioManutencao(BaseModel):
    id_compressor: str
    estado_estimado: str = Field(description="Deve ser 'Amaciado' ou 'Transição/Não Amaciado'")
    confianca_modelo_cnn: float = Field(description="Probabilidade gerada pelo modelo preditivo CNN (0 a 1)")
    risco_operacional: str = Field(description="Classificação de risco: Alto, Médio ou Baixo")
    plano_acao: list[str] = Field(description="Lista de ações executáveis recomendadas com base no estado e no manual técnico")

class RAGRetriever:
    """Simula a busca semântica em base vetorial contendo diretrizes de manutenção industrial."""
    @staticmethod
    def buscar_contexto(probabilidade_cnn: float) -> str:
        # Em ambiente cloud, isto seria uma consulta ao Pinecone, Weaviate ou Milvus.
        if probabilidade_cnn > 0.8:
            return "DIRETRIZ TÉCNICA: Compressores herméticos com alta estabilidade acústica devem prosseguir para ensaios de performance imediatos."
        return "DIRETRIZ TÉCNICA: Transientes tribológicos requerem operação de estabilização em carga parcial por mais 12 horas antes da avaliação metrológica."

class AnalistaGenAI:
    def __init__(self, modelo: str = settings.llm_model, temperatura: float = settings.llm_temperature):
        self.modelo = modelo
        self.temperatura = temperatura

    def gerar_relatorio_estruturado(self, compressor_id: str, probabilidade_cnn: float) -> RelatorioManutencao:
        """
        Consome a predição determinística da CNN, busca contexto via RAG e emite relatório JSON estrito.
        """
        contexto_tecnico = RAGRetriever.buscar_contexto(probabilidade_cnn)
        
        # Gerenciamento de Prompts isolado
        prompt_sistema = (
            "Você é um Engenheiro de Confiabilidade diagnosticando compressores alternativos.\n"
            "Sua tarefa é cruzar a predição da Rede Neural com as diretrizes técnicas "
            "e fornecer um plano de ação seguro e estruturado em formato JSON rigoroso."
        )
        
        prompt_usuario = (
            f"Dados da Análise Preditiva:\n"
            f"- Identificador do Compressor: {compressor_id}\n"
            f"- Probabilidade de estar Amaciado (CNN): {probabilidade_cnn:.4f}\n\n"
            f"Contexto Recuperado dos Manuais (RAG):\n{contexto_tecnico}"
        )

        try:
            logger.info(f"Iniciando análise generativa para o compressor {compressor_id}...")
            # A chamada ao LLM exige aderência ao modelo Pydantic 'RelatorioManutencao'
            relatorio = client.chat.completions.create(
                model=self.modelo,
                temperature=self.temperatura,
                response_model=RelatorioManutencao,
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": prompt_usuario},
                ],
                max_retries=3 # Controle nativo para lidar com eventuais timeouts de API
            )
            return relatorio
            
        except Exception as e:
            logger.error(f"Falha na comunicação com a API do LLM: {e}")
            raise

if __name__ == "__main__":
    analista = AnalistaGenAI()
    
    # Simulação do recebimento do output da camada de rede neural (CNN)
    predicao_maquina_b15 = 0.912 
    
    relatorio_gerado = analista.gerar_relatorio_estruturado(
        compressor_id="B15", 
        probabilidade_cnn=predicao_maquina_b15
    )
    
    logger.info("Relatório de Inteligência Artificial Generativa gerado com sucesso:\n")
    print(relatorio_gerado.model_dump_json(indent=2))
