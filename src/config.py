"""
Módulo de Gerenciamento de Configuração Centralizada
Substitui as variáveis globais, integrando Pydantic Settings para validação
de tipos e leitura segura de chaves do arquivo .env.
"""

from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class PipelineSettings(BaseSettings):
    # Configurações de Deep Learning
    frequencia_amostragem: int = Field(default=25600)
    caminho_dados: Path = Field(default=Path("./dados_brutos"))
    caminho_resultados: Path = Field(default=Path("./resultados"))
    
    # Configurações de Inteligência Artificial Generativa
    openai_api_key: SecretStr
    llm_model: str = Field(default="gpt-4-turbo")
    llm_temperature: float = Field(default=0.1)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = PipelineSettings()
