from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    anthropic_api_key: str
    ollama_url: str = 'http://localhost:11434'
    embed_model: str = 'nomic-embed-text'
    embed_dim: int = 768
    top_k: int = 5

    class Config:
        env_file = '.env'


settings = Settings()
