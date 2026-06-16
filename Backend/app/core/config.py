import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    db_host : str = os.getenv("MYSQL_HOST", "")
    db_user : str = os.getenv("MYSQL_USER", "")
    db_password : str = os.getenv("MYSQL_PASSWORD", "")
    db_port : int = os.getenv("MYSQL_PORT", "")
    db_name : str = os.getenv("MYSQL_DB", "")

    # password hashing
    hash_salt : str = os.getenv("HASH_SALT", "SPRING_FIX_SALT")
    git_repo_directory : str = os.getenv("GIT_REPO_DIRECTORY", "../../Git Repositories")

    llm_provider : str = os.getenv("LLM_PROVIDER","openai")
    coding_chat_model : str = os.getenv("CODING_CHAT_MODEL", "gpt-5.4")
    llm_api_key : str = os.getenv("LLM_API_KEY", "")
    llm_base_url : str = os.getenv("LLM_BASE_URL", "")
    llm_max_tokens : int = int(os.getenv("LLM_MAX_TOKENS", 2048))
    _raw_temp = os.getenv("LLM_TEMPERATURE")
    llm_temperature: float | None = (
        float(_raw_temp) if _raw_temp is not None and float(_raw_temp) != -10 else None
    )
    INCLUDE_FULL_HISTORY : bool = str(os.getenv("INCLUDE_FULL_HISTORY", "False")).strip().lower() == "true"
    kafka_bootstrap_servers : str = os.getenv("KAFKA_URL", "localhost:9092")
    template_folder : str = os.getenv("TEMPLATE_FOLDER", "")

    email_to : str = os.getenv("EMAIL_RECEIVER", "")
    http_client_verify : str = os.getenv("HTTP_CLIENT_VERIFY", True)

    app_host : str = os.getenv("APP_HOST", "localhost")
    app_port : str = os.getenv("APP_PORT", "8000")

    # vector db configs
    VECTOR_DB_PROVIDER : str = os.getenv("VECTOR_DB_PROVIDER", "qdrant")
    VECTOR_DB_HOST : str = os.getenv("VECTOR_DB_HOST", "localhost")
    VECTOR_DB_PORT : int = int(os.getenv("VECTOR_DB_PORT", '6333'))
    VECTOR_DB_COLLECTION : str = os.getenv("VECTOR_DB_COLLECTION", "fix_knowledge")
    # Milvus-specific
    MILVUS_URI : str = os.getenv("MILVUS_URI", "http://localhost:19530")

    # embedding model configs
    EMBEDDING_PROVIDER : str = os.getenv("EMBEDDING_PROVIDER", "openai")   # "openai" | "ollama"
    EMBEDDING_MODEL : str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION : int = int(os.getenv("EMBEDDING_DIMENSION", 1536))
    EMBEDDING_BASE_URL : str = os.getenv("EMBEDDING_BASE_URL", "")          # required for ollama (e.g. http://localhost:11434)

settings = Settings()