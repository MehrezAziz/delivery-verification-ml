from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_key: str | None = None
    insightface_root: str = "/root/.insightface"
    insightface_model: str = "buffalo_l"
    face_det_size: int = 640
    ocr_engine: str = "rapidocr"


settings = Settings()
