from dataclasses import dataclass
from pathlib import Path


@dataclass
class NexusConfig:
    asr_grpc_addr: str
    chat_base_url: str
    chat_api_key: str
    realtime_api_key: str | None = None
    realtime_client_secret_ttl_seconds: int = 600
    realtime_session_max_seconds: int = 3600

    tts_grpc_addr: str | None = None
    tts_grpc_preset_voice_id: str | None = None
    tts_grpc_ref_voice_dir: str | None = None
    tts_grpc_language: str = "auto"
    tts_grpc_decoder_chunk_size: int = 50
    tts_grpc_text_chunk_size: int = 1
    asr_interim_results: bool = True

    def __post_init__(self) -> None:
        if not self.tts_grpc_addr:
            raise ValueError("tts_grpc_addr is required")

        has_preset_voice = bool(self.tts_grpc_preset_voice_id)
        has_reference_voice_dir = bool(self.tts_grpc_ref_voice_dir)
        if has_preset_voice == has_reference_voice_dir:
            raise ValueError(
                "Configure exactly one gRPC voice source: preset voice or reference voice directory"
            )

        if self.tts_grpc_ref_voice_dir:
            ref_voice_dir = Path(self.tts_grpc_ref_voice_dir)
            if not ref_voice_dir.exists():
                raise ValueError(f"gRPC reference voice directory not found: {ref_voice_dir}")
            if not ref_voice_dir.is_dir():
                raise ValueError(
                    f"gRPC reference voice directory must be a directory: {ref_voice_dir}"
                )
            tsv_files = list(ref_voice_dir.glob("*.tsv"))
            if len(tsv_files) != 1:
                raise ValueError(
                    "gRPC reference voice directory must contain exactly one TSV file"
                )

        if self.tts_grpc_decoder_chunk_size <= 0:
            raise ValueError("tts_grpc_decoder_chunk_size must be greater than 0")
        if self.tts_grpc_text_chunk_size <= 0:
            raise ValueError("tts_grpc_text_chunk_size must be greater than 0")
        if self.realtime_client_secret_ttl_seconds <= 0:
            raise ValueError("realtime_client_secret_ttl_seconds must be greater than 0")
        if self.realtime_session_max_seconds <= 0:
            raise ValueError("realtime_session_max_seconds must be greater than 0")
