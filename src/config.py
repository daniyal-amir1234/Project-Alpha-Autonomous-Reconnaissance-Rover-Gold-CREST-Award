from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    camera_id: int = 0
    width: int = 640
    height: int = 480
    max_results: int = 3
    score_threshold: float = 0.30
    num_threads: int = 2
    use_gpio: bool = False
    mirror: bool = True
