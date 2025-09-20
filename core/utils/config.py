from dataclasses import dataclass

@dataclass
class DataConfig:
    ticker: str = "AAPL"
    period: str = "1y"
    interval: str = "1d"

@dataclass
class TrainingConfig:
    episodes: int = 1
    seed: int = 42
