"""Typed configuration loaded from ``config/settings.yaml`` + environment overrides.

Precedence (highest first): explicit kwargs, ``PEAD_*`` env vars, ``.env`` file, YAML.
API keys are read from the environment / ``.env`` only and never from YAML.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BacktestConfig(_Section):
    start_date: date
    end_date: date
    warmup_calendar_days: int = Field(400, ge=300)
    initial_capital_eur: float = Field(1000.0, gt=0)
    eur_usd_rate: float = Field(1.06, gt=0)
    benchmark_symbol: str = "SPY"
    force_close_at_end: bool = True

    @property
    def initial_capital_usd(self) -> float:
        return self.initial_capital_eur * self.eur_usd_rate

    @model_validator(mode="after")
    def _check_dates(self) -> "BacktestConfig":
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class DataConfig(_Section):
    provider: Literal["auto", "fmp_massive", "yfinance", "sample"] = "auto"
    cache_dir: str = "data/cache"
    massive_base_url: str = "https://api.massive.com"
    fmp_base_url: str = "https://financialmodelingprep.com"
    massive_calls_per_minute: int = Field(5, gt=0)
    fmp_calls_per_minute: int = Field(250, gt=0)
    yfinance_threads: int = Field(8, ge=1, le=32)
    include_removed_index_members: bool = True
    universe_indexes: list[str] = Field(default_factory=lambda: ["sp400", "sp600"])
    sample_n_symbols: int = Field(60, ge=5)
    sample_seed: int = 7

    @property
    def cache_path(self) -> Path:
        p = Path(self.cache_dir)
        return p if p.is_absolute() else REPO_ROOT / p


class RegimeConfig(_Section):
    sma_window: int = 200
    vol_window: int = 20
    normal_vol_max: float = 0.22
    panic_vol_min: float = 0.35
    normal_min_gap: float = 0.05
    caution_min_gap: float = 0.08
    caution_risk_multiplier: float = Field(0.5, ge=0, le=1)
    max_positions_normal: int = Field(3, ge=1)
    max_positions_caution: int = Field(2, ge=0)


class ScreenerConfig(_Section):
    min_market_cap: float = 300e6
    max_market_cap: float = 10e9
    min_price: float = 5.0
    max_price: float = 150.0
    min_avg_dollar_volume: float = 5e6
    avg_window: int = 20
    min_volume_ratio: float = 2.5
    min_close_location: float = Field(0.65, ge=0, le=1)
    min_eps_surprise_pct: float = 0.0
    min_revenue_surprise_pct: float = 0.0
    require_revenue_surprise: bool = False
    sue_threshold: float = 1.5
    sue_lookback_quarters: int = 8
    sue_min_quarters: int = 4
    trend_sma_window: int = 50
    high_52w_window: int = 252
    max_distance_from_52w_high: float = 0.20
    analyst_tiebreak_enabled: bool = False
    analyst_tiebreak_max: int = 8


class RiskConfig(_Section):
    base_risk_per_trade: float = Field(0.05, gt=0)
    max_risk_per_trade: float = Field(0.10, gt=0)
    atr_window: int = 14
    atr_stop_multiple: float = 1.5
    max_stop_pct: float = Field(0.15, gt=0, lt=1)
    max_position_pct_equity: float = Field(0.50, gt=0, le=1)

    @model_validator(mode="after")
    def _cap_risk(self) -> "RiskConfig":
        if self.base_risk_per_trade > self.max_risk_per_trade:
            raise ValueError("base_risk_per_trade exceeds max_risk_per_trade (10% cap)")
        return self


class ExecutionConfig(_Section):
    entry_limit_cap_pct: float = 0.003
    commission_per_side_usd: float = 0.38
    slippage_per_side_pct: float = 0.0025
    vps_monthly_eur: float = 6.50
    vps_enabled: bool = False
    breakeven_trigger_r: float = 1.5
    trail_trigger_r: float = 2.0
    trail_atr_multiple: float = 2.5
    time_stop_days: int = 45
    time_stop_min_r: float = 1.0
    settlement_days: int = Field(1, ge=0)


class MetricsConfig(_Section):
    periods_per_year: int = 252
    risk_free_rate: float = 0.0
    dsr_n_trials: int = Field(10, ge=1)
    dsr_trial_sharpe_std: float = Field(0.5, ge=0)


class _YamlSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._data: dict[str, Any] = yaml.safe_load(path.read_text()) if path.exists() else {}

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._data or {})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PEAD_",
        env_nested_delimiter="__",
        env_file=REPO_ROOT / ".env",
        extra="ignore",
        frozen=True,
    )

    backtest: BacktestConfig
    data: DataConfig = DataConfig()
    regime: RegimeConfig = RegimeConfig()
    screener: ScreenerConfig = ScreenerConfig()
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfig = ExecutionConfig()
    metrics: MetricsConfig = MetricsConfig()

    fmp_api_key: SecretStr | None = Field(None, validation_alias="FMP_API_KEY")
    massive_api_key: SecretStr | None = Field(None, validation_alias="MASSIVE_API_KEY")

    _yaml_path: Path = DEFAULT_SETTINGS_PATH

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls, _ACTIVE_YAML[0]),
        )


_ACTIVE_YAML: list[Path] = [DEFAULT_SETTINGS_PATH]


def load_settings(path: Path | str | None = None, **overrides: Any) -> Settings:
    """Load settings from YAML (default ``config/settings.yaml``) with overrides."""
    _ACTIVE_YAML[0] = Path(path) if path else DEFAULT_SETTINGS_PATH
    return Settings(**overrides)
