"""Execution contract shared by the backtest broker and a future IBKR (ib_async) broker."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.models import AccountState, ExitReason, OrderIntent


class ExecutionBroker(ABC):
    @abstractmethod
    def submit_bracket_order(self, intent: OrderIntent) -> str:
        """Queue a limit entry (cap = intent.limit_price) with a child stop at intent.stop_price.

        Returns an order id. The entry is only attempted at the next session's open.
        """

    @abstractmethod
    def update_stop(self, symbol: str, new_stop: float) -> None:
        """Raise (never lower) the protective stop of an open position."""

    @abstractmethod
    def close_position(self, symbol: str, reason: ExitReason) -> None:
        """Flatten at the next available price (next open in the backtest)."""

    @abstractmethod
    def get_account_state(self) -> AccountState:
        """Cash (settled / unsettled), market value and open position count."""
