from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BalanceUser:
    umo: str
    enabled: bool
    balance: float
    cost_per_image: float


@dataclass(frozen=True)
class BalancePrecheckResult:
    ok: bool
    message: str | None = None
    user: BalanceUser | None = None


@dataclass(frozen=True)
class BalanceChargeResult:
    used_amount: float
    remaining_balance: float
    image_count: int


class BalanceManager:
    """Persisted per-UMO balance state backed by balances.json."""

    def __init__(self, data_dir: Path | str, config: Any):
        self.data_dir = Path(data_dir)
        self.config = config
        self.path = self.data_dir / "balances.json"
        self._balances: dict[str, dict[str, Any]] = {}
        self._users: dict[str, BalanceUser] = {}

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load()
        self.sync_from_config()

    def sync_from_config(self) -> None:
        users: dict[str, BalanceUser] = {}
        changed = False

        for entry in self._balance_entries():
            umo = str(entry.get("umo") or "").strip()
            if not umo:
                continue

            enabled = bool(entry.get("enabled", True))
            cost = self._non_negative_float(entry.get("cost_per_image"), 0.1)
            balance = self._stored_balance(umo)

            add_amount = self._float(entry.get("add_amount"), 0.0)
            if add_amount:
                balance = self._round_amount(max(0.0, balance + add_amount))
                self._set_balance(umo, balance, cost)
                entry["add_amount"] = 0
                changed = True
            else:
                self._ensure_record(umo, balance, cost)

            entry["balance_display"] = self.format_amount(balance)
            users[umo] = BalanceUser(
                umo=umo,
                enabled=enabled,
                balance=balance,
                cost_per_image=cost,
            )

        self._users = users
        if changed:
            self._save()

    def precheck(self, umo: str, requested_count: int) -> BalancePrecheckResult:
        self.sync_from_config()

        user = self._users.get(umo)
        if user is None:
            return BalancePrecheckResult(False, "当前会话未配置余额用户，禁止使用生图")

        if not user.enabled:
            return BalancePrecheckResult(False, "当前会话余额用户已禁用，禁止使用生图", user)

        count = max(1, self._int(requested_count, 1))
        cost = user.cost_per_image
        balance = user.balance
        needed = self._round_amount(count * cost)

        if balance < cost:
            return BalancePrecheckResult(
                False,
                (
                    "余额不足，禁止生图。"
                    f"当前余额 {self.format_amount(balance)}，"
                    f"生成下一张图需要 {self.format_amount(cost)}。请先充值。"
                ),
                user,
            )

        if balance < needed:
            max_count = int(balance // cost) if cost else count
            return BalancePrecheckResult(
                False,
                (
                    f"余额不足，本次需要 {self.format_amount(needed)}，"
                    f"当前余额 {self.format_amount(balance)}，"
                    f"最多可生成 {max_count} 张。"
                ),
                user,
            )

        return BalancePrecheckResult(True, user=user)

    def charge(self, umo: str, image_count: int) -> BalanceChargeResult:
        self.sync_from_config()

        user = self._users.get(umo)
        if user is None:
            raise ValueError(f"UMO not configured for balance charging: {umo}")

        count = max(0, self._int(image_count, 0))
        used_amount = self._round_amount(count * user.cost_per_image)
        remaining = self._round_amount(max(0.0, user.balance - used_amount))
        self._set_balance(umo, remaining, user.cost_per_image)
        self._save()
        self.sync_from_config()
        return BalanceChargeResult(
            used_amount=used_amount,
            remaining_balance=remaining,
            image_count=count,
        )

    def format_usage_message(self, result: BalanceChargeResult) -> str:
        return (
            f"本次使用额度 {self.format_amount(result.used_amount)}，"
            f"剩余额度 {self.format_amount(result.remaining_balance)}"
        )

    @staticmethod
    def format_amount(value: float) -> str:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text or "0"

    def _balance_entries(self) -> list[dict[str, Any]]:
        if not hasattr(self.config, "get"):
            return []

        entries = self.config.get("balance_users", [])
        if not isinstance(entries, list):
            return []
        return [entry for entry in entries if isinstance(entry, dict)]

    def _load(self) -> None:
        if not self.path.exists():
            self._balances = {}
            return

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._balances = {}
            return

        self._balances = data if isinstance(data, dict) else {}

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self._balances, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _stored_balance(self, umo: str) -> float:
        record = self._balances.get(umo)
        if not isinstance(record, dict):
            return 0.0
        return self._non_negative_float(record.get("balance"), 0.0)

    def _ensure_record(self, umo: str, balance: float, cost: float) -> None:
        if umo not in self._balances:
            self._set_balance(umo, balance, cost)
            self._save()
            return

        record = self._balances.get(umo)
        if isinstance(record, dict) and record.get("cost_per_image") != cost:
            self._set_balance(umo, balance, cost)
            self._save()

    def _set_balance(self, umo: str, balance: float, cost: float) -> None:
        self._balances[umo] = {
            "balance": self._round_amount(balance),
            "cost_per_image": self._round_amount(cost),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _non_negative_float(self, value: Any, default: float) -> float:
        return max(0.0, self._float(value, default))

    def _float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _round_amount(self, value: float) -> float:
        return round(float(value), 6)
