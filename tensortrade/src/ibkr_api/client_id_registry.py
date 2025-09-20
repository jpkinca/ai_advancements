#!/usr/bin/env python3
"""Vendored minimal client_id_registry for tensortrade.
Keep in sync with TradeAppComponents/ibkr_api/client_id_registry.py if updated upstream.
"""
from enum import Enum
from typing import Dict, Optional

class ClientIDRegistry(Enum):
    ENHANCED_TECHNICAL_ANALYST = 1
    MARKET_DATA_STREAMER = 2
    ORDER_MANAGER = 3
    PORTFOLIO_MANAGER = 4
    RISK_MANAGER = 5
    SCANNER = 106
    CHART_VISUALIZER = 7
    MARKET_PULSE_MONITOR = 8
    TRADE_EXECUTOR = 9
    POSITION_MANAGER = 10
    AI_TECHNICAL_ANALYST = 11
    HISTORICAL_DATA_COLLECTOR = 16
    PIPELINE_1 = 101
    PIPELINE_2 = 102
    PIPELINE_3 = 103

class ClientIDManager:
    def __init__(self):
        self._active_connections: Dict[int, str] = {}
        self._reserved_ids: Dict[str, int] = {c.name.lower(): c.value for c in ClientIDRegistry}
    def get_client_id(self, component_name: str) -> Optional[int]:
        key = component_name.upper().replace(" ", "_").replace("-", "_")
        try:
            return ClientIDRegistry[key].value
        except KeyError:
            for item in ClientIDRegistry:
                if key in item.name or item.name in key:
                    return item.value
        return None
    def register_connection(self, client_id: int, component_name: str) -> bool:
        if client_id in self._active_connections and self._active_connections[client_id] != component_name:
            return False
        self._active_connections[client_id] = component_name
        return True
    def unregister_connection(self, client_id: int) -> bool:
        return self._active_connections.pop(client_id, None) is not None
    def get_active_connections(self) -> Dict[int, str]:
        return self._active_connections.copy()

client_id_manager = ClientIDManager()

def get_component_client_id(component_name: str) -> Optional[int]:
    return client_id_manager.get_client_id(component_name)
