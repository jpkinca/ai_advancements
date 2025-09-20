#!/usr/bin/env python3
"""Minimal vendored connect_me for tensortrade real IBKR access.
Copied from TradeAppComponents/ibkr_api/connect_me.py (trimmed comments).
"""
import asyncio
import logging
from typing import Optional, Union, Dict, Any
try:
    from ib_insync import IB  # type: ignore
    ib_available = True
except ImportError:  # pragma: no cover
    IB = None  # type: ignore
    ib_available = False
from .client_id_registry import client_id_manager, get_component_client_id
_active_connections: Dict[int, Any] = {}
logger = logging.getLogger(__name__)
class IBKRConnectionError(Exception):
    pass
async def connect_me(
    component_name_or_id: Union[str, int],
    host: str = "127.0.0.1",
    port: int = 4002,
    timeout: int = 30,
    force_reconnect: bool = False,
    auto_fallback: bool = True,
    fallback_client_ids: Optional[list[int]] = None,
    stable_wait_seconds: float = 0.35,
    **kwargs: Any,
) -> Any:
    if not ib_available:
        raise IBKRConnectionError("ib_insync not available. Install with: pip install ib_insync")
    if isinstance(component_name_or_id, str):
        component_name = component_name_or_id
        base_client_id = get_component_client_id(component_name)
        if base_client_id is None:
            raise ValueError(f"Unknown component: {component_name}")
    else:
        base_client_id = component_name_or_id
        component_name = f"client_{base_client_id}"
    candidate_ids: list[int] = [base_client_id]
    if fallback_client_ids:
        for cid in fallback_client_ids:
            if cid not in candidate_ids:
                candidate_ids.append(cid)
    lowered = component_name.lower()
    if auto_fallback and any(k in lowered for k in ["scanner", "scan"]):
        for alt in [6, 106, base_client_id + 1, base_client_id + 10]:
            if isinstance(alt, int) and alt not in candidate_ids:
                candidate_ids.append(int(alt))
    last_error: Optional[Exception] = None
    for attempt_index, client_id in enumerate(candidate_ids, start=1):
        logger.info(f"Connecting {component_name} (candidate clientId {client_id}) attempt {attempt_index}/{len(candidate_ids)}")
        if client_id in _active_connections and not force_reconnect:
            existing_ib = _active_connections[client_id]
            if hasattr(existing_ib, "isConnected") and existing_ib.isConnected():
                return existing_ib
        try:
            ib = IB()  # type: ignore
            await ib.connectAsync(host, port, client_id, timeout=timeout)
            await asyncio.sleep(stable_wait_seconds)
            if not (hasattr(ib, "isConnected") and ib.isConnected()):
                try:
                    ib.disconnect()
                except Exception:
                    pass
                last_error = IBKRConnectionError(f"Unstable connection clientId {client_id}")
                continue
            _active_connections[client_id] = ib
            client_id_manager.register_connection(client_id, component_name)
            return ib
        except Exception as e:  # pragma: no cover (network dependent)
            last_error = e
            continue
    if last_error:
        raise IBKRConnectionError(f"Connection failed after {len(candidate_ids)} attempts: {last_error}") from last_error
    raise IBKRConnectionError("Connection failed with no specific error captured")
async def disconnect_me(component_name_or_id: Union[str, int]) -> bool:
    if isinstance(component_name_or_id, str):
        client_id = get_component_client_id(component_name_or_id)
        if client_id is None:
            return False
    else:
        client_id = component_name_or_id
    return await _disconnect_client(client_id)
async def _disconnect_client(client_id: int) -> bool:
    if client_id not in _active_connections:
        return False
    try:
        ib = _active_connections[client_id]
        if hasattr(ib, "isConnected") and ib.isConnected():
            ib.disconnect()
        del _active_connections[client_id]
        client_id_manager.unregister_connection(client_id)
        return True
    except Exception:
        if client_id in _active_connections:
            del _active_connections[client_id]
        client_id_manager.unregister_connection(client_id)
        return False
async def disconnect_all() -> int:
    ids = list(_active_connections.keys())
    count = 0
    for cid in ids:
        if await _disconnect_client(cid):
            count += 1
    return count
