#!/usr/bin/env python3
"""
IBKR Universal Connection Manager - connect_me.py

Standardized IBKR connection function for the entire trading platform.
Handles connection management, client ID registry, and graceful reconnections.

Author: GitHub Copilot
Date: 2025-01-21
Version: 1.0.0

Usage:
    from ibkr_api.connect_me import connect_me

    # Connect with component name (auto client ID lookup)
    ib = await connect_me("enhanced_technical_analyst")

    # Connect with specific client ID
    ib = await connect_me(client_id=5)

    # Connect with custom parameters
    ib = await connect_me("scanner", host="192.168.1.100", port=4001)
"""

import asyncio
import logging
from typing import Optional, Union, Dict, Any

try:
    from ib_insync import IB

    ib_available = True
except ImportError:
    print("Warning: ib_insync not available. Install with: pip install ib_insync")
    IB = None  # type: ignore
    ib_available = False

# Simple client ID registry for this standalone version
CLIENT_ID_REGISTRY = {
    "enhanced_technical_analyst": 1,
    "scanner": 2,
    "demo": 3,
    "mvp_pipeline": 4,
    "watchlist_loader": 5,
    "data_fetcher": 6,
    "pipeline_1": 7,
    "pipeline_2": 8,
    "pipeline_3": 9,
}

# Global connection tracking
_active_connections: Dict[int, Any] = {}

logger = logging.getLogger(__name__)


class IBKRConnectionError(Exception):
    """Custom exception for IBKR connection errors"""

    pass


def get_component_client_id(component_name: str) -> Optional[int]:
    """Get client ID for a component name."""
    return CLIENT_ID_REGISTRY.get(component_name)


async def connect_me(
    component_name_or_id: Union[str, int],
    host: str = "127.0.0.1",
    port: int = 4002,
    timeout: int = 30,
    force_reconnect: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Universal IBKR connection function for the trading platform.

    Features:
    - Automatic client ID lookup from registry
    - Existing connection detection and management
    - Graceful disconnection and reconnection
    - Comprehensive error handling and logging
    - Connection state tracking

    Args:
        component_name_or_id: Component name (str) or client ID (int)
        host: IBKR Gateway host (default: localhost)
        port: IBKR Gateway port (default: 4002 for paper trading)
        timeout: Connection timeout in seconds
        force_reconnect: Force reconnection even if already connected
        **kwargs: Additional parameters for future use

    Returns:
        Connected IB instance

    Raises:
        IBKRConnectionError: If connection fails
        ValueError: If invalid component name or client ID

    Examples:
        # Connect by component name
        ib = await connect_me("enhanced_technical_analyst")

        # Connect by client ID
        ib = await connect_me(1)

        # Connect to different host/port
        ib = await connect_me("scanner", host="192.168.1.100", port=4001)

        # Force reconnection
        ib = await connect_me("demo", force_reconnect=True)
    """

    if not ib_available:
        raise IBKRConnectionError(
            "ib_insync not available. Install with: pip install ib_insync"
        )

    # Determine client ID
    if isinstance(component_name_or_id, str):
        component_name = component_name_or_id
        client_id = get_component_client_id(component_name)
        if client_id is None:
            raise ValueError(
                f"Unknown component: '{component_name}'. "
                f"Check CLIENT_ID_REGISTRY for valid component names."
            )
    else:
        client_id = component_name_or_id
        component_name = f"client_{client_id}"

    logger.info(
        f"🔌 Connecting {component_name} (Client ID: {client_id}) to IBKR Gateway"
    )
    logger.info(f"   Target: {host}:{port}")

    # Check for existing connection
    if client_id in _active_connections:
        existing_ib = _active_connections[client_id]

        if (
            hasattr(existing_ib, "isConnected")
            and existing_ib.isConnected()
            and not force_reconnect
        ):
            logger.info(
                f"✅ Reusing existing connection for {component_name} (Client ID: {client_id})"
            )
            if hasattr(existing_ib, "client") and hasattr(
                existing_ib.client, "serverVersion"
            ):
                logger.info(f"   Server version: {existing_ib.client.serverVersion()}")
            return existing_ib
        else:
            # Connection exists but is disconnected or force reconnect requested
            logger.info(
                f"🔄 Existing connection found for {component_name}, performing graceful reconnection..."
            )
            await _disconnect_client(client_id, component_name)

    # Create new connection
    try:
        ib = IB()

        logger.info(f"📡 Establishing new connection for {component_name}...")
        await ib.connectAsync(host, port, client_id, timeout=timeout)

        if ib.isConnected():
            # Register successful connection
            _active_connections[client_id] = ib

            logger.info(f"✅ Successfully connected {component_name}!")
            logger.info(f"   Host: {host}:{port}")
            logger.info(f"   Client ID: {client_id}")
            if hasattr(ib, "client") and hasattr(ib.client, "serverVersion"):
                logger.info(f"   Server version: {ib.client.serverVersion()}")
            if hasattr(ib, "client") and hasattr(ib.client, "connectionTime"):
                logger.info(f"   Connection time: {ib.client.connectionTime}")

            return ib
        else:
            raise IBKRConnectionError(
                f"Connection established but not active for {component_name}"
            )

    except ConnectionRefusedError as e:
        error_msg = (
            f"❌ CONNECTION REFUSED for {component_name} (Client ID: {client_id})"
        )
        logger.error(error_msg)
        logger.error(
            "💡 IBKR Gateway 10.30+ is not running or not accepting connections"
        )
        logger.error("")
        logger.error("TO FIX IBKR Gateway 10.30+:")
        logger.error("1. Launch IBKR Gateway (IB Gateway)")
        logger.error("2. Login with your trading account (paper or live)")
        logger.error("3. Gateway 10.30+ automatically enables API connections")
        logger.error("4. Default ports:")
        logger.error("   - Port 4002 for paper trading")
        logger.error("   - Port 4001 for live trading")
        logger.error("5. No additional configuration needed!")
        logger.error("")
        logger.error(
            "NOTE: Gateway 10.30+ simplified API access - no manual settings required!"
        )

        raise IBKRConnectionError(f"Connection refused: {e}") from e

    except asyncio.TimeoutError as e:
        error_msg = (
            f"❌ CONNECTION TIMEOUT for {component_name} after {timeout} seconds"
        )
        logger.error(error_msg)
        logger.error("💡 IBKR Gateway may be slow to respond")
        logger.error("Try increasing timeout or check network connectivity")

        raise IBKRConnectionError(f"Connection timeout: {e}") from e

    except Exception as e:
        error_msg = f"❌ UNEXPECTED ERROR connecting {component_name}: {e}"
        logger.error(error_msg)

        # Clean up failed connection attempt
        if client_id in _active_connections:
            del _active_connections[client_id]

        raise IBKRConnectionError(f"Connection failed: {e}") from e


async def disconnect_me(component_name_or_id: Union[str, int]) -> bool:
    """
    Disconnect a specific component from IBKR Gateway.

    Args:
        component_name_or_id: Component name (str) or client ID (int)

    Returns:
        True if disconnected successfully, False if not connected

    Examples:
        # Disconnect by component name
        await disconnect_me("enhanced_technical_analyst")

        # Disconnect by client ID
        await disconnect_me(1)
    """

    # Determine client ID
    if isinstance(component_name_or_id, str):
        component_name = component_name_or_id
        client_id = get_component_client_id(component_name)
        if client_id is None:
            logger.warning(f"Unknown component: '{component_name}'")
            return False
    else:
        client_id = component_name_or_id
        component_name = f"client_{client_id}"

    return await _disconnect_client(client_id, component_name)


async def _disconnect_client(client_id: int, component_name: str) -> bool:
    """
    Internal function to disconnect a specific client.

    Args:
        client_id: Client ID to disconnect
        component_name: Component name for logging

    Returns:
        True if disconnected, False if not connected
    """
    if client_id not in _active_connections:
        logger.info(f"ℹ️ {component_name} (Client ID: {client_id}) not connected")
        return False

    try:
        ib = _active_connections[client_id]

        if hasattr(ib, "isConnected") and ib.isConnected():
            logger.info(
                f"🔌 Disconnecting {component_name} (Client ID: {client_id})..."
            )
            ib.disconnect()
            logger.info(f"✅ {component_name} disconnected successfully")

        # Clean up tracking
        del _active_connections[client_id]

        return True

    except Exception as e:
        logger.error(f"❌ Error disconnecting {component_name}: {e}")

        # Force cleanup even if disconnect failed
        if client_id in _active_connections:
            del _active_connections[client_id]

        return False


async def disconnect_all() -> int:
    """
    Disconnect all active IBKR connections.

    Returns:
        Number of connections disconnected
    """
    logger.info("🔌 Disconnecting all IBKR connections...")

    disconnected_count = 0
    client_ids = list(
        _active_connections.keys()
    )  # Copy to avoid modification during iteration

    for client_id in client_ids:
        component_name = f"client_{client_id}"
        if await _disconnect_client(client_id, component_name):
            disconnected_count += 1

    logger.info(f"✅ Disconnected {disconnected_count} connections")
    return disconnected_count


def get_connection_status() -> Dict[int, Dict[str, Any]]:
    """
    Get status of all IBKR connections.

    Returns:
        Dictionary with client ID as key and status info as value
    """
    status = {}

    for client_id, ib in _active_connections.items():
        component_name = f"client_{client_id}"

        is_connected = hasattr(ib, "isConnected") and ib.isConnected() if ib else False
        server_version = None
        connection_time = None

        if ib and is_connected:
            if hasattr(ib, "client") and hasattr(ib.client, "serverVersion"):
                server_version = ib.client.serverVersion()
            if hasattr(ib, "client") and hasattr(ib.client, "connectionTime"):
                connection_time = str(ib.client.connectionTime)

        status[client_id] = {
            "component_name": component_name,
            "is_connected": is_connected,
            "server_version": server_version,
            "connection_time": connection_time,
        }

    return status


def is_connected(component_name_or_id: Union[str, int]) -> bool:
    """
    Check if a component is connected to IBKR Gateway.

    Args:
        component_name_or_id: Component name (str) or client ID (int)

    Returns:
        True if connected, False otherwise
    """
    # Determine client ID
    if isinstance(component_name_or_id, str):
        client_id = get_component_client_id(component_name_or_id)
        if client_id is None:
            return False
    else:
        client_id = component_name_or_id

    if client_id not in _active_connections:
        return False

    ib = _active_connections[client_id]
    return hasattr(ib, "isConnected") and ib.isConnected() if ib else False


async def get_connection(component_name_or_id: Union[str, int]) -> Optional[Any]:
    """
    Get existing connection for a component without creating new one.

    Args:
        component_name_or_id: Component name (str) or client ID (int)

    Returns:
        IB instance if connected, None otherwise
    """
    # Determine client ID
    if isinstance(component_name_or_id, str):
        client_id = get_component_client_id(component_name_or_id)
        if client_id is None:
            return None
    else:
        client_id = component_name_or_id

    if client_id not in _active_connections:
        return None

    ib = _active_connections[client_id]
    if ib and hasattr(ib, "isConnected") and ib.isConnected():
        return ib

    return None


# Demo and testing functions
async def demo_connection_manager():
    """
    Demonstrate the connection manager capabilities.
    """
    print("🚀 IBKR Connection Manager Demo")
    print("=" * 50)

    try:
        # Test 1: Connect by component name
        print("\n📡 Test 1: Connecting Enhanced Technical Analyst...")
        ib1 = await connect_me("enhanced_technical_analyst")
        print(
            f"✅ Connected! Server version: {getattr(getattr(ib1, 'client', None), 'serverVersion', lambda: 'Unknown')()}"
        )

        # Test 2: Connect by client ID
        print("\n📡 Test 2: Connecting Scanner (Client ID 6)...")
        ib2 = await connect_me(6)
        print(
            f"✅ Connected! Server version: {getattr(getattr(ib2, 'client', None), 'serverVersion', lambda: 'Unknown')()}"
        )

        # Test 3: Try to reconnect same component (should reuse)
        print("\n📡 Test 3: Reconnecting Enhanced Technical Analyst...")
        ib3 = await connect_me("enhanced_technical_analyst")
        print(f"✅ Reused connection: {ib1 is ib3}")

        # Test 4: Check connection status
        print("\n📊 Connection Status:")
        status = get_connection_status()
        for client_id, info in status.items():
            print(
                f"   Client {client_id}: {info['component_name']} - Connected: {info['is_connected']}"
            )

        # Test 5: Disconnect specific component
        print("\n🔌 Test 5: Disconnecting Scanner...")
        await disconnect_me(6)

        # Test 6: Final status check
        print("\n📊 Final Status:")
        status = get_connection_status()
        for client_id, info in status.items():
            print(
                f"   Client {client_id}: {info['component_name']} - Connected: {info['is_connected']}"
            )

    except IBKRConnectionError as e:
        print(f"❌ Connection failed: {e}")
        print("   Make sure IBKR Gateway is running on port 4002")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Cleanup
        print("\n🧹 Cleaning up connections...")
        disconnected = await disconnect_all()
        print(f"✅ Disconnected {disconnected} connections")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run demo
    asyncio.run(demo_connection_manager())
