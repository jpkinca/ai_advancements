"""
Advanced Reinforcement Learning Module

This module provides sophisticated RL algorithms for trading:
- PPO (Proximal Policy Optimization) with custom trading environments
- Multi-agent ensemble systems with regime-aware strategy selection
- Advanced training pipelines with performance optimization
"""

# Import working PPO Trader implementation
from .ppo_trader import (
    PPOTrader,
    TradingSignal as PPOTradingSignal,
    create_ppo_trader
)

# Try to import advanced modules if dependencies are available
try:
    from .multi_agent_system import (
        MarketRegime,
        MarketRegimeDetector,
        SpecializedAgent,
        MetaAgent,
        MultiAgentTradingSystem,
        create_multi_agent_system,
        AgentConfig,
        EnsembleConfig
    )
    _MULTI_AGENT_AVAILABLE = True
except ImportError:
    _MULTI_AGENT_AVAILABLE = False

# Create placeholder multi-agent system if not available
if not _MULTI_AGENT_AVAILABLE:
    class MultiAgentTradingSystem:
        """Placeholder multi-agent system."""
        def __init__(self, *args, **kwargs):
            pass

__all__ = [
    # Working PPO Trader
    'PPOTrader',
    'PPOTradingSignal',
    'create_ppo_trader',
    
    # Multi-Agent Components (if available)
    'MultiAgentTradingSystem',
]

# Add multi-agent components to __all__ if available
if _MULTI_AGENT_AVAILABLE:
    __all__.extend([
        'MarketRegime',
        'MarketRegimeDetector',
        'SpecializedAgent',
        'MetaAgent',
        'create_multi_agent_system',
        'AgentConfig',
        'EnsembleConfig'
    ])
