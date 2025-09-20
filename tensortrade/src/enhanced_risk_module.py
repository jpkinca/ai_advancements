"""
Enhanced Risk Management Module for TensorTrade

This module provides production-grade risk management capabilities including:
- Dynamic position sizing based on volatility
- Portfolio exposure limits and monitoring
- Real-time risk metrics calculation
- Circuit breaker controls for loss prevention
- Comprehensive risk reporting

Author: TensorTrade Development Team
Created: August 17, 2025
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math

from .db_utils import DatabaseManager


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Risk alert types"""
    POSITION_LIMIT = "position_limit"
    EXPOSURE_LIMIT = "exposure_limit"
    DRAWDOWN_WARNING = "drawdown_warning"
    VOLATILITY_SPIKE = "volatility_spike"
    CIRCUIT_BREAKER = "circuit_breaker"
    MARGIN_CALL = "margin_call"


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    # Position limits
    max_position_size: float = 0.05  # 5% of portfolio per position
    max_sector_exposure: float = 0.30  # 30% per sector
    max_gross_exposure: float = 1.0   # 100% gross exposure
    max_net_exposure: float = 0.8     # 80% net exposure
    
    # Drawdown limits
    max_daily_drawdown: float = 0.03  # 3% daily loss limit
    max_total_drawdown: float = 0.10  # 10% total drawdown limit
    
    # Volatility limits
    max_portfolio_volatility: float = 0.20  # 20% annualized volatility
    volatility_lookback: int = 30  # Days for volatility calculation
    
    # Circuit breaker settings
    circuit_breaker_threshold: float = 0.05  # 5% loss triggers circuit breaker
    circuit_breaker_cooldown: int = 300  # 5 minutes cooldown
    
    # Risk capacity
    max_var_95: float = 0.02  # 2% VaR at 95% confidence
    max_leverage: float = 2.0  # Maximum leverage ratio


@dataclass
class Position:
    """Position data"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    sector: str = "Unknown"
    
    @property
    def weight(self) -> float:
        """Position weight in portfolio"""
        return abs(self.market_value)
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity  float:
        """Gross exposure (sum of absolute position values)"""
        return sum(abs(pos.market_value) for pos in self.positions.values())
    
    @property
    def net_exposure(self) -> float:
        """Net exposure (sum of signed position values)"""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def leverage(self) -> float:
        """Portfolio leverage ratio"""
        return self.gross_exposure / max(self.total_value, 1)
    
    def get_sector_exposure(self) -> Dict[str, float]:
        """Calculate sector exposure"""
        sector_exposure = {}
        for position in self.positions.values():
            sector = position.sector
            if sector not in sector_exposure:
                sector_exposure[sector] = 0
            sector_exposure[sector] += abs(position.market_value)
        return sector_exposure


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    # Portfolio metrics
    total_value: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    
    # Risk measures
    portfolio_volatility: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    max_drawdown: float
    daily_drawdown: float
    
    # Position metrics
    largest_position: float
    position_concentration: float  # HHI index
    sector_concentrations: Dict[str, float]
    
    # Alert status
    risk_level: RiskLevel
    active_alerts: List['RiskAlert']
    
    # Performance metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None


@dataclass
class RiskAlert:
    """Risk alert notification"""
    alert_type: AlertType
    risk_level: RiskLevel
    message: str
    timestamp: datetime
    symbol: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    
    def __str__(self) -> str:
        return f"[{self.risk_level.value.upper()}] {self.alert_type.value}: {self.message}"


class VolatilityCalculator:
    """Volatility calculation utilities"""
    
    @staticmethod
    def calculate_realized_volatility(returns: pd.Series, 
                                    window: int = 30,
                                    annualize: bool = True) -> float:
        """Calculate realized volatility"""
        if len(returns)  float:
        """Calculate EWMA volatility"""
        if len(returns)  float:
        """Calculate historical VaR"""
        if len(returns)  float:
        """Calculate parametric VaR assuming normal distribution"""
        if len(returns)  None:
        """Update current portfolio state"""
        # Calculate P&L
        total_pnl = total_value - self.initial_capital
        
        # Calculate daily P&L (simplified - would need previous day value)
        daily_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        
        self.portfolio = Portfolio(
            cash=cash,
            positions=positions,
            total_value=total_value,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl
        )
        
        # Update returns history
        if len(self.returns_history) > 0:
            prev_value = self.initial_capital + self.returns_history.sum() * self.initial_capital
            current_return = (total_value - prev_value) / prev_value
            self.returns_history = pd.concat([
                self.returns_history, 
                pd.Series([current_return], index=[datetime.now()])
            ])
        else:
            # First update
            self.returns_history = pd.Series([0.0], index=[datetime.now()])
        
        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=365)
        self.returns_history = self.returns_history[self.returns_history.index >= cutoff_date]
    
    def calculate_dynamic_position_size(self, 
                                      symbol: str,
                                      volatility: float,
                                      confidence: float = 1.0) -> float:
        """
        Calculate dynamic position size based on volatility and portfolio heat
        
        Args:
            symbol: Symbol to calculate position size for
            volatility: Asset volatility (annualized)
            confidence: Model confidence (0-1)
            
        Returns:
            Recommended position size as fraction of portfolio
        """
        if not self.portfolio:
            return 0.0
        
        # Base position size from risk limits
        base_size = self.risk_limits.max_position_size
        
        # Adjust for volatility (inverse relationship)
        vol_adjustment = min(1.0, 0.20 / max(volatility, 0.01))  # Target 20% volatility
        
        # Adjust for confidence
        confidence_adjustment = confidence
        
        # Adjust for current portfolio heat
        current_exposure = self.portfolio.gross_exposure / self.portfolio.total_value
        heat_adjustment = max(0.1, 1.0 - current_exposure)
        
        # Calculate final position size
        position_size = base_size * vol_adjustment * confidence_adjustment * heat_adjustment
        
        # Ensure within limits
        position_size = min(position_size, self.risk_limits.max_position_size)
        
        self.logger.debug(f"Dynamic position size for {symbol}: {position_size:.3f} "
                         f"(vol_adj: {vol_adjustment:.3f}, conf_adj: {confidence_adjustment:.3f}, "
                         f"heat_adj: {heat_adjustment:.3f})")
        
        return position_size
    
    def validate_trade_request(self, 
                             symbol: str,
                             quantity: float,
                             price: float) -> Tuple[bool, List[str]]:
        """
        Validate trade request against risk limits
        
        Args:
            symbol: Symbol to trade
            quantity: Quantity to trade (positive for buy, negative for sell)
            price: Expected execution price
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        if not self.portfolio:
            return False, ["Portfolio state not available"]
        
        violations = []
        
        # Calculate proposed position value
        trade_value = abs(quantity * price)
        
        # Check circuit breaker
        if self.circuit_breaker_active:
            violations.append("Circuit breaker is active - trading suspended")
        
        # Check position size limit
        position_weight = trade_value / self.portfolio.total_value
        if position_weight > self.risk_limits.max_position_size:
            violations.append(f"Position size {position_weight:.3f} exceeds limit "
                            f"{self.risk_limits.max_position_size:.3f}")
        
        # Check exposure limits
        new_gross_exposure = self.portfolio.gross_exposure + trade_value
        gross_exposure_ratio = new_gross_exposure / self.portfolio.total_value
        if gross_exposure_ratio > self.risk_limits.max_gross_exposure:
            violations.append(f"Gross exposure {gross_exposure_ratio:.3f} would exceed limit "
                            f"{self.risk_limits.max_gross_exposure:.3f}")
        
        # Check sector exposure
        sector = self.sector_mapping.get(symbol, "Unknown")
        sector_exposures = self.portfolio.get_sector_exposure()
        current_sector_exposure = sector_exposures.get(sector, 0)
        new_sector_exposure = (current_sector_exposure + trade_value) / self.portfolio.total_value
        if new_sector_exposure > self.risk_limits.max_sector_exposure:
            violations.append(f"Sector exposure for {sector} {new_sector_exposure:.3f} "
                            f"would exceed limit {self.risk_limits.max_sector_exposure:.3f}")
        
        # Check leverage
        new_leverage = new_gross_exposure / self.portfolio.total_value
        if new_leverage > self.risk_limits.max_leverage:
            violations.append(f"Leverage {new_leverage:.3f} would exceed limit "
                            f"{self.risk_limits.max_leverage:.3f}")
        
        return len(violations) == 0, violations
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        if not self.portfolio:
            return self._default_risk_metrics()
        
        # Portfolio basic metrics
        gross_exposure = self.portfolio.gross_exposure
        net_exposure = self.portfolio.net_exposure
        leverage = self.portfolio.leverage
        
        # Volatility calculation
        portfolio_volatility = 0.0
        if len(self.returns_history) > 30:
            portfolio_volatility = self.vol_calculator.calculate_realized_volatility(
                self.returns_history, window=30
            )
        
        # VaR calculation
        var_95 = 0.0
        var_99 = 0.0
        if len(self.returns_history) > 50:
            var_95 = self.var_calculator.calculate_historical_var(
                self.returns_history, confidence=0.95
            )
            var_99 = self.var_calculator.calculate_historical_var(
                self.returns_history, confidence=0.99
            )
        
        # Drawdown calculation
        if len(self.returns_history) > 1:
            cumulative_returns = (1 + self.returns_history).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdowns.min())
            daily_drawdown = abs(self.portfolio.daily_pnl / self.portfolio.total_value)
        else:
            max_drawdown = 0.0
            daily_drawdown = 0.0
        
        # Position concentration
        if self.portfolio.positions:
            position_weights = [abs(pos.market_value) / self.portfolio.total_value 
                              for pos in self.portfolio.positions.values()]
            largest_position = max(position_weights) if position_weights else 0.0
            
            # Herfindahl-Hirschman Index for concentration
            position_concentration = sum(w**2 for w in position_weights)
        else:
            largest_position = 0.0
            position_concentration = 0.0
        
        # Sector concentrations
        sector_exposures = self.portfolio.get_sector_exposure()
        sector_concentrations = {
            sector: exposure / self.portfolio.total_value 
            for sector, exposure in sector_exposures.items()
        }
        
        # Determine risk level
        risk_level = self._assess_risk_level(
            leverage, portfolio_volatility, max_drawdown, daily_drawdown
        )
        
        # Performance ratios (if sufficient history)
        sharpe_ratio = None
        sortino_ratio = None
        calmar_ratio = None
        
        if len(self.returns_history) > 60:  # At least 2 months of data
            mean_return = self.returns_history.mean() * 252  # Annualized
            std_return = self.returns_history.std() * np.sqrt(252)  # Annualized
            
            if std_return > 0:
                sharpe_ratio = mean_return / std_return
            
            # Sortino ratio (downside deviation)
            downside_returns = self.returns_history[self.returns_history  0:
                downside_deviation = downside_returns.std() * np.sqrt(252)
                if downside_deviation > 0:
                    sortino_ratio = mean_return / downside_deviation
            
            # Calmar ratio
            if max_drawdown > 0:
                calmar_ratio = mean_return / max_drawdown
        
        return RiskMetrics(
            total_value=self.portfolio.total_value,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            leverage=leverage,
            portfolio_volatility=portfolio_volatility,
            var_95=var_95,
            var_99=var_99,
            max_drawdown=max_drawdown,
            daily_drawdown=daily_drawdown,
            largest_position=largest_position,
            position_concentration=position_concentration,
            sector_concentrations=sector_concentrations,
            risk_level=risk_level,
            active_alerts=self.get_active_alerts(),
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio
        )
    
    def monitor_real_time_risk(self) -> List[RiskAlert]:
        """Monitor real-time risk and generate alerts"""
        if not self.portfolio:
            return []
        
        new_alerts = []
        
        # Check drawdown limits
        daily_drawdown_pct = abs(self.portfolio.daily_pnl / self.portfolio.total_value)
        if daily_drawdown_pct > self.risk_limits.max_daily_drawdown:
            alert = RiskAlert(
                alert_type=AlertType.DRAWDOWN_WARNING,
                risk_level=RiskLevel.HIGH,
                message=f"Daily drawdown {daily_drawdown_pct:.2%} exceeds limit "
                       f"{self.risk_limits.max_daily_drawdown:.2%}",
                timestamp=datetime.now(),
                value=daily_drawdown_pct,
                threshold=self.risk_limits.max_daily_drawdown
            )
            new_alerts.append(alert)
        
        # Check circuit breaker
        if daily_drawdown_pct > self.risk_limits.circuit_breaker_threshold:
            if not self.circuit_breaker_active:
                self._activate_circuit_breaker()
                alert = RiskAlert(
                    alert_type=AlertType.CIRCUIT_BREAKER,
                    risk_level=RiskLevel.CRITICAL,
                    message=f"Circuit breaker activated due to {daily_drawdown_pct:.2%} loss",
                    timestamp=datetime.now(),
                    value=daily_drawdown_pct,
                    threshold=self.risk_limits.circuit_breaker_threshold
                )
                new_alerts.append(alert)
        
        # Check exposure limits
        leverage = self.portfolio.leverage
        if leverage > self.risk_limits.max_leverage:
            alert = RiskAlert(
                alert_type=AlertType.EXPOSURE_LIMIT,
                risk_level=RiskLevel.MEDIUM,
                message=f"Leverage {leverage:.2f} exceeds limit {self.risk_limits.max_leverage:.2f}",
                timestamp=datetime.now(),
                value=leverage,
                threshold=self.risk_limits.max_leverage
            )
            new_alerts.append(alert)
        
        # Check position concentration
        for symbol, position in self.portfolio.positions.items():
            position_weight = abs(position.market_value) / self.portfolio.total_value
            if position_weight > self.risk_limits.max_position_size:
                alert = RiskAlert(
                    alert_type=AlertType.POSITION_LIMIT,
                    risk_level=RiskLevel.MEDIUM,
                    message=f"Position {symbol} weight {position_weight:.2%} exceeds limit "
                           f"{self.risk_limits.max_position_size:.2%}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    value=position_weight,
                    threshold=self.risk_limits.max_position_size
                )
                new_alerts.append(alert)
        
        # Check sector concentration
        sector_exposures = self.portfolio.get_sector_exposure()
        for sector, exposure in sector_exposures.items():
            sector_weight = exposure / self.portfolio.total_value
            if sector_weight > self.risk_limits.max_sector_exposure:
                alert = RiskAlert(
                    alert_type=AlertType.EXPOSURE_LIMIT,
                    risk_level=RiskLevel.MEDIUM,
                    message=f"Sector {sector} exposure {sector_weight:.2%} exceeds limit "
                           f"{self.risk_limits.max_sector_exposure:.2%}",
                    timestamp=datetime.now(),
                    value=sector_weight,
                    threshold=self.risk_limits.max_sector_exposure
                )
                new_alerts.append(alert)
        
        # Add new alerts
        self.risk_alerts.extend(new_alerts)
        
        # Clean old alerts (keep last 100)
        if len(self.risk_alerts) > 100:
            self.risk_alerts = self.risk_alerts[-100:]
        
        return new_alerts
    
    def _activate_circuit_breaker(self):
        """Activate circuit breaker"""
        self.circuit_breaker_active = True
        self.circuit_breaker_activated_at = datetime.now()
        self.logger.critical("CIRCUIT BREAKER ACTIVATED - Trading suspended")
    
    def check_circuit_breaker_reset(self) -> bool:
        """Check if circuit breaker can be reset"""
        if not self.circuit_breaker_active:
            return True
        
        if self.circuit_breaker_activated_at:
            time_elapsed = datetime.now() - self.circuit_breaker_activated_at
            if time_elapsed.total_seconds() > self.risk_limits.circuit_breaker_cooldown:
                self.circuit_breaker_active = False
                self.circuit_breaker_activated_at = None
                self.logger.info("Circuit breaker reset - Trading resumed")
                return True
        
        return False
    
    def _assess_risk_level(self, 
                          leverage: float,
                          volatility: float,
                          max_drawdown: float,
                          daily_drawdown: float) -> RiskLevel:
        """Assess overall risk level"""
        risk_score = 0
        
        # Leverage risk
        if leverage > self.risk_limits.max_leverage * 0.8:
            risk_score += 2
        elif leverage > self.risk_limits.max_leverage * 0.6:
            risk_score += 1
        
        # Volatility risk
        if volatility > self.risk_limits.max_portfolio_volatility * 1.2:
            risk_score += 2
        elif volatility > self.risk_limits.max_portfolio_volatility:
            risk_score += 1
        
        # Drawdown risk
        if max_drawdown > self.risk_limits.max_total_drawdown * 0.8:
            risk_score += 2
        elif max_drawdown > self.risk_limits.max_total_drawdown * 0.6:
            risk_score += 1
        
        if daily_drawdown > self.risk_limits.max_daily_drawdown * 0.8:
            risk_score += 1
        
        # Map score to risk level
        if risk_score >= 4:
            return RiskLevel.CRITICAL
        elif risk_score >= 2:
            return RiskLevel.HIGH
        elif risk_score >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def get_active_alerts(self) -> List[RiskAlert]:
        """Get recent active alerts"""
        # Return alerts from last hour
        cutoff_time = datetime.now() - timedelta(hours=1)
        return [alert for alert in self.risk_alerts if alert.timestamp >= cutoff_time]
    
    def _default_risk_metrics(self) -> RiskMetrics:
        """Return default risk metrics when portfolio not available"""
        return RiskMetrics(
            total_value=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            leverage=0.0,
            portfolio_volatility=0.0,
            var_95=0.0,
            var_99=0.0,
            max_drawdown=0.0,
            daily_drawdown=0.0,
            largest_position=0.0,
            position_concentration=0.0,
            sector_concentrations={},
            risk_level=RiskLevel.LOW,
            active_alerts=[]
        )
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        metrics = self.calculate_risk_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': {
                'total_value': metrics.total_value,
                'gross_exposure': metrics.gross_exposure,
                'net_exposure': metrics.net_exposure,
                'leverage': metrics.leverage,
                'cash': self.portfolio.cash if self.portfolio else 0.0,
                'num_positions': len(self.portfolio.positions) if self.portfolio else 0
            },
            'risk_metrics': {
                'portfolio_volatility': metrics.portfolio_volatility,
                'var_95': metrics.var_95,
                'var_99': metrics.var_99,
                'max_drawdown': metrics.max_drawdown,
                'daily_drawdown': metrics.daily_drawdown,
                'risk_level': metrics.risk_level.value
            },
            'concentration_metrics': {
                'largest_position': metrics.largest_position,
                'position_concentration_hhi': metrics.position_concentration,
                'sector_concentrations': metrics.sector_concentrations
            },
            'performance_metrics': {
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'calmar_ratio': metrics.calmar_ratio
            },
            'risk_limits': {
                'max_position_size': self.risk_limits.max_position_size,
                'max_sector_exposure': self.risk_limits.max_sector_exposure,
                'max_gross_exposure': self.risk_limits.max_gross_exposure,
                'max_daily_drawdown': self.risk_limits.max_daily_drawdown,
                'max_total_drawdown': self.risk_limits.max_total_drawdown,
                'circuit_breaker_threshold': self.risk_limits.circuit_breaker_threshold
            },
            'alerts': {
                'circuit_breaker_active': self.circuit_breaker_active,
                'active_alerts': [
                    {
                        'type': alert.alert_type.value,
                        'level': alert.risk_level.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'symbol': alert.symbol,
                        'value': alert.value,
                        'threshold': alert.threshold
                    }
                    for alert in metrics.active_alerts
                ]
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the risk management system
    db_manager = DatabaseManager()
    risk_manager = EnhancedRiskManager(db_manager, initial_capital=100000)
    
    # Create sample positions
    positions = {
        'AAPL': Position(
            symbol='AAPL',
            quantity=100,
            avg_price=150.0,
            current_price=155.0,
            market_value=15500.0,
            unrealized_pnl=500.0,
            sector='Technology'
        ),
        'MSFT': Position(
            symbol='MSFT',
            quantity=50,
            avg_price=300.0,
            current_price=305.0,
            market_value=15250.0,
            unrealized_pnl=250.0,
            sector='Technology'
        )
    }
    
    # Update portfolio state
    risk_manager.update_portfolio_state(
        positions=positions,
        cash=70000.0,
        total_value=100750.0
    )
    
    # Calculate risk metrics
    metrics = risk_manager.calculate_risk_metrics()
    print("Risk Metrics:")
    print(f"  Risk Level: {metrics.risk_level.value}")
    print(f"  Leverage: {metrics.leverage:.2f}")
    print(f"  Largest Position: {metrics.largest_position:.2%}")
    
    # Test trade validation
    is_valid, violations = risk_manager.validate_trade_request('GOOGL', 50, 2800)
    print(f"\nTrade Validation: {'VALID' if is_valid else 'INVALID'}")
    if violations:
        for violation in violations:
            print(f"  - {violation}")
    
    # Monitor risk
    alerts = risk_manager.monitor_real_time_risk()
    if alerts:
        print("\nRisk Alerts:")
        for alert in alerts:
            print(f"  {alert}")
    
    # Generate risk report
    report = risk_manager.generate_risk_report()
    print(f"\nRisk Report Generated: {len(report)} sections")
