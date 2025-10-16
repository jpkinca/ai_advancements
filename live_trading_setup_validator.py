#!/usr/bin/env python3
"""
Live Trading Setup Validator
Quick assessment of current trading infrastructure readiness for strategic deployment
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class LiveTradingSetupValidator:
    """Validate current live trading infrastructure and readiness"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.validation_results = {}
        
    def run_comprehensive_validation(self):
        """Run complete live trading setup validation"""
        logger.info("🔍 LIVE TRADING SETUP VALIDATION")
        logger.info("="*50)
        
        validations = [
            ("IBKR Gateway Connection", self.validate_ibkr_gateway),
            ("Database Infrastructure", self.validate_database),
            ("AI Module Availability", self.validate_ai_modules),
            ("Paper Trading Capability", self.validate_paper_trading),
            ("Data Pipeline Readiness", self.validate_data_pipeline),
            ("Risk Management Systems", self.validate_risk_management),
            ("Monitoring Infrastructure", self.validate_monitoring),
            ("Production Readiness", self.assess_production_readiness)
        ]
        
        total_score = 0
        max_score = len(validations)
        
        for name, validator in validations:
            logger.info(f"\n🔎 {name}...")
            try:
                result = validator()
                self.validation_results[name] = result
                if result.get('status') == 'pass':
                    total_score += 1
                    logger.info(f"✅ {name}: PASS - {result.get('message', 'OK')}")
                else:
                    logger.warning(f"⚠️ {name}: {result.get('status', 'FAIL')} - {result.get('message', 'Issues detected')}")
            except Exception as e:
                logger.error(f"❌ {name}: ERROR - {e}")
                self.validation_results[name] = {'status': 'error', 'message': str(e)}
        
        # Generate summary
        readiness_score = (total_score / max_score) * 100
        
        logger.info("\n" + "="*50)
        logger.info("📊 LIVE TRADING READINESS ASSESSMENT")
        logger.info("="*50)
        logger.info(f"Overall Readiness Score: {readiness_score:.1f}% ({total_score}/{max_score})")
        
        if readiness_score >= 80:
            logger.info("🚀 READY FOR IMMEDIATE DEPLOYMENT")
            logger.info("Recommendation: Proceed with strategic deployment plan")
        elif readiness_score >= 60:
            logger.info("⚠️ MOSTLY READY - MINOR ISSUES TO ADDRESS")
            logger.info("Recommendation: Fix identified issues then deploy")
        else:
            logger.info("❌ NOT READY - SIGNIFICANT SETUP REQUIRED")
            logger.info("Recommendation: Address infrastructure issues before deployment")
        
        self.generate_readiness_report(readiness_score)
        return readiness_score
    
    def validate_ibkr_gateway(self):
        """Validate IBKR Gateway connectivity"""
        try:
            from ib_insync import IB
            
            # Test both paper and live gateway ports
            gateways_tested = []
            
            for port, environment in [(4002, 'Paper Trading'), (4001, 'Live Trading')]:
                try:
                    ib = IB()
                    ib.connect('127.0.0.1', port, clientId=999, timeout=5)
                    
                    if ib.isConnected():
                        version = ib.client.serverVersion()
                        gateways_tested.append(f"{environment} (port {port}, v{version})")
                        ib.disconnect()
                    
                except Exception as e:
                    pass  # Connection failed for this port
            
            if gateways_tested:
                return {
                    'status': 'pass',
                    'message': f"Connected to: {', '.join(gateways_tested)}",
                    'details': {'available_gateways': gateways_tested}
                }
            else:
                return {
                    'status': 'fail',
                    'message': 'No IBKR Gateway connections available',
                    'recommendation': 'Start IBKR Gateway (Paper Trading: port 4002, Live: port 4001)'
                }
                
        except ImportError:
            return {
                'status': 'fail',
                'message': 'ib_insync module not available',
                'recommendation': 'Install: pip install ib_insync'
            }
    
    def validate_database(self):
        """Validate database infrastructure"""
        try:
            # Try multiple import paths for database manager
            db_manager = None
            import_paths = [
                'modules.database.railway_db_manager',
                'TradeAppComponents_fresh.modules.database.railway_db_manager'
            ]
            
            for import_path in import_paths:
                try:
                    module = __import__(import_path, fromlist=['RailwayPostgreSQLManager'])
                    RailwayPostgreSQLManager = getattr(module, 'RailwayPostgreSQLManager')
                    db_manager = RailwayPostgreSQLManager()
                    break
                except ImportError:
                    continue
            
            if not db_manager:
                return {
                    'status': 'fail',
                    'message': 'Database manager not found',
                    'recommendation': 'Ensure railway_db_manager.py is available'
                }
            
            # Test database connection
            session = db_manager.get_session()
            
            # Check for AI trading tables
            result = session.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema IN ('ai_trading', 'level_ii_data')
            """)
            
            table_count = result.fetchone()[0]
            session.close()
            
            if table_count >= 5:  # Expect AI trading + Level II tables
                return {
                    'status': 'pass',
                    'message': f'Database connected with {table_count} AI trading tables',
                    'details': {'table_count': table_count}
                }
            else:
                return {
                    'status': 'partial',
                    'message': f'Database connected but only {table_count} AI tables found',
                    'recommendation': 'Run database schema setup scripts'
                }
                
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Database validation failed: {e}',
                'recommendation': 'Check Railway PostgreSQL connection and credentials'
            }
    
    def validate_ai_modules(self):
        """Validate AI module availability"""
        try:
            # Check if AI module directories exist
            src_path = self.base_path / 'src'
            
            expected_modules = [
                'reinforcement_learning',
                'genetic_optimization', 
                'sparse_spectrum'
            ]
            
            available_modules = []
            missing_modules = []
            
            for module in expected_modules:
                module_path = src_path / module
                if module_path.exists() and (module_path / '__init__.py').exists():
                    available_modules.append(module)
                else:
                    missing_modules.append(module)
            
            # Test imports
            importable_modules = []
            sys.path.append(str(src_path))
            
            for module in available_modules:
                try:
                    __import__(module)
                    importable_modules.append(module)
                except ImportError as e:
                    pass
            
            if len(importable_modules) == len(expected_modules):
                return {
                    'status': 'pass',
                    'message': f'All AI modules available: {", ".join(importable_modules)}',
                    'details': {'available_modules': importable_modules}
                }
            elif importable_modules:
                return {
                    'status': 'partial',
                    'message': f'Partial AI modules: {len(importable_modules)}/{len(expected_modules)}',
                    'details': {'available': importable_modules, 'missing': missing_modules}
                }
            else:
                return {
                    'status': 'fail',
                    'message': 'No AI modules available',
                    'recommendation': 'Ensure src/ directory contains AI module implementations'
                }
                
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'AI module validation failed: {e}'
            }
    
    def validate_paper_trading(self):
        """Validate paper trading capability"""
        try:
            # Check if we can access paper trading gateway
            from ib_insync import IB
            
            ib = IB()
            ib.connect('127.0.0.1', 4002, clientId=998, timeout=5)  # Paper trading port
            
            if ib.isConnected():
                # Test basic market data request
                from ib_insync import Stock
                contract = Stock('AAPL', 'SMART', 'USD')
                ib.qualifyContracts(contract)
                
                # Test paper account info
                account_info = ib.accountSummary()
                
                ib.disconnect()
                
                return {
                    'status': 'pass',
                    'message': f'Paper trading ready, account info retrieved ({len(account_info)} fields)',
                    'details': {'account_fields': len(account_info)}
                }
            else:
                return {
                    'status': 'fail',
                    'message': 'Cannot connect to paper trading gateway',
                    'recommendation': 'Ensure IBKR Gateway is running on port 4002 with paper account'
                }
                
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Paper trading validation failed: {e}',
                'recommendation': 'Check IBKR Gateway paper trading configuration'
            }
    
    def validate_data_pipeline(self):
        """Validate data collection and processing pipeline"""
        try:
            # Check for key data pipeline components
            key_files = [
                'level_ii_data_integration.py',
                'optimized_faiss_trading.py',
                'daily_launcher.py'
            ]
            
            available_files = []
            for file in key_files:
                file_path = self.base_path / file
                if file_path.exists():
                    available_files.append(file)
            
            if len(available_files) == len(key_files):
                return {
                    'status': 'pass',
                    'message': 'All data pipeline components available',
                    'details': {'components': available_files}
                }
            else:
                missing = set(key_files) - set(available_files)
                return {
                    'status': 'partial',
                    'message': f'Missing pipeline components: {", ".join(missing)}',
                    'recommendation': 'Ensure all data pipeline scripts are present'
                }
                
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Data pipeline validation failed: {e}'
            }
    
    def validate_risk_management(self):
        """Validate risk management systems"""
        # Basic risk management validation
        risk_components = {
            'position_sizing': False,
            'stop_loss': False, 
            'drawdown_limits': False,
            'exposure_limits': False
        }
        
        # Check for risk management in code (simplified)
        try:
            # Look for risk management keywords in key files
            risk_keywords = ['stop_loss', 'drawdown', 'position_size', 'risk']
            
            for file in ['daily_launcher.py', 'backtesting_framework.py']:
                file_path = self.base_path / file
                if file_path.exists():
                    content = file_path.read_text()
                    for keyword in risk_keywords:
                        if keyword in content.lower():
                            risk_components[keyword] = True
            
            implemented_count = sum(risk_components.values())
            
            if implemented_count >= 2:
                return {
                    'status': 'pass',
                    'message': f'Risk management components detected ({implemented_count}/4)',
                    'details': risk_components
                }
            else:
                return {
                    'status': 'partial',
                    'message': 'Limited risk management detected',
                    'recommendation': 'Implement comprehensive risk controls before live trading'
                }
                
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Risk management validation failed: {e}'
            }
    
    def validate_monitoring(self):
        """Validate monitoring and logging infrastructure"""
        try:
            # Check for monitoring components
            monitoring_files = [
                'DAILY_OPERATING_PROCEDURES.md',
                'efficiency_validation_results.py'
            ]
            
            available = []
            for file in monitoring_files:
                if (self.base_path / file).exists():
                    available.append(file)
            
            # Check if logging directories exist or can be created
            log_dirs = ['logs', 'daily_reports', 'performance_tracking']
            writable_dirs = []
            
            for dir_name in log_dirs:
                dir_path = self.base_path / dir_name
                try:
                    dir_path.mkdir(exist_ok=True)
                    writable_dirs.append(dir_name)
                except:
                    pass
            
            if available and writable_dirs:
                return {
                    'status': 'pass',
                    'message': f'Monitoring ready: {len(available)} files, {len(writable_dirs)} dirs',
                    'details': {'files': available, 'directories': writable_dirs}
                }
            else:
                return {
                    'status': 'partial',
                    'message': 'Basic monitoring available',
                    'recommendation': 'Set up comprehensive monitoring and logging'
                }
                
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Monitoring validation failed: {e}'
            }
    
    def assess_production_readiness(self):
        """Overall production readiness assessment"""
        try:
            # Count successful validations
            passed_validations = sum(1 for result in self.validation_results.values() 
                                   if result.get('status') == 'pass')
            
            total_validations = len(self.validation_results)
            readiness_percentage = (passed_validations / total_validations * 100) if total_validations > 0 else 0
            
            # Production readiness assessment
            if readiness_percentage >= 80:
                status = 'pass'
                message = f'Production ready ({readiness_percentage:.1f}% systems operational)'
                recommendation = 'Proceed with immediate deployment'
            elif readiness_percentage >= 60:
                status = 'partial'
                message = f'Near production ready ({readiness_percentage:.1f}% systems operational)'
                recommendation = 'Address remaining issues then deploy'
            else:
                status = 'fail'
                message = f'Not production ready ({readiness_percentage:.1f}% systems operational)'
                recommendation = 'Significant infrastructure work needed before deployment'
            
            return {
                'status': status,
                'message': message,
                'recommendation': recommendation,
                'details': {
                    'readiness_percentage': readiness_percentage,
                    'passed_validations': passed_validations,
                    'total_validations': total_validations
                }
            }
            
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Production readiness assessment failed: {e}'
            }
    
    def generate_readiness_report(self, readiness_score):
        """Generate comprehensive readiness report"""
        try:
            report = {
                'validation_timestamp': datetime.now().isoformat(),
                'overall_readiness_score': readiness_score,
                'validation_results': self.validation_results,
                'recommendations': self.get_priority_recommendations(),
                'deployment_plan': self.get_deployment_recommendations(readiness_score)
            }
            
            # Save report
            report_file = self.base_path / f'live_trading_readiness_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"\n📋 Readiness report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
    
    def get_priority_recommendations(self):
        """Get prioritized recommendations based on validation results"""
        recommendations = []
        
        for name, result in self.validation_results.items():
            if result.get('status') != 'pass' and result.get('recommendation'):
                recommendations.append({
                    'component': name,
                    'priority': 'high' if result.get('status') == 'fail' else 'medium',
                    'recommendation': result.get('recommendation')
                })
        
        return recommendations
    
    def get_deployment_recommendations(self, readiness_score):
        """Get deployment timeline recommendations based on readiness"""
        if readiness_score >= 80:
            return {
                'timeline': 'immediate',
                'approach': 'Deploy Tier 1 modules immediately',
                'focus': 'Level II + Advanced AI + FAISS integration'
            }
        elif readiness_score >= 60:
            return {
                'timeline': '1-2 weeks',
                'approach': 'Address infrastructure issues first',
                'focus': 'Fix identified problems then deploy selectively'
            }
        else:
            return {
                'timeline': '2-4 weeks',
                'approach': 'Comprehensive setup required',
                'focus': 'Build infrastructure before attempting deployment'
            }


def main():
    """Main validation execution"""
    print("🔍 LIVE TRADING SETUP VALIDATOR")
    print("="*50)
    print("Assessing current infrastructure readiness for strategic deployment")
    print()
    
    validator = LiveTradingSetupValidator()
    readiness_score = validator.run_comprehensive_validation()
    
    print("\n🎯 NEXT STEPS BASED ON READINESS:")
    if readiness_score >= 80:
        print("✅ Execute strategic deployment immediately")
        print("   Run: python strategic_deployment_executor.py")
    elif readiness_score >= 60:
        print("⚠️ Address identified issues then deploy")
        print("   Priority: Fix failed validations first")
    else:
        print("🔧 Infrastructure setup required before deployment")
        print("   Focus: Database, IBKR Gateway, AI modules")
    
    return 0 if readiness_score >= 60 else 1


if __name__ == "__main__":
    sys.exit(main())