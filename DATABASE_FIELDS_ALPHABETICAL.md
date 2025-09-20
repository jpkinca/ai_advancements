# Database Fields - Alphabetical Reference by Table

## Complete Alphabetical Field Listing with Purpose and Data Meaning

---

## Table 1: `ai_models` - AI Model Registry

Fields listed alphabetically:

### `configuration` (JSONB, NOT NULL)
**Purpose**: Stores all model hyperparameters and configuration settings as a JSON object
**Data Meaning**: Complete parameter set that defines how the model operates
**Example Data**: `{"learning_rate": 0.0003, "gamma": 0.99, "epsilon": 0.2, "state_size": 10}`
**Usage**: Retrieved during model initialization to restore exact training/inference settings

### `created_at` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records when the model was first registered in the system
**Data Meaning**: Timestamp of model creation for audit trail and lifecycle tracking
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Model age analysis, cleanup policies, and chronological model comparison

### `description` (TEXT, NULL)
**Purpose**: Provides detailed human-readable explanation of the model
**Data Meaning**: Comprehensive description of model purpose, methodology, and expected applications
**Example Data**: `"PPO reinforcement learning model optimized for AAPL trading with conservative risk management"`
**Usage**: Model documentation, team communication, and model selection guidance

### `is_active` (BOOLEAN, DEFAULT TRUE)
**Purpose**: Controls whether the model is currently enabled for trading operations
**Data Meaning**: Active status flag for production deployment control
**Example Data**: `true` (enabled) or `false` (disabled)
**Usage**: Production deployment control, A/B testing, and model retirement management

### `model_id` (UUID, PRIMARY KEY)
**Purpose**: Unique identifier for each AI model instance
**Data Meaning**: Globally unique identifier that serves as the primary key
**Example Data**: `f47ac10b-58cc-4372-a567-0e02b2c3d479`
**Usage**: Foreign key relationships, model references, and database joins

### `model_name` (VARCHAR(100), NOT NULL)
**Purpose**: Human-readable name for the model
**Data Meaning**: Descriptive identifier for easy recognition and management
**Example Data**: `"PPOTrader_AAPL_v2"`, `"GeneticOptimizer_Portfolio"`
**Usage**: User interfaces, logging, reporting, and model identification in business contexts

### `model_subtype` (VARCHAR(50), NULL)
**Purpose**: Specific algorithm or implementation variant within the model type
**Data Meaning**: Technical specification of the exact algorithm used
**Example Data**: `"ppo"`, `"fourier"`, `"wavelet"`, `"parameter_optimizer"`
**Usage**: Algorithm-specific processing, technical documentation, and implementation routing

### `model_type` (VARCHAR(50), NOT NULL)
**Purpose**: Primary category classification of the AI model
**Data Meaning**: High-level classification of the model's AI approach
**Example Data**: `"reinforcement_learning"`, `"genetic_optimization"`, `"sparse_spectrum"`
**Usage**: Model categorization, processing pipeline routing, and architecture organization

### `performance_metrics` (JSONB, NULL)
**Purpose**: Latest performance summary for quick reference without querying performance tables
**Data Meaning**: Most recent key performance indicators stored as JSON
**Example Data**: `{"sharpe_ratio": 1.85, "total_return": 0.247, "max_drawdown": 0.12}`
**Usage**: Dashboard displays, quick performance checks, and model comparison without complex queries

### `updated_at` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records the timestamp of the most recent configuration change
**Data Meaning**: Last modification time for change tracking and version control
**Example Data**: `2025-08-31 15:22:45.789012+00`
**Usage**: Change auditing, configuration versioning, and maintenance scheduling

### `version` (VARCHAR(20), NOT NULL, DEFAULT '1.0.0')
**Purpose**: Version control identifier for model iterations
**Data Meaning**: Semantic version number following standard versioning practices
**Example Data**: `"1.0.0"`, `"2.1.3"`, `"3.0.0-beta"`
**Usage**: Model deployment management, rollback procedures, and version comparison

---

## Table 2: `training_sessions` - Training History Tracking

Fields listed alphabetically:

### `created_at` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records when the training session record was created in the database
**Data Meaning**: Database insertion timestamp for record management
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Record lifecycle management and database auditing

### `end_time` (TIMESTAMPTZ, NULL)
**Purpose**: Records when the training process completed
**Data Meaning**: Training completion timestamp; NULL indicates ongoing training
**Example Data**: `2025-08-31 16:30:15.456789+00` or `NULL`
**Usage**: Training duration calculation, completion detection, and resource planning

### `error_message` (TEXT, NULL)
**Purpose**: Stores detailed error information if training failed
**Data Meaning**: Complete error description including stack traces and failure context
**Example Data**: `"CUDA out of memory error at epoch 150. GPU memory: 8GB exceeded"`
**Usage**: Debugging failed training runs, error pattern analysis, and troubleshooting

### `final_performance` (JSONB, NULL)
**Purpose**: Stores end-of-training validation metrics and results
**Data Meaning**: Final performance summary including accuracy, loss, and business metrics
**Example Data**: `{"final_accuracy": 0.847, "sharpe_ratio": 1.85, "total_reward": 295.72}`
**Usage**: Model evaluation, performance comparison, and deployment decision support

### `hyperparameters` (JSONB, NULL)
**Purpose**: Stores training-specific parameters used for this particular session
**Data Meaning**: Complete set of training configuration parameters as JSON
**Example Data**: `{"learning_rate": 0.0003, "batch_size": 32, "epochs": 100, "optimizer": "adam"}`
**Usage**: Training reproducibility, hyperparameter tuning analysis, and experiment tracking

### `model_artifacts_path` (TEXT, NULL)
**Purpose**: File system path to saved model weights, checkpoints, and related artifacts
**Data Meaning**: Storage location for trained model files and associated resources
**Example Data**: `"/models/ppo_trader/20250831_134522/checkpoint_final.pth"`
**Usage**: Model loading for inference, backup management, and artifact organization

### `model_id` (UUID, NOT NULL, FOREIGN KEY)
**Purpose**: Links the training session to the specific model being trained
**Data Meaning**: Reference to the parent model in the ai_models table
**Example Data**: `f47ac10b-58cc-4372-a567-0e02b2c3d479`
**Usage**: Model-specific training history, relationship queries, and data integrity

### `session_id` (UUID, PRIMARY KEY)
**Purpose**: Unique identifier for each training session
**Data Meaning**: Globally unique identifier for the training run
**Example Data**: `b12ef89a-1234-5678-9abc-def012345678`
**Usage**: Session tracking, foreign key relationships, and training run identification

### `session_name` (VARCHAR(100), NULL)
**Purpose**: Human-readable descriptive name for the training session
**Data Meaning**: Meaningful identifier for training runs to aid in organization
**Example Data**: `"AAPL_training_2025Q3"`, `"portfolio_optimization_run_5"`
**Usage**: Training run identification, experiment organization, and reporting

### `start_time` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records when the training process began
**Data Meaning**: Training initiation timestamp for duration tracking
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Training duration calculation, resource scheduling, and progress monitoring

### `status` (VARCHAR(20), DEFAULT 'running')
**Purpose**: Current state of the training process
**Data Meaning**: Training lifecycle status indicator
**Example Data**: `"running"`, `"completed"`, `"failed"`, `"stopped"`
**Usage**: Training monitoring, resource management, and automated cleanup procedures

### `training_data_period` (DATERANGE, NULL)
**Purpose**: Defines the time range of market data used for training
**Data Meaning**: Start and end dates of the training dataset
**Example Data**: `[2024-01-01, 2024-12-31)`
**Usage**: Data lineage tracking, training reproducibility, and temporal validation

### `training_metrics` (JSONB, NULL)
**Purpose**: Stores progress metrics throughout the training process
**Data Meaning**: Time series of training progress including loss curves and convergence data
**Example Data**: `{"loss_curve": [0.5, 0.3, 0.2, 0.15], "rewards": [10, 15, 22, 28]}`
**Usage**: Training progress monitoring, convergence analysis, and performance visualization

### `validation_data_period` (DATERANGE, NULL)
**Purpose**: Defines the time range of market data used for validation
**Data Meaning**: Start and end dates of the validation dataset
**Example Data**: `[2025-01-01, 2025-03-31)`
**Usage**: Model validation, overfitting detection, and performance assessment

---

## Table 3: `ai_signals` - Trading Signal Generation

Fields listed alphabetically:

### `confidence` (DECIMAL(5,4), NOT NULL, CHECK 0-1)
**Purpose**: Model's confidence level in the generated signal
**Data Meaning**: Probability score representing the model's certainty about the signal
**Example Data**: `0.8750` (87.5% confidence), `0.6234` (62.34% confidence)
**Usage**: Risk-proportional position sizing, signal filtering, and trade execution decisions

### `created_at` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records when the signal was stored in the database
**Data Meaning**: Database insertion timestamp for record management
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Signal age tracking, database maintenance, and audit trails

### `market_data_timestamp` (TIMESTAMPTZ, NULL)
**Purpose**: Timestamp of the underlying market data used to generate the signal
**Data Meaning**: Time point of the market conditions that influenced the signal
**Example Data**: `2025-08-31 13:44:00.000000+00`
**Usage**: Data freshness validation, signal timing analysis, and market synchronization

### `metadata` (JSONB, NULL)
**Purpose**: Additional model-specific information about the signal generation
**Data Meaning**: Extended details about the reasoning, features, and context
**Example Data**: `{"rsi": 75.2, "sma_cross": true, "volume_spike": false, "pattern": "bullish_flag"}`
**Usage**: Signal analysis, feature importance tracking, and model debugging

### `model_id` (UUID, NOT NULL, FOREIGN KEY)
**Purpose**: Identifies which AI model generated this signal
**Data Meaning**: Reference to the source model in the ai_models table
**Example Data**: `f47ac10b-58cc-4372-a567-0e02b2c3d479`
**Usage**: Model performance attribution, signal source tracking, and model comparison

### `price_target` (DECIMAL(15,6), NOT NULL)
**Purpose**: The expected price level where the signal should be profitable
**Data Meaning**: Target price for the trading action expressed in currency units
**Example Data**: `152.750000` (target price of $152.75)
**Usage**: Profit target setting, signal validation, and performance measurement

### `signal_id` (UUID, PRIMARY KEY)
**Purpose**: Unique identifier for each trading signal
**Data Meaning**: Globally unique identifier for the signal record
**Example Data**: `c23df45e-6789-1abc-def0-123456789abc`
**Usage**: Signal tracking, performance linkage, and database relationships

### `signal_timestamp` (TIMESTAMPTZ, NOT NULL)
**Purpose**: When the AI model generated the trading signal
**Data Meaning**: Signal generation time for timing analysis and execution
**Example Data**: `2025-08-31 13:45:00.000000+00`
**Usage**: Signal timing analysis, execution timing, and market condition correlation

### `signal_type` (VARCHAR(10), NOT NULL)
**Purpose**: The recommended trading action
**Data Meaning**: Specific trading instruction generated by the model
**Example Data**: `"BUY"`, `"SELL"`, `"HOLD"`
**Usage**: Trading execution, signal categorization, and action distribution analysis

### `stop_loss` (DECIMAL(15,6), NULL)
**Purpose**: Recommended stop loss level for risk management
**Data Meaning**: Price level where losses should be cut to limit downside risk
**Example Data**: `145.500000` (stop loss at $145.50)
**Usage**: Risk management, automated stop orders, and downside protection

### `symbol` (VARCHAR(20), NOT NULL)
**Purpose**: The financial instrument for which the signal was generated
**Data Meaning**: Trading symbol or ticker identifier for the asset
**Example Data**: `"AAPL"`, `"GOOGL"`, `"SPY"`, `"EURUSD"`
**Usage**: Asset-specific analysis, portfolio construction, and instrument filtering

### `take_profit` (DECIMAL(15,6), NULL)
**Purpose**: Recommended profit-taking level for the signal
**Data Meaning**: Price level where profits should be realized
**Example Data**: `158.250000` (take profit at $158.25)
**Usage**: Profit realization, automated limit orders, and upside capture

---

## Table 4: `signal_performance` - Signal Quality Tracking

Fields listed alphabetically:

### `actual_price` (DECIMAL(15,6), NULL)
**Purpose**: The real market price at the time of performance evaluation
**Data Meaning**: Actual market price used to calculate signal performance
**Example Data**: `154.320000` (actual price of $154.32)
**Usage**: Performance calculation, signal accuracy assessment, and reality checking

### `created_at` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records when the performance evaluation was stored
**Data Meaning**: Database insertion timestamp for the performance record
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Performance tracking timeline and record management

### `evaluation_timestamp` (TIMESTAMPTZ, NOT NULL)
**Purpose**: When the signal performance was measured
**Data Meaning**: Time point at which the signal's success was evaluated
**Example Data**: `2025-08-31 16:00:00.000000+00`
**Usage**: Performance timing analysis and evaluation scheduling

### `holding_period_hours` (INTEGER, NULL)
**Purpose**: Duration for which the signal remained relevant or was held
**Data Meaning**: Time span in hours from signal generation to resolution
**Example Data**: `24` (24-hour holding period), `168` (1-week holding period)
**Usage**: Signal timing optimization, holding period analysis, and strategy tuning

### `performance_id` (UUID, PRIMARY KEY)
**Purpose**: Unique identifier for each performance evaluation record
**Data Meaning**: Globally unique identifier for the performance measurement
**Example Data**: `d34ef56f-789a-1bcd-ef01-23456789abcd`
**Usage**: Performance record tracking and database relationships

### `pnl_if_executed` (DECIMAL(15,6), NULL)
**Purpose**: Theoretical profit or loss if the signal had been traded
**Data Meaning**: Calculated financial outcome assuming the signal was executed
**Example Data**: `2.750000` (profit of $2.75), `-1.250000` (loss of $1.25)
**Usage**: Signal profitability analysis, model ROI calculation, and strategy validation

### `price_change_pct` (DECIMAL(8,4), NULL)
**Purpose**: Percentage change in price from signal generation to evaluation
**Data Meaning**: Relative price movement expressed as a percentage
**Example Data**: `1.8500` (+1.85%), `-0.7500` (-0.75%)
**Usage**: Signal accuracy measurement, percentage-based performance analysis

### `signal_id` (UUID, NOT NULL, FOREIGN KEY)
**Purpose**: Links the performance record to the original signal
**Data Meaning**: Reference to the signal being evaluated in the ai_signals table
**Example Data**: `c23df45e-6789-1abc-def0-123456789abc`
**Usage**: Signal-performance relationship tracking and analysis

### `stop_loss_hit` (BOOLEAN, NULL)
**Purpose**: Indicates whether the stop loss level was triggered
**Data Meaning**: Binary flag showing if the risk management level was reached
**Example Data**: `true` (stop loss triggered), `false` (stop loss not hit)
**Usage**: Risk management effectiveness analysis and stop loss optimization

### `target_hit` (BOOLEAN, NULL)
**Purpose**: Indicates whether the price target was achieved
**Data Meaning**: Binary flag showing if the profit target was reached
**Example Data**: `true` (target achieved), `false` (target not reached)
**Usage**: Signal success rate calculation and target setting optimization

---

## Table 5: `feature_data` - Machine Learning Features

Fields listed alphabetically:

### `created_at` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records when the feature data was stored in the database
**Data Meaning**: Database insertion timestamp for feature record management
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Feature freshness tracking and data lifecycle management

### `extraction_method` (VARCHAR(50), NULL)
**Purpose**: Identifies the algorithm or approach used to extract the features
**Data Meaning**: Technical specification of the feature calculation method
**Example Data**: `"ta_lib_indicators"`, `"custom_fourier"`, `"sklearn_scaler"`
**Usage**: Feature reproducibility, method comparison, and processing pipeline tracking

### `feature_id` (UUID, PRIMARY KEY)
**Purpose**: Unique identifier for each feature extraction record
**Data Meaning**: Globally unique identifier for the feature set
**Example Data**: `e45fg67g-890b-2cde-f012-3456789abcde`
**Usage**: Feature record tracking and database relationships

### `feature_set_name` (VARCHAR(50), NOT NULL)
**Purpose**: Categorizes the type of features extracted
**Data Meaning**: Logical grouping of related features for organization
**Example Data**: `"technical_indicators"`, `"fourier_coefficients"`, `"market_microstructure"`
**Usage**: Feature categorization, model input selection, and processing pipeline routing

### `features` (JSONB, NOT NULL)
**Purpose**: Contains the actual feature values as a JSON object
**Data Meaning**: Complete feature vector with named features and their calculated values
**Example Data**: `{"sma_20": 150.25, "rsi_14": 65.8, "bollinger_upper": 155.0}`
**Usage**: Model training input, inference data, and feature analysis

### `raw_data_hash` (VARCHAR(64), NULL)
**Purpose**: Hash of the input market data used for feature extraction
**Data Meaning**: Cryptographic fingerprint for data versioning and integrity
**Example Data**: `"a1b2c3d4e5f6789012345678901234567890abcdef"`
**Usage**: Data lineage tracking, cache invalidation, and feature reproducibility

### `symbol` (VARCHAR(20), NOT NULL)
**Purpose**: The financial instrument for which features were extracted
**Data Meaning**: Trading symbol or ticker identifier for the asset
**Example Data**: `"AAPL"`, `"GOOGL"`, `"SPY"`
**Usage**: Asset-specific feature analysis and model input organization

### `timestamp` (TIMESTAMPTZ, NOT NULL)
**Purpose**: The time point for which the features were calculated
**Data Meaning**: Temporal reference for the feature values
**Example Data**: `2025-08-31 13:45:00.000000+00`
**Usage**: Time series analysis, feature alignment, and temporal model inputs

---

## Table 6: `optimization_runs` - Genetic Algorithm Results

Fields listed alphabetically:

### `best_fitness` (DECIMAL(15,6), NULL)
**Purpose**: The highest fitness score achieved during the optimization
**Data Meaning**: Best objective function value found by the genetic algorithm
**Example Data**: `2.507500` (Sharpe ratio of 2.51), `0.247000` (24.7% return)
**Usage**: Optimization success measurement and solution quality assessment

### `best_individual` (JSONB, NULL)
**Purpose**: The parameter combination that achieved the best fitness
**Data Meaning**: Complete parameter set of the optimal solution
**Example Data**: `{"sma_short": 12, "sma_long": 31, "rsi_period": 20}`
**Usage**: Optimal parameter deployment and solution implementation

### `convergence_data` (JSONB, NULL)
**Purpose**: Evolution of fitness scores across generations
**Data Meaning**: Time series of optimization progress and convergence patterns
**Example Data**: `{"generation_best": [1.2, 1.8, 2.1, 2.5], "generation_avg": [0.8, 1.2, 1.6, 1.9]}`
**Usage**: Convergence analysis, optimization monitoring, and algorithm tuning

### `created_at` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records when the optimization run was registered
**Data Meaning**: Database insertion timestamp for the optimization record
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Optimization history tracking and record management

### `end_time` (TIMESTAMPTZ, NULL)
**Purpose**: When the genetic algorithm optimization completed
**Data Meaning**: Completion timestamp; NULL indicates ongoing optimization
**Example Data**: `2025-08-31 16:30:15.456789+00` or `NULL`
**Usage**: Optimization duration calculation and completion detection

### `final_population` (JSONB, NULL)
**Purpose**: Complete state of the final generation for analysis
**Data Meaning**: All individuals in the last generation with their parameters and fitness
**Example Data**: `{"individuals": [{"genes": {...}, "fitness": 2.5}, {"genes": {...}, "fitness": 2.3}]}`
**Usage**: Population diversity analysis and genetic algorithm research

### `generations` (INTEGER, NULL)
**Purpose**: Number of evolutionary generations executed
**Data Meaning**: Count of complete generation cycles in the optimization
**Example Data**: `50`, `100`, `200`
**Usage**: Optimization effort measurement and algorithm parameter tuning

### `model_id` (UUID, NULL, FOREIGN KEY)
**Purpose**: Links optimization to a specific AI model if applicable
**Data Meaning**: Reference to associated model in the ai_models table
**Example Data**: `f47ac10b-58cc-4372-a567-0e02b2c3d479` or `NULL`
**Usage**: Model-specific optimization tracking and integration

### `objective_function` (VARCHAR(50), NULL)
**Purpose**: The metric being optimized by the genetic algorithm
**Data Meaning**: Specific fitness function used to evaluate solutions
**Example Data**: `"sharpe_ratio"`, `"total_return"`, `"max_drawdown"`, `"multi_objective"`
**Usage**: Optimization goal specification and results interpretation

### `optimization_type` (VARCHAR(50), NOT NULL)
**Purpose**: Category of optimization being performed
**Data Meaning**: High-level classification of what is being optimized
**Example Data**: `"parameter_optimization"`, `"portfolio_optimization"`, `"strategy_optimization"`
**Usage**: Optimization categorization and processing pipeline routing

### `parameter_ranges` (JSONB, NULL)
**Purpose**: Defines the search space bounds for each parameter
**Data Meaning**: Minimum and maximum values for each parameter being optimized
**Example Data**: `{"sma_short": [5, 20], "sma_long": [20, 50], "rsi_period": [10, 30]}`
**Usage**: Search space definition, constraint enforcement, and optimization bounds

### `population_size` (INTEGER, NULL)
**Purpose**: Number of individuals in each generation of the genetic algorithm
**Data Meaning**: Population size parameter that affects optimization diversity
**Example Data**: `50`, `100`, `200`
**Usage**: Algorithm configuration, performance analysis, and resource planning

### `run_id` (UUID, PRIMARY KEY)
**Purpose**: Unique identifier for each optimization run
**Data Meaning**: Globally unique identifier for the optimization session
**Example Data**: `f56gh78h-901c-3def-0123-456789abcdef`
**Usage**: Optimization tracking, individual linkage, and database relationships

### `start_time` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: When the genetic algorithm optimization began
**Data Meaning**: Optimization initiation timestamp
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Optimization duration calculation and progress monitoring

### `status` (VARCHAR(20), DEFAULT 'running')
**Purpose**: Current state of the optimization process
**Data Meaning**: Optimization lifecycle status indicator
**Example Data**: `"running"`, `"completed"`, `"failed"`, `"stopped"`
**Usage**: Optimization monitoring and automated management

---

## Table 7: `optimization_individuals` - Individual GA Solutions

Fields listed alphabetically:

### `created_at` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records when the individual was evaluated and stored
**Data Meaning**: Database insertion timestamp for the individual record
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Individual tracking timeline and record management

### `fitness_score` (DECIMAL(15,6), NULL)
**Purpose**: Performance score for this specific parameter combination
**Data Meaning**: Objective function value achieved by this individual
**Example Data**: `2.345000` (fitness score of 2.345)
**Usage**: Individual performance evaluation and selection for breeding

### `generation` (INTEGER, NOT NULL)
**Purpose**: Which evolutionary generation this individual belongs to
**Data Meaning**: Generation number in the evolutionary sequence
**Example Data**: `1`, `25`, `50`
**Usage**: Evolutionary progress tracking and generation-based analysis

### `genes` (JSONB, NOT NULL)
**Purpose**: The parameter values that define this individual solution
**Data Meaning**: Complete parameter set representing the individual's chromosome
**Example Data**: `{"sma_short": 12, "sma_long": 31, "rsi_period": 20, "stop_loss": 0.02}`
**Usage**: Solution representation, breeding operations, and parameter analysis

### `individual_id` (UUID, PRIMARY KEY)
**Purpose**: Unique identifier for each individual in the genetic algorithm
**Data Meaning**: Globally unique identifier for the individual solution
**Example Data**: `g67hi89i-012d-4ef0-1234-56789abcdefg`
**Usage**: Individual tracking and database relationships

### `individual_rank` (INTEGER, NULL)
**Purpose**: Ranking of this individual within its generation based on fitness
**Data Meaning**: Relative performance rank (1 = best in generation)
**Example Data**: `1` (best), `5` (5th best), `25` (25th best)
**Usage**: Selection pressure, elite preservation, and performance analysis

### `objectives` (JSONB, NULL)
**Purpose**: Breakdown of fitness components for multi-objective optimization
**Data Meaning**: Individual objective scores when multiple goals are optimized
**Example Data**: `{"return": 0.25, "sharpe": 1.8, "drawdown": 0.12, "volatility": 0.18}`
**Usage**: Multi-objective analysis, trade-off evaluation, and Pareto frontier analysis

### `run_id` (UUID, NOT NULL, FOREIGN KEY)
**Purpose**: Links the individual to its specific optimization run
**Data Meaning**: Reference to the parent optimization run
**Example Data**: `f56gh78h-901c-3def-0123-456789abcdef`
**Usage**: Run-specific analysis and individual grouping

---

## Table 8: `anomaly_detections` - Market Anomaly Detection

Fields listed alphabetically:

### `anomaly_details` (JSONB, NULL)
**Purpose**: Model-specific information about the detected anomaly
**Data Meaning**: Extended details about the anomaly characteristics and context
**Example Data**: `{"price_jump": 5.2, "volume_multiple": 3.4, "time_window": "5min"}`
**Usage**: Anomaly analysis, pattern recognition, and detailed investigation

### `anomaly_id` (UUID, PRIMARY KEY)
**Purpose**: Unique identifier for each detected anomaly
**Data Meaning**: Globally unique identifier for the anomaly event
**Example Data**: `h78ij90j-123e-5f01-2345-6789abcdefgh`
**Usage**: Anomaly tracking, event correlation, and database relationships

### `anomaly_type` (VARCHAR(50), NULL)
**Purpose**: Classification of the type of anomaly detected
**Data Meaning**: Category describing the nature of the unusual market behavior
**Example Data**: `"price_jump"`, `"volume_spike"`, `"pattern_break"`, `"correlation_shift"`
**Usage**: Anomaly categorization, response procedures, and pattern analysis

### `confidence` (DECIMAL(5,4), NULL)
**Purpose**: Model's confidence in the anomaly detection
**Data Meaning**: Probability score for the anomaly detection accuracy
**Example Data**: `0.9250` (92.5% confidence), `0.7800` (78% confidence)
**Usage**: Anomaly filtering, alert prioritization, and detection reliability assessment

### `created_at` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records when the anomaly detection was stored
**Data Meaning**: Database insertion timestamp for the anomaly record
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Anomaly timeline tracking and record management

### `detection_timestamp` (TIMESTAMPTZ, NOT NULL)
**Purpose**: When the anomaly was first identified by the AI model
**Data Meaning**: Time point of anomaly occurrence in the market
**Example Data**: `2025-08-31 13:42:15.789012+00`
**Usage**: Anomaly timing analysis, market correlation, and response timing

### `market_data_context` (JSONB, NULL)
**Purpose**: Relevant market conditions at the time of anomaly detection
**Data Meaning**: Contextual market data that influenced or surrounded the anomaly
**Example Data**: `{"bid": 150.25, "ask": 150.27, "volume": 125000, "volatility": 0.25}`
**Usage**: Anomaly context analysis, market condition correlation, and investigation support

### `model_id` (UUID, NOT NULL, FOREIGN KEY)
**Purpose**: Identifies which AI model detected the anomaly
**Data Meaning**: Reference to the detector model in the ai_models table
**Example Data**: `f47ac10b-58cc-4372-a567-0e02b2c3d479`
**Usage**: Model performance attribution and detector-specific analysis

### `resolved` (BOOLEAN, DEFAULT FALSE)
**Purpose**: Indicates whether the anomaly situation has been addressed
**Data Meaning**: Resolution status flag for anomaly lifecycle management
**Example Data**: `true` (resolved), `false` (still active)
**Usage**: Active anomaly tracking, resolution procedures, and status monitoring

### `resolution_timestamp` (TIMESTAMPTZ, NULL)
**Purpose**: When the anomaly was marked as resolved or returned to normal
**Data Meaning**: Time point when the anomalous condition ended
**Example Data**: `2025-08-31 14:15:30.123456+00` or `NULL`
**Usage**: Anomaly duration calculation and resolution analysis

### `severity_score` (DECIMAL(8,4), NOT NULL)
**Purpose**: Quantifies how extreme or unusual the anomaly is
**Data Meaning**: Statistical measure of anomaly magnitude (often in standard deviations)
**Example Data**: `3.2500` (3.25 standard deviations), `5.7800` (5.78 standard deviations)
**Usage**: Anomaly prioritization, alert thresholds, and severity-based response

### `symbol` (VARCHAR(20), NOT NULL)
**Purpose**: The financial instrument where the anomaly was detected
**Data Meaning**: Trading symbol or ticker identifier for the affected asset
**Example Data**: `"AAPL"`, `"GOOGL"`, `"SPY"`
**Usage**: Asset-specific anomaly analysis and portfolio risk assessment

---

## Table 9: `model_performance` - Performance Metrics Over Time

Fields listed alphabetically:

### `avg_confidence` (DECIMAL(5,4), NULL)
**Purpose**: Average confidence score across all signals in the evaluation period
**Data Meaning**: Mean confidence level showing model certainty consistency
**Example Data**: `0.7850` (78.5% average confidence)
**Usage**: Model reliability assessment and confidence calibration analysis

### `avg_holding_period_hours` (DECIMAL(8,2), NULL)
**Purpose**: Average duration signals were held or remained relevant
**Data Meaning**: Mean time between signal generation and resolution in hours
**Example Data**: `24.50` (24.5 hours), `168.25` (about 1 week)
**Usage**: Strategy timing optimization and holding period analysis

### `created_at` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records when the performance metrics were calculated and stored
**Data Meaning**: Database insertion timestamp for the performance record
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Performance tracking timeline and calculation scheduling

### `evaluation_period` (DATERANGE, NOT NULL)
**Purpose**: Time range for which the performance metrics were calculated
**Data Meaning**: Start and end dates of the performance evaluation window
**Example Data**: `[2025-08-01, 2025-08-31]`
**Usage**: Performance period identification and temporal analysis

### `max_drawdown` (DECIMAL(8,4), NULL)
**Purpose**: Largest peak-to-trough decline during the evaluation period
**Data Meaning**: Maximum portfolio decline expressed as a decimal percentage
**Example Data**: `0.1250` (12.5% maximum drawdown)
**Usage**: Risk assessment, drawdown control, and risk management evaluation

### `model_id` (UUID, NOT NULL, FOREIGN KEY)
**Purpose**: Identifies which model's performance is being tracked
**Data Meaning**: Reference to the model in the ai_models table
**Example Data**: `f47ac10b-58cc-4372-a567-0e02b2c3d479`
**Usage**: Model-specific performance analysis and comparison

### `performance_id` (UUID, PRIMARY KEY)
**Purpose**: Unique identifier for each performance evaluation record
**Data Meaning**: Globally unique identifier for the performance measurement
**Example Data**: `i89jk01k-234f-6012-3456-789abcdefghi`
**Usage**: Performance record tracking and database relationships

### `risk_adjusted_return` (DECIMAL(8,4), NULL)
**Purpose**: Return per unit of risk taken during the evaluation period
**Data Meaning**: Risk-adjusted performance metric (often Sharpe ratio or similar)
**Example Data**: `1.8500` (1.85 risk-adjusted return)
**Usage**: Risk-adjusted performance comparison and investment efficiency assessment

### `sharpe_ratio` (DECIMAL(8,4), NULL)
**Purpose**: Risk-adjusted return metric for the evaluation period
**Data Meaning**: Excess return per unit of volatility (return - risk-free rate) / volatility
**Example Data**: `1.8500` (Sharpe ratio of 1.85)
**Usage**: Risk-adjusted performance evaluation and model comparison

### `signal_accuracy` (DECIMAL(5,4), NULL)
**Purpose**: Percentage of signals that correctly predicted market direction
**Data Meaning**: Accuracy rate for directional predictions
**Example Data**: `0.6750` (67.5% accuracy rate)
**Usage**: Signal quality assessment and model accuracy evaluation

### `successful_signals` (INTEGER, NULL)
**Purpose**: Count of signals that achieved their targets during the period
**Data Meaning**: Number of signals that reached profit targets or objectives
**Example Data**: `45` (45 successful signals)
**Usage**: Success rate calculation and target achievement analysis

### `total_return` (DECIMAL(8,4), NULL)
**Purpose**: Cumulative return achieved during the evaluation period
**Data Meaning**: Total portfolio return expressed as a decimal percentage
**Example Data**: `0.2470` (24.7% total return)
**Usage**: Absolute performance measurement and return analysis

### `total_signals` (INTEGER, NULL)
**Purpose**: Total number of signals generated during the evaluation period
**Data Meaning**: Count of all signals produced by the model
**Example Data**: `67` (67 total signals)
**Usage**: Signal frequency analysis and model activity measurement

### `volatility` (DECIMAL(8,4), NULL)
**Purpose**: Standard deviation of returns during the evaluation period
**Data Meaning**: Measure of return variability and risk
**Example Data**: `0.1850` (18.5% volatility)
**Usage**: Risk measurement, volatility targeting, and risk management

### `win_rate` (DECIMAL(5,4), NULL)
**Purpose**: Percentage of signals that resulted in profitable trades
**Data Meaning**: Profitability rate for executed signals
**Example Data**: `0.7200` (72% win rate)
**Usage**: Profitability assessment and strategy evaluation

---

## Table 10: `rl_episodes` - Reinforcement Learning Training Data

Fields listed alphabetically:

### `actions_taken` (JSONB, NULL)
**Purpose**: Complete sequence of actions taken during the episode
**Data Meaning**: Ordered list of all decisions made by the RL agent
**Example Data**: `[0, 1, 0, 2, 1, 0]` (where 0=HOLD, 1=BUY, 2=SELL)
**Usage**: Action pattern analysis, strategy understanding, and behavior debugging

### `created_at` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records when the episode data was stored in the database
**Data Meaning**: Database insertion timestamp for the episode record
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Episode timeline tracking and record management

### `episode_end_reason` (VARCHAR(50), NULL)
**Purpose**: Explanation for why the training episode terminated
**Data Meaning**: Reason for episode completion or early termination
**Example Data**: `"completed"`, `"early_stop"`, `"max_steps"`, `"failure"`
**Usage**: Episode analysis, training debugging, and termination condition evaluation

### `episode_id` (UUID, PRIMARY KEY)
**Purpose**: Unique identifier for each reinforcement learning episode
**Data Meaning**: Globally unique identifier for the training episode
**Example Data**: `j90kl12l-345g-7123-4567-89abcdefghij`
**Usage**: Episode tracking and database relationships

### `episode_length` (INTEGER, NULL)
**Purpose**: Number of actions or steps taken during the episode
**Data Meaning**: Count of decision points or time steps in the episode
**Example Data**: `250` (250 steps), `500` (500 steps)
**Usage**: Episode duration analysis and training efficiency measurement

### `episode_number` (INTEGER, NOT NULL)
**Purpose**: Sequential episode number within the training session
**Data Meaning**: Chronological order of episodes in the training sequence
**Example Data**: `1`, `150`, `1000`
**Usage**: Training progress tracking and episode sequencing

### `final_portfolio_value` (DECIMAL(15,6), NULL)
**Purpose**: Portfolio value at the end of the episode
**Data Meaning**: Final wealth or asset value achieved by the RL agent
**Example Data**: `10567.250000` (final portfolio value of $10,567.25)
**Usage**: Episode performance evaluation and wealth accumulation tracking

### `max_drawdown` (DECIMAL(8,4), NULL)
**Purpose**: Maximum decline in portfolio value during the episode
**Data Meaning**: Largest peak-to-trough decline within the episode
**Example Data**: `0.0850` (8.5% maximum drawdown)
**Usage**: Risk assessment and drawdown control evaluation

### `rewards` (JSONB, NULL)
**Purpose**: Sequence of rewards received at each step of the episode
**Data Meaning**: Ordered list of reward values corresponding to each action
**Example Data**: `[0.1, 0.5, -0.2, 0.8, 0.3, 0.0]`
**Usage**: Reward analysis, learning signal evaluation, and training debugging

### `session_id` (UUID, NOT NULL, FOREIGN KEY)
**Purpose**: Links the episode to its specific training session
**Data Meaning**: Reference to the parent training session
**Example Data**: `b12ef89a-1234-5678-9abc-def012345678`
**Usage**: Session-specific episode analysis and training run grouping

### `total_reward` (DECIMAL(15,6), NULL)
**Purpose**: Cumulative reward accumulated during the entire episode
**Data Meaning**: Sum of all rewards received throughout the episode
**Example Data**: `295.720000` (total reward of 295.72)
**Usage**: Episode performance evaluation and learning progress measurement

---

## Table 11: `spectrum_analysis` - Frequency Domain Analysis

Fields listed alphabetically:

### `analysis_id` (UUID, PRIMARY KEY)
**Purpose**: Unique identifier for each spectral analysis result
**Data Meaning**: Globally unique identifier for the analysis record
**Example Data**: `k01lm23m-456h-8234-5678-9abcdefghijk`
**Usage**: Analysis tracking and database relationships

### `analysis_timestamp` (TIMESTAMPTZ, NOT NULL)
**Purpose**: When the spectral analysis was performed
**Data Meaning**: Time point of the frequency domain analysis execution
**Example Data**: `2025-08-31 13:45:00.000000+00`
**Usage**: Analysis timing and temporal correlation with market events

### `analysis_type` (VARCHAR(50), NOT NULL)
**Purpose**: The method used for spectral analysis
**Data Meaning**: Specific algorithm or technique applied
**Example Data**: `"fourier"`, `"wavelet"`, `"compressed_sensing"`, `"hilbert_transform"`
**Usage**: Analysis method selection and technique comparison

### `created_at` (TIMESTAMPTZ, DEFAULT NOW())
**Purpose**: Records when the analysis results were stored
**Data Meaning**: Database insertion timestamp for the analysis record
**Example Data**: `2025-08-31 13:45:22.123456+00`
**Usage**: Analysis timeline tracking and record management

### `frequency_components` (JSONB, NULL)
**Purpose**: Dominant frequencies, amplitudes, and phases discovered in the analysis
**Data Meaning**: Complete spectral decomposition with frequency domain characteristics
**Example Data**: `{"frequencies": [0.05, 0.12], "amplitudes": [15.2, 8.7], "phases": [0.2, 1.1]}`
**Usage**: Frequency analysis, cycle detection, and spectral feature extraction

### `model_id` (UUID, NULL, FOREIGN KEY)
**Purpose**: Identifies which model performed the spectral analysis
**Data Meaning**: Reference to the analyzer model in the ai_models table
**Example Data**: `f47ac10b-58cc-4372-a567-0e02b2c3d479` or `NULL`
**Usage**: Model-specific analysis tracking and analyzer attribution

### `pattern_confidence` (DECIMAL(5,4), NULL)
**Purpose**: Confidence level in the detected pattern identification
**Data Meaning**: Probability score for pattern recognition accuracy
**Example Data**: `0.8750` (87.5% confidence in pattern detection)
**Usage**: Pattern filtering, detection reliability, and analysis quality assessment

### `pattern_detected` (VARCHAR(100), NULL)
**Purpose**: Named pattern if recognized by the analysis
**Data Meaning**: Specific chart pattern or market structure identified
**Example Data**: `"head_and_shoulders"`, `"cup_and_handle"`, `"double_bottom"`
**Usage**: Pattern-based trading, technical analysis, and market structure recognition

### `reconstruction_error` (DECIMAL(15,10), NULL)
**Purpose**: Error metric for signal reconstruction quality
**Data Meaning**: Measure of how well the spectral model reconstructs the original signal
**Example Data**: `0.0012345678` (low reconstruction error indicating good fit)
**Usage**: Analysis quality assessment and model validation

### `spectral_features` (JSONB, NULL)
**Purpose**: Features derived from the spectral analysis for machine learning
**Data Meaning**: Processed spectral characteristics suitable for ML model input
**Example Data**: `{"dominant_freq": 0.12, "spectral_entropy": 2.45, "bandwidth": 0.08}`
**Usage**: Feature engineering, ML model input, and spectral characteristic analysis

### `symbol` (VARCHAR(20), NOT NULL)
**Purpose**: The financial instrument that was analyzed
**Data Meaning**: Trading symbol or ticker identifier for the analyzed asset
**Example Data**: `"AAPL"`, `"GOOGL"`, `"SPY"`
**Usage**: Asset-specific spectral analysis and instrument-based pattern recognition

---

## Summary Statistics

**Total Fields Documented**: 123 individual fields across 11 tables
**Data Types Used**: UUID (Primary Keys), TIMESTAMPTZ (Timestamps), DECIMAL (Financial), JSONB (Flexible), VARCHAR (Text), INTEGER (Counts), BOOLEAN (Flags), DATERANGE (Periods)
**Relationship Fields**: 8 foreign key relationships connecting the tables
**Performance Indexes**: 15+ optimized indexes for query performance
**Business Functions**: Model management, signal generation, performance tracking, optimization, anomaly detection, feature engineering

This alphabetical reference provides complete field-level documentation for understanding and implementing the AI trading database schema.
