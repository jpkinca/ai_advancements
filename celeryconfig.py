#!/usr/bin/env python3
"""
Celery Configuration for Candlestick Analysis Tasks

This file contains the configuration for Celery workers handling
candlestick analysis tasks.
"""

import os
from datetime import timedelta

# Broker settings (Redis)
broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')

# Result backend
result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Task serialization
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'UTC'
enable_utc = True

# Worker settings
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 1000
worker_disable_rate_limits = False

# Task settings
task_acks_late = True
task_reject_on_worker_lost = True
task_default_retry_delay = 60
task_max_retries = 3

# Queue settings
task_default_queue = 'candlestick'
task_queues = {
    'candlestick': {
        'exchange': 'candlestick',
        'exchange_type': 'direct',
        'routing_key': 'candlestick',
    },
}

# Route tasks to specific queues
task_routes = {
    'candlestick.analyze_symbol': {'queue': 'candlestick'},
    'candlestick.generate_signal': {'queue': 'candlestick'},
    'candlestick.batch_analyze': {'queue': 'candlestick'},
    'candlestick.real_time_monitor': {'queue': 'candlestick'},
    'candlestick.update_models': {'queue': 'candlestick'},
}

# Beat settings (for periodic tasks)
beat_schedule = {
    'update-candlestick-models-daily': {
        'task': 'candlestick.update_models',
        'schedule': timedelta(days=1),  # Every day
    },
    'monitor-major-symbols-5min': {
        'task': 'candlestick.real_time_monitor',
        'schedule': timedelta(minutes=5),  # Every 5 minutes
        'args': (['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA'], '5min'),
    },
    'batch-analyze-market-hours': {
        'task': 'candlestick.batch_analyze',
        'schedule': timedelta(hours=1),  # Every hour during market hours
        'args': (['SPY', 'QQQ', 'IWM', 'VXX'], '5min', 50),
    },
}

# Monitoring and logging
worker_send_task_events = True
task_send_sent_event = True

# Error handling
task_publish_retry = True
task_publish_retry_policy = {
    'max_retries': 3,
    'interval_start': 0,
    'interval_step': 0.2,
    'interval_max': 0.5,
}

# Time limits
task_time_limit = 300  # 5 minutes
task_soft_time_limit = 240  # 4 minutes

# Concurrency
worker_concurrency = 4  # Number of worker processes

# Logging
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s'

# Redis settings
redis_max_connections = 20
redis_socket_connect_timeout = 30
redis_socket_timeout = 30

# Flower monitoring (optional)
# flower_port = 5555
# flower_address = '127.0.0.1'