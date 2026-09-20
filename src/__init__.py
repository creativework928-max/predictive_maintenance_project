"""
Predictive Maintenance Project
Task 2 - Feature Engineering

Source package initialization.

This package contains the reusable components for:
- Data loading
- Feature engineering
- Feature quality analysis
- Visualization
- HTML/DOCX reporting
- Real-time/online feature generation

Project structure

src/
    __init__.py
    config.py
    data_loader.py
    feature_engineering.py
    feature_quality.py
    visualization.py
    report_generator.py
    realtime_features.py
"""

from __future__ import annotations


# Project metadata

project_name = "Predictive Maintenance Project"
task_name = "Task 2 - Feature Engineering"
version = "1.0.0"


# Public package metadata

author = "Predictive Maintenance Engineering Team"
license = "MIT"


__all__ = [
    "project_name",
    "task_name",
    "version",
    "author",
    "license",
]
