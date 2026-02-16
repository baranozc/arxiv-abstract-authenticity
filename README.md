# Article Summary AI Detector

This project is a machine learning-based tool designed to analyze article summaries and predict the probability of them being human-written or AI-generated.

## Overview

The application processes text input and provides a percentage-based score indicating the likelihood of AI involvement. It utilizes multiple classification models and follows clean code principles to ensure a modular and scalable architecture.

## Core Features

    Probabilistic Classification: Instead of a simple binary output, the system provides a confidence percentage for its predictions.

    Model Variety: Supports multiple pre-trained classification models including Logistic Regression, Naive Bayes, and Random Forest.

    Robust Data Management: Features a normalized SQLite database to store results and metadata efficiently.

    Design Patterns: Implements the Singleton pattern for database connection management to ensure resource efficiency and data consistency.

    Code Quality: Integrated with SonarCloud via GitHub Actions for continuous quality gate checks and security scanning.

# Project Structure

    AIResultService.py: Manages the core logic for processing summaries and returning AI detection results.

    DBConnectorService.py: Handles all interactions with the SQLite database using a Singleton instance.

    main.py: The primary entry point for the application.

    test_suite.py: Contains comprehensive whitebox tests to verify the integrity of the detection logic and database operations.

    .pkl files: Serialized machine learning models (Logistic Regression, Naive Bayes, Random Forest) used for inference.

# Technical Implementation

This project reflects a strong emphasis on software engineering best practices:

    Database Normalization: The SQLite schema is designed to reduce redundancy and maintain data integrity.

    SecDevOps: GitHub Actions are configured to pin specific commit SHAs for third-party actions, ensuring supply chain security.

    Modularization: High cohesion and low coupling are maintained across the service layers.
