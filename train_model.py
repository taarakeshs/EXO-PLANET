"""
Training Script for AI-Based Intelligent Exoplanet Detection System
-------------------------------------------------------------------
Loads real NASA Kepler KOI dataset, preprocesses features, splits data,
trains Random Forest & Logistic Regression baseline, generates empirical metrics,
and saves pipeline objects and metadata.
"""

import sys
from src.training import train_and_evaluate_models

# Ensure UTF-8 output formatting for terminal compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    print("==================================================")
    print("Starting Exoplanet Model Training Pipeline...")
    print("==================================================")
    metadata = train_and_evaluate_models()
    print("\nDataset Info:")
    print(f" - Total Observations: {metadata['num_samples']}")
    print(f" - Training Set: {metadata['num_train_samples']}")
    print(f" - Test Set: {metadata['num_test_samples']}")
    print(f" - Features ({len(metadata['features'])}): {', '.join(metadata['features'])}")
    print(f" - Target Classes ({len(metadata['classes'])}): {', '.join(metadata['classes'])}")
    
    print("\nEmpirical Test Results:")
    print("Random Forest Classifier:")
    print(f" - Accuracy:  {metadata['rf_metrics']['accuracy']*100:.2f}%")
    print(f" - F1 (Weighted): {metadata['rf_metrics']['f1_weighted']*100:.2f}%")
    print("\nLogistic Regression Baseline:")
    print(f" - Accuracy:  {metadata['lr_metrics']['accuracy']*100:.2f}%")
    print(f" - F1 (Weighted): {metadata['lr_metrics']['f1_weighted']*100:.2f}%")
    print("\nPipeline artifacts successfully exported to models/")
