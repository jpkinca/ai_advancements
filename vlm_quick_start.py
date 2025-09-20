#!/usr/bin/env python3
"""
Quick Start Script for IBD 50 VLM Training

This script provides easy-to-use commands for training VLM models on IBD 50 stocks.

Usage:
    python vlm_quick_start.py --quick-test        # Test on 5 stocks
    python vlm_quick_start.py --full-training     # Full IBD 50 training
    python vlm_quick_start.py --custom-stocks 10  # Custom number of stocks
    python vlm_quick_start.py --evaluate-model    # Evaluate existing model
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'transformers', 'pandas', 'numpy',
        'yfinance', 'matplotlib', 'mplfinance', 'PIL', 'tqdm'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    # Skip datasets package check for now due to metadata issues
    print("⚠️  Skipping datasets package check due to metadata issues")
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstall with: pip install -r requirements_vlm.txt")
        return False

    print("✅ All required packages are installed")
    return True

def run_quick_test(num_stocks: int = 5):
    """Run quick test training"""
    print(f"\n🚀 Starting Quick IBD 50 Test ({num_stocks} stocks)")
    print("=" * 50)

    from vlm_ibd50_training import quick_ibd50_test

    try:
        results = quick_ibd50_test(num_stocks)

        print("\n✅ Quick test completed successfully!")
        print(f"📊 Processed {results['stocks_processed']} stocks")
        print(f"⏱️  Training time: {results.get('training_time', 0):.1f} hours")
        print(f"📁 Results saved to: vlm/models/ibd50/training_summary.json")

        return results

    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        return None

def run_full_training():
    """Run full IBD 50 training"""
    print("\n🚀 Starting Full IBD 50 Training Pipeline")
    print("=" * 50)
    print("This will train VLM models on all 50 IBD stocks.")
    print("Estimated time: 2-4 hours (depending on hardware)")
    print("Required disk space: ~5-10GB")
    print()

    response = input("Continue? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        return None

    from vlm_ibd50_training import run_ibd50_training

    try:
        results = run_ibd50_training()

        print("\n✅ Full training completed successfully!")
        print(f"📊 Processed {results['stocks_processed']} stocks")
        print(f"⏱️  Training time: {results.get('training_time', 0):.1f} hours")
        print(f"📁 Models saved to: {results['model_path']}")
        print(f"📁 Dataset saved to: {results['dataset_path']}")

        return results

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return None

def evaluate_existing_model():
    """Evaluate existing trained model"""
    print("\n🔍 Evaluating Existing VLM Model")
    print("=" * 30)

    model_path = "vlm/models/ibd50/clip_models/best_model.pt"
    dataset_path = "vlm/data/ibd50/ibd50_training_dataset_v1.0"

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None

    from vlm_ibd50_training import IBD50VLMTrainer

    try:
        trainer = IBD50VLMTrainer()
        results = trainer.evaluate_models(model_path, dataset_path)

        print("\n✅ Evaluation completed!")
        print(f"🎯 Accuracy: {results.get('accuracy', 0):.4f}")
        print(f"📈 Precision: {results.get('precision', 0):.4f}")
        print(f"🔄 Recall: {results.get('recall', 0):.4f}")
        return results

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return None

def show_training_status():
    """Show current training status"""
    print("\n📊 VLM Training Status")
    print("=" * 30)

    # Check for existing models and data
    model_path = Path("vlm/models/ibd50/clip_models/best_model.pt")
    dataset_path = Path("vlm/data/ibd50/ibd50_training_dataset_v1.0")
    summary_path = Path("vlm/models/ibd50/training_summary.json")

    print(f"Model exists:     {'✅' if model_path.exists() else '❌'} {model_path}")
    print(f"Dataset exists:   {'✅' if dataset_path.exists() else '❌'} {dataset_path}")
    print(f"Summary exists:   {'✅' if summary_path.exists() else '❌'} {summary_path}")

    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            print("\nLast training:")
            print(f"  Status: {summary.get('status', 'unknown')}")
            print(f"  Time: {summary.get('training_time', 0):.1f} hours")
            print(f"  Stocks: {summary.get('stocks_processed', 0)}")
            print(f"  Samples: {summary.get('total_samples', 0)}")
            print(f"  Date: {summary.get('timestamp', 'unknown')}")

            if 'evaluation_results' in summary:
                eval_res = summary['evaluation_results']
                print("\nPerformance:")
                print(f"  Accuracy: {eval_res.get('accuracy', 0):.4f}")
                print(f"  Precision: {eval_res.get('precision', 0):.4f}")
                print(f"  Recall: {eval_res.get('recall', 0):.4f}")
        except Exception as e:
            print(f"Error reading summary: {e}")

def show_usage():
    """Show usage instructions"""
    print("""
IBD 50 VLM Training Quick Start
================================

This script helps you train Vision-Language Models on IBD 50 stocks.

COMMANDS:
  --quick-test        Run quick test on 5 stocks (recommended first)
  --full-training     Run complete training on all 50 IBD stocks
  --custom-stocks N   Run test on N stocks
  --evaluate-model    Evaluate existing trained model
  --status           Show current training status
  --help             Show this help message

EXAMPLES:
  python vlm_quick_start.py --quick-test
  python vlm_quick_start.py --full-training
  python vlm_quick_start.py --custom-stocks 10
  python vlm_quick_start.py --evaluate-model

REQUIREMENTS:
  - Python 3.8+
  - GPU recommended (CUDA-compatible)
  - 8GB+ RAM
  - Internet connection for data fetching

OUTPUT:
  - Models: vlm/models/ibd50/
  - Data: vlm/data/ibd50/
  - Logs: vlm_ibd50_training.log

For detailed documentation, see: vlm/Visual Language Model.md
""")

def main():
    parser = argparse.ArgumentParser(description="IBD 50 VLM Training Quick Start")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test on 5 stocks")
    parser.add_argument("--full-training", action="store_true", help="Run full IBD 50 training")
    parser.add_argument("--custom-stocks", type=int, help="Run test on N stocks")
    parser.add_argument("--evaluate-model", action="store_true", help="Evaluate existing model")
    parser.add_argument("--status", action="store_true", help="Show training status")

    args = parser.parse_args()

    # Check requirements first
    if not check_requirements():
        sys.exit(1)

    # Handle commands
    if args.quick_test:
        run_quick_test()
    elif args.full_training:
        run_full_training()
    elif args.custom_stocks:
        run_quick_test(args.custom_stocks)
    elif args.evaluate_model:
        evaluate_existing_model()
    elif args.status:
        show_training_status()
    else:
        show_usage()

if __name__ == "__main__":
    main()