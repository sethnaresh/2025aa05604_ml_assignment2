"""
Offline training script to train all models on the full dataset.
Run this once to generate saved models and preprocessor.

Usage:
    python train_offline.py --data data/ecommerce_customer_churn_dataset.csv --target Churn
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from model.train_models import train_all_models
from model.utils import build_preprocessor


def train_and_save(data_path: str, target_col: str, output_dir: str = "model/saved", test_size: float = 0.2):
    """Train all models on training set and save them with preprocessor. Save test set separately."""
    from sklearn.model_selection import train_test_split
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"📖 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"   Dataset shape: {df.shape}")
    
    # Validate target column
    if target_col not in df.columns:
        print(f"❌ Error: Target column '{target_col}' not found in dataset.")
        sys.exit(1)
    
    # Split into train and test
    print(f"✂️  Splitting dataset (80% train, 20% test)...")
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42,
        stratify=df[target_col] if df[target_col].nunique() <= 50 else None
    )
    print(f"   Training set: {train_df.shape[0]} rows")
    print(f"   Test set: {test_df.shape[0]} rows")
    
    # Save test dataset
    test_path = Path("data/test_dataset.csv")
    test_df.to_csv(test_path, index=False)
    print(f"   ✅ Test dataset saved → {test_path}")
    
    # Build and fit preprocessor on TRAINING dataset only
    print(f"🔧 Building preprocessor on training data...")
    preprocessor = build_preprocessor(train_df, target_col)
    
    # Prepare training data
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    # Fit preprocessor on training data only
    preprocessor.fit(X_train)
    X_processed = preprocessor.transform(X_train)
    
    print(f"✂️  Processed features shape: {X_processed.shape}")
    
    # Train all models on TRAINING dataset only
    print(f"🤖 Training all models on training set...")
    models = train_all_models(X_processed, y_train, random_state=42)
    
    # Save models
    print(f"💾 Saving models...")
    for name, model in models.items():
        if model is not None:
            model_path = output_dir / f"{name}_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"   ✅ Saved {name} → {model_path}")
        else:
            print(f"   ⏭️  Skipped {name} (not installed)")
    
    # Save preprocessor
    preprocessor_path = output_dir / "preprocessor.pkl"
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"   ✅ Saved preprocessor → {preprocessor_path}")
    
    # Save metadata
    metadata_path = output_dir / "metadata.pkl"
    metadata = {
        "target_col": target_col,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "classes": sorted(y_train.unique().tolist()),
        "test_size": test_size,
    }
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"   ✅ Saved metadata → {metadata_path}")
    
    print(f"\n✨ Training complete! Models ready for inference.")
    print(f"\n📊 Summary:")
    print(f"   Training set: {train_df.shape[0]} rows (80%)")
    print(f"   Test set: {test_df.shape[0]} rows (20%) → data/test_dataset.csv")
    print(f"   Models: 6 trained models in {output_dir}/")
    print(f"   Preprocessor: fitted on training data only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and save all models on full dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training CSV dataset"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target column name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model/saved",
        help="Output directory for saved models (default: model/saved)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size ratio (default: 0.2 = 20%)"
    )
    
    args = parser.parse_args()
    train_and_save(args.data, args.target, args.output, args.test_size)
