import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_process import DataProcessor
from model.model import XGBoostModel


def train_model(data_dir, label_name, model_save_dir, window=100, test_size=0.2, random_state=42, cache_dir='./cache', batch_size=5000):
    print(f"Training model for {label_name}")
    print(f"Data directory: {data_dir}")
    print(f"Window size: {window}")
    print(f"Batch size: {batch_size}")
    
    processor = DataProcessor(data_dir)
    
    print("Processing data and saving to binary DMatrix blocks...")
    train_meta, val_meta, train_label_counts = processor.get_train_test_split(
        label_name=label_name,
        test_size=test_size,
        random_state=random_state,
        window=window,
        cache_dir=cache_dir,
        batch_size=batch_size
    )
    
    import numpy as np
    import xgboost as xgb
    
    print("Calculating class weights...")
    total = sum(train_label_counts.values())
    num_classes = len([v for v in train_label_counts.values() if v > 0])
    class_weights = {}
    for label, count in train_label_counts.items():
        class_weights[label] = total / (num_classes * count) if count > 0 else 1.0
    print(f"Class distribution - Down: {train_label_counts.get(0, 0)}, Unchanged: {train_label_counts.get(1, 0)}, Up: {train_label_counts.get(2, 0)}")
    print(f"Class weights - Down: {class_weights.get(0, 1.0):.3f}, Unchanged: {class_weights.get(1, 1.0):.3f}, Up: {class_weights.get(2, 1.0):.3f}")
    
    print("Loading external memory DMatrix from meta files...")
    dtrain = xgb.DMatrix(train_meta)
    train_labels = dtrain.get_label().astype(int)
    sample_weights = np.array([class_weights.get(label, 1.0) for label in train_labels])
    dtrain.set_weight(sample_weights)
    dval = xgb.DMatrix(val_meta)
    
    print(f"Train samples: {dtrain.num_row()}, Features: {dtrain.num_col()}")
    print(f"Validation samples: {dval.num_row()}")
    
    model_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_boost_round': 500,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': random_state,
        'nthread': 4,
        'tree_method': 'hist'
    }
    model = XGBoostModel(**model_params)
    
    print("Training model...")
    model.fit(dtrain, dval=dval, early_stopping_rounds=10)
    
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"model_{label_name}.pth")
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    print("Evaluating on validation set...")
    pred_proba = model.predict_proba(dval)
    pred = pred_proba.argmax(axis=1)
    
    from sklearn.metrics import accuracy_score, classification_report
    
    val_labels = dval.get_label().astype(int)
    accuracy = accuracy_score(val_labels, pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(val_labels, pred, target_names=['Down', 'Unchanged', 'Up']))
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost models for stock price direction prediction')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--model_save_dir', type=str, default='./Experiments', help='Directory to save trained models')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Directory for binary DMatrix cache files')
    parser.add_argument('--label', type=str, default='all', 
                       choices=['label_5', 'label_10', 'label_20', 'label_40', 'label_60', 'all'],
                       help='Label to train (default: all)')
    parser.add_argument('--window', type=int, default=100, help='Window size for historical features')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for processing data blocks')
    
    args = parser.parse_args()
    
    labels = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60'] if args.label == 'all' else [args.label]
    
    for label in labels:
        print(f"\n{'='*60}")
        train_model(
            data_dir=args.data_dir,
            label_name=label,
            model_save_dir=args.model_save_dir,
            window=args.window,
            test_size=args.test_size,
            random_state=args.random_state,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size
        )
        print(f"{'='*60}\n")
    
    print("All models trained successfully!")


if __name__ == '__main__':
    main()

