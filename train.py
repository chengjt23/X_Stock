import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_process import DataProcessor
from model.model import XGBoostModel


def train_model(data_dir, label_name, model_save_dir, window=100, test_size=0.2, random_state=42, cache_dir='./cache', batch_size=10000, rounds_per_batch=20):
    print(f"Training model for {label_name}")
    print(f"Data directory: {data_dir}")
    print(f"Window size: {window}")
    
    processor = DataProcessor(data_dir)
    
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, classification_report
    
    # 划分文件以避免跨文件泄漏
    train_files, val_files = processor._split_files(test_size)
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    
    # 仅统计标签分布，避免提前加载全部特征
    print("Calculating class weights...")
    label_counts = processor.get_label_distribution(train_files, label_name)
    total = sum(label_counts.values()) or 1
    num_classes = len([v for v in label_counts.values() if v > 0]) or 1
    class_weights = {lbl: (total / (num_classes * cnt)) if cnt > 0 else 1.0 for lbl, cnt in label_counts.items()}
    print(f"Class distribution {label_counts}")
    print(f"Class weights {class_weights}")

    def train_generator():
        base_gen = processor._load_and_process_files_generator(train_files, label_name, window, batch_size)
        for X_batch, y_batch in base_gen:
            weights = np.array([class_weights.get(int(lbl), 1.0) for lbl in y_batch])
            yield X_batch, y_batch, weights

    def val_generator():
        for X_batch, y_batch in processor._load_and_process_files_generator(val_files, label_name, window, batch_size):
            yield X_batch, y_batch
    
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
    
    print("Training model (incremental batches)...")
    model.fit_incremental(train_generator(), val_gen=val_generator(), rounds_per_batch=rounds_per_batch, early_stopping_rounds=10)
    
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"model_{label_name}.pth")
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    print("Evaluating on validation set...")
    val_true = []
    val_pred = []
    val_proba = []
    for X_batch, y_batch in val_generator():
        dval_batch = xgb.DMatrix(X_batch)
        batch_proba = model.predict_proba(dval_batch)
        batch_pred = batch_proba.argmax(axis=1)
        val_true.extend(y_batch.tolist())
        val_pred.extend(batch_pred.tolist())
        val_proba.append(batch_proba)
    if val_true:
        accuracy = accuracy_score(val_true, val_pred)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(val_true, val_pred, target_names=['Down', 'Unchanged', 'Up']))
    else:
        print("Validation set is empty, skip evaluation.")
    
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
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size when streaming features')
    parser.add_argument('--rounds_per_batch', type=int, default=20, help='Boosting rounds per streamed batch')
    
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
            batch_size=args.batch_size,
            rounds_per_batch=args.rounds_per_batch
        )
        print(f"{'='*60}\n")
    
    print("All models trained successfully!")


if __name__ == '__main__':
    main()

