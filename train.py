import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_process import DataProcessor
from model.model import XGBoostModel


def train_model(data_dir, label_name, model_save_dir, window=100, test_size=0.2, random_state=42, cache_dir='./cache', batch_size=5000, use_cache=False):
    print(f"Training model for {label_name}")
    print(f"Data directory: {data_dir}")
    print(f"Window size: {window}")
    print(f"Batch size: {batch_size}")
    print(f"Use cache: {use_cache}")
    
    import numpy as np
    import xgboost as xgb
    
    train_meta = os.path.join(cache_dir, f'train_{label_name}.meta')
    val_meta = os.path.join(cache_dir, f'val_{label_name}.meta')
    
    if use_cache:
        if not os.path.exists(train_meta) or not os.path.exists(val_meta):
            raise FileNotFoundError(f"Cache files not found. train_meta: {train_meta}, val_meta: {val_meta}")
        
        print("Using existing cache files...")
        
        with open(train_meta, 'r') as f:
            buffer_files = [line.strip() for line in f if line.strip()]
        
        print("Calculating class weights from cache...")
        label_counts = {0: 0, 1: 0, 2: 0}
        for buffer_file in buffer_files:
            if os.path.exists(buffer_file):
                dm = xgb.DMatrix(buffer_file)
                labels = dm.get_label().astype(int)
                unique, counts = np.unique(labels, return_counts=True)
                for label, count in zip(unique, counts):
                    label_counts[int(label)] = label_counts.get(int(label), 0) + int(count)
        
        train_label_counts = label_counts
    else:
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
    
    print("Calculating class weights...")
    total = sum(train_label_counts.values())
    num_classes = len([v for v in train_label_counts.values() if v > 0])
    class_weights = {}
    for label, count in train_label_counts.items():
        class_weights[label] = total / (num_classes * count) if count > 0 else 1.0
    print(f"Class distribution - Down: {train_label_counts.get(0, 0)}, Unchanged: {train_label_counts.get(1, 0)}, Up: {train_label_counts.get(2, 0)}")
    print(f"Class weights - Down: {class_weights.get(0, 1.0):.3f}, Unchanged: {class_weights.get(1, 1.0):.3f}, Up: {class_weights.get(2, 1.0):.3f}")
    
    print("Loading external memory DMatrix from meta files...")
    with open(train_meta, 'r') as f:
        buffer_files = [line.strip() for line in f if line.strip()]
    
    with open(val_meta, 'r') as f:
        val_buffer_files = [line.strip() for line in f if line.strip()]
    
    if len(buffer_files) == 1 and len(val_buffer_files) == 1:
        dtrain = xgb.DMatrix(buffer_files[0])
        train_labels = dtrain.get_label().astype(int)
        sample_weights = np.array([class_weights.get(label, 1.0) for label in train_labels])
        dtrain.set_weight(sample_weights)
        dval = xgb.DMatrix(val_buffer_files[0])
        
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
    else:
        print(f"Using incremental training with {len(buffer_files)} train buffers and {len(val_buffer_files)} val buffers")
        
        def train_dmatrix_generator():
            for buffer_file in buffer_files:
                dm = xgb.DMatrix(buffer_file)
                labels = dm.get_label().astype(int)
                weights = np.array([class_weights.get(label, 1.0) for label in labels])
                dm.set_weight(weights)
                yield dm
        
        def val_dmatrix_generator():
            for buffer_file in val_buffer_files:
                yield xgb.DMatrix(buffer_file)
        
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
        
        print("Training model incrementally...")
        train_gen = train_dmatrix_generator()
        val_gen = val_dmatrix_generator() if val_buffer_files else None
        
        params = model_params.copy()
        params.pop('num_boost_round', None)
        num_boost_round = model_params.get('num_boost_round', 500)
        rounds_per_buffer = max(5, num_boost_round // max(1, len(buffer_files) // 10))
        
        first_train = next(train_gen)
        first_val = next(val_gen) if val_gen else None
        
        print(f"Training on first buffer with {num_boost_round} rounds...")
        if first_val is not None:
            model.model = xgb.train(
                params, first_train,
                num_boost_round=num_boost_round,
                evals=[(first_val, 'eval')],
                early_stopping_rounds=10,
                verbose_eval=20
            )
        else:
            model.model = xgb.train(params, first_train, num_boost_round=num_boost_round)
        
        buffer_idx = 1
        for train_dm in train_gen:
            val_dm = next(val_gen, None) if val_gen else None
            if val_dm is not None:
                model.model = xgb.train(
                    params, train_dm,
                    num_boost_round=rounds_per_buffer,
                    xgb_model=model.model,
                    evals=[(val_dm, 'eval')],
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
            else:
                model.model = xgb.train(
                    params, train_dm,
                    num_boost_round=rounds_per_buffer,
                    xgb_model=model.model
                )
            buffer_idx += 1
            if buffer_idx % 50 == 0:
                print(f"Processed {buffer_idx}/{len(buffer_files)} buffers...")
        
        dval = first_val if val_buffer_files else None
    
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
    parser.add_argument('--use_cache', action='store_true', default=False, help='Use existing cache files instead of processing data')
    
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
            use_cache=args.use_cache
        )
        print(f"{'='*60}\n")
    
    print("All models trained successfully!")


if __name__ == '__main__':
    main()

