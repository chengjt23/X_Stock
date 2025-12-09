import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_process import DataProcessor
from model.model import XGBoostModel
from tqdm import tqdm


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
        
        label_counts = {0: 0, 1: 0, 2: 0}
        for buffer_file in tqdm(buffer_files, desc="Calculating class weights", ncols=80):
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
    
    print("Loading buffer file lists...")
    with open(train_meta, 'r') as f:
        buffer_files = [line.strip() for line in f if line.strip()]
    
    with open(val_meta, 'r') as f:
        val_buffer_files = [line.strip() for line in f if line.strip()]
    
    print(f"Using external memory training with {len(buffer_files)} train buffers and {len(val_buffer_files)} val buffers")
    
    class DMatrixIterator(xgb.DataIter):
        def __init__(self, buffer_files, class_weights=None, desc="Loading", cache_prefix=None):
            self.buffer_files = buffer_files
            self.class_weights = class_weights
            self.current_idx = 0
            self.desc = desc
            self.pbar = None
            self._cache_prefix = cache_prefix
            super().__init__(cache_prefix=cache_prefix)
        
        def next(self, input_data):
            if self.current_idx >= len(self.buffer_files):
                if self.pbar:
                    self.pbar.close()
                    self.pbar = None
                return 0
            if self.pbar is None:
                self.pbar = tqdm(total=len(self.buffer_files), desc=self.desc, ncols=80)
            dm = xgb.DMatrix(self.buffer_files[self.current_idx])
            X = dm.get_data()
            y = dm.get_label()
            if self.class_weights:
                w = np.array([self.class_weights.get(int(label), 1.0) for label in y])
                input_data(data=X, label=y, weight=w)
            else:
                input_data(data=X, label=y)
            self.current_idx += 1
            self.pbar.update(1)
            return 1
        
        def reset(self):
            self.current_idx = 0
            if self.pbar:
                self.pbar.close()
                self.pbar = None
    
    print("Creating ExtMemQuantileDMatrix for true external memory training...")
    train_cache = os.path.join(cache_dir, f"extmem_train_{label_name}")
    val_cache = os.path.join(cache_dir, f"extmem_val_{label_name}")
    
    train_iter = DMatrixIterator(buffer_files, class_weights, desc="Loading train data", cache_prefix=train_cache)
    dtrain = xgb.ExtMemQuantileDMatrix(train_iter, max_bin=256)
    
    val_iter = DMatrixIterator(val_buffer_files, desc="Loading val data", cache_prefix=val_cache)
    dval = xgb.ExtMemQuantileDMatrix(val_iter, max_bin=256, ref=dtrain)
    
    print(f"Train samples: {dtrain.num_row()}, Features: {dtrain.num_col()}")
    print(f"Validation samples: {dval.num_row()}")
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': random_state,
        'nthread': -1,
        'tree_method': 'hist'
    }
    
    print("Training model with external memory (each tree covers all data)...")
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'eval')] if dval else None,
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    model = XGBoostModel(**params)
    model.model = xgb_model
    
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

