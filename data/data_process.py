import os
import pandas as pd
import glob
import xgboost as xgb
import numpy as np
from tqdm import tqdm


class DataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_columns = [
            'n_close', 'amount_delta', 'n_midprice',
            'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3',
            'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5',
            'n_ask1', 'n_asize1', 'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3',
            'n_ask4', 'n_asize4', 'n_ask5', 'n_asize5'
        ]
        self.label_columns = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
        self.label_horizons = {
            'label_5': 5,
            'label_10': 10,
            'label_20': 20,
            'label_40': 40,
            'label_60': 60
        }
        self.alpha_thresholds = {
            5: 0.0005,
            10: 0.0005,
            20: 0.001,
            40: 0.001,
            60: 0.001
        }
    
    def load_all_data(self):
        pattern = os.path.join(self.data_dir, 'snapshot_sym*_date*_*.csv')
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        dataframes = []
        for file in tqdm(csv_files, desc="Loading data", ncols=80):
            try:
                dataframes.append(pd.read_csv(file))
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}")
        
        if not dataframes:
            raise ValueError("No valid CSV files were loaded")
        
        return pd.concat(dataframes, ignore_index=True)
    
    def create_features(self, df, window=100):
        df = df.sort_values(['sym', 'date', 'time']).reset_index(drop=True)
        
        available_features = [col for col in self.feature_columns if col in df.columns]
        feature_df = df[available_features].copy()
        
        for col in feature_df.columns:
            feature_df[col] = feature_df[col].fillna(0)
        
        grouped = df.groupby(['sym', 'date'])
        total_rows = len(df)
        
        feature_list = []
        processed_rows = 0
        pbar = tqdm(total=total_rows, desc="Creating features", ncols=80)
        
        for (sym, date), group in grouped:
            group_features = feature_df.loc[group.index].values
            
            for i in range(len(group)):
                if i < window - 1:
                    hist_features = group_features[:i+1]
                    padding = np.zeros((window - i - 1, len(available_features)))
                    hist_features = np.vstack([padding, hist_features])
                else:
                    hist_features = group_features[i-window+1:i+1]
                
                feature_list.append(hist_features.flatten())
                processed_rows += 1
                pbar.update(1)
        
        pbar.close()
        return np.array(feature_list)
    
    def process_data(self, label_name='label_5', window=100):
        print("Processing data...")
        df = self.load_all_data()
        print("Data loaded successfully")
        
        if label_name not in self.label_columns:
            raise ValueError(f"Label {label_name} not found. Available labels: {self.label_columns}")
        
        df = df.sort_values(['sym', 'date', 'time']).reset_index(drop=True)
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        X = self.create_features(df, window)
        
        grouped = df.groupby(['sym', 'date'])
        y_list = []
        for (sym, date), group in grouped:
            y_list.extend(group[label_name].fillna(1).astype(int).values)
        
        y = np.array(y_list)
        
        return xgb.DMatrix(X, label=y)
    
    def get_train_test_split(self, label_name='label_5', test_size=0.2, random_state=42, window=100, cache_dir='./cache', batch_size=50000):
        if label_name not in self.label_columns:
            raise ValueError(f"Label {label_name} not found. Available labels: {self.label_columns}")
        
        os.makedirs(cache_dir, exist_ok=True)
        train_meta = os.path.join(cache_dir, f'train_{label_name}.meta')
        val_meta = os.path.join(cache_dir, f'val_{label_name}.meta')
        
        train_files, val_files = self._split_files(test_size)
        
        sample_file = train_files[0] if train_files else val_files[0]
        available_features = [col for col in self.feature_columns if col in pd.read_csv(sample_file, nrows=1).columns]
        
        print("Processing training data and saving to binary DMatrix blocks...")
        train_labels = self._process_and_save_binary_blocks(train_files, label_name, window, available_features, train_meta, cache_dir, batch_size)
        
        print("Processing validation data and saving to binary DMatrix blocks...")
        self._process_and_save_binary_blocks(val_files, label_name, window, available_features, val_meta, cache_dir, batch_size)
        
        return train_meta, val_meta, np.array(train_labels)
    
    def _split_files(self, test_size):
        pattern = os.path.join(self.data_dir, 'snapshot_sym*_date*_*.csv')
        csv_files = sorted(glob.glob(pattern))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        file_infos = []
        for file_path in csv_files:
            date_idx, session_idx, sym_idx = self._parse_file_metadata(file_path)
            file_infos.append((date_idx, session_idx, sym_idx, file_path))
        
        if not file_infos:
            raise ValueError("No parsable CSV files were found.")
        
        session_keys = sorted({(info[0], info[1]) for info in file_infos})
        if len(session_keys) > 1:
            split_idx = max(1, int(len(session_keys) * (1 - test_size)))
            if split_idx >= len(session_keys):
                split_idx = len(session_keys) - 1
            train_key_set = set(session_keys[:split_idx])
            val_key_set = set(session_keys[split_idx:])
            train_files = [info[3] for info in file_infos if (info[0], info[1]) in train_key_set]
            val_files = [info[3] for info in file_infos if (info[0], info[1]) in val_key_set]
        else:
            split_file_idx = max(1, int(len(file_infos) * (1 - test_size)))
            if split_file_idx >= len(file_infos):
                split_file_idx = len(file_infos) - 1
            sorted_files = [info[3] for info in sorted(file_infos, key=lambda x: (x[0], x[1], x[2]))]
            train_files = sorted_files[:split_file_idx]
            val_files = sorted_files[split_file_idx:]
        
        if not val_files:
            raise ValueError("Validation split is empty. Reduce test_size or provide more data.")
        
        return train_files, val_files

    def _parse_file_metadata(self, file_path):
        name = os.path.basename(file_path)
        try:
            parts = name.split('_')
            sym_part = parts[1] if len(parts) > 1 else 'sym0'
            date_part = parts[2] if len(parts) > 2 else 'date0'
            session_part = parts[3].split('.')[0] if len(parts) > 3 else ''
            sym_idx = int(sym_part.replace('sym', '') or 0)
            date_idx = int(date_part.replace('date', '') or 0)
            session_lower = session_part.lower()
            if 'pm' in session_lower:
                session_idx = 1
            else:
                session_idx = 0
            return date_idx, session_idx, sym_idx
        except Exception:
            return (0, 0, 0)

    def _compute_midprice_series(self, df):
        if 'midprice' in df.columns:
            mid = df['midprice'].copy()
        elif 'n_midprice' in df.columns:
            mid = df['n_midprice'].copy()
        elif {'ask1', 'bid1'}.issubset(df.columns):
            ask = df['ask1'].fillna(0)
            bid = df['bid1'].fillna(0)
            mid = np.where(
                (ask != 0) & (bid != 0),
                (ask + bid) / 2,
                np.where(bid == 0, ask, bid)
            )
            mid = pd.Series(mid, index=df.index)
        else:
            raise ValueError("Midprice columns not found (expected midprice/n_midprice or ask1/bid1).")
        return mid

    def verify_label_formula(self, df, label_names=None):
        df = df.sort_values(['sym', 'date', 'time']).reset_index(drop=True)
        label_names = label_names or self.label_columns
        mid = self._compute_midprice_series(df)
        df = df.copy()
        df['_mid_'] = mid
        grouped = df.groupby(['sym', 'date'], sort=False)
        report = {}
        for label in label_names:
            if label not in df.columns or label not in self.label_horizons:
                continue
            horizon = self.label_horizons[label]
            alpha = self.alpha_thresholds.get(horizon, 0.0005)
            future_mid = grouped['_mid_'].shift(-horizon)
            safe_mid = df['_mid_'].replace(0, np.nan)
            rel_move = (future_mid - df['_mid_']) / safe_mid
            calc = np.where(rel_move < -alpha, 0, np.where(rel_move > alpha, 2, 1))
            invalid_mask = future_mid.isna() | safe_mid.isna()
            calc = np.where(invalid_mask, np.nan, calc)
            existing = df[label].to_numpy()
            mismatches = np.sum((existing != calc) & ~np.isnan(calc))
            total = int((~invalid_mask).sum())
            report[label] = {
                'total_checked': total,
                'mismatched': int(mismatches),
                'mismatch_rate': float(mismatches / total) if total else 0.0
            }
        df.drop(columns=['_mid_'], inplace=True)
        return report

    def verify_label_formula_from_file(self, file_path, label_names=None):
        df = pd.read_csv(file_path)
        return self.verify_label_formula(df, label_names=label_names)

    def get_label_distribution(self, csv_files, label_name):
        """
        仅统计标签分布，用于计算类别权重，避免一次性加载全部特征
        """
        counts = {0: 0, 1: 0, 2: 0}
        for labels in self._get_labels_only_generator(csv_files, label_name):
            unique, cnts = np.unique(labels, return_counts=True)
            for lbl, c in zip(unique, cnts):
                counts[int(lbl)] = counts.get(int(lbl), 0) + int(c)
        return counts

    def _process_and_save_binary_blocks(self, csv_files, label_name, window, available_features, meta_path, cache_dir, batch_size=50000):
        batch_features = []
        batch_labels = []
        all_labels = []
        buffer_files = []
        buffer_idx = 0
        
        base_name = os.path.splitext(os.path.basename(meta_path))[0]
        
        def save_buffer():
            nonlocal buffer_idx, batch_features, batch_labels
            if not batch_features:
                return
            X_batch = np.array(batch_features)
            y_batch = np.array(batch_labels)
            buffer_file = os.path.join(cache_dir, f'{base_name}_{buffer_idx}.buffer')
            dtrain_batch = xgb.DMatrix(X_batch, label=y_batch)
            dtrain_batch.save_binary(buffer_file)
            buffer_files.append(buffer_file)
            batch_features = []
            batch_labels = []
            buffer_idx += 1
        
        for file in tqdm(csv_files, desc="Processing files", ncols=80):
            df = pd.read_csv(file)
            df = df.sort_values(['sym', 'date', 'time']).reset_index(drop=True)
            
            feature_df = df[available_features].copy()
            for col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(0)
            
            grouped = df.groupby(['sym', 'date'])
            for (sym, date), group in grouped:
                group_features = feature_df.loc[group.index].values
                group_labels = group[label_name].fillna(1).astype(int).values
                
                for i in range(len(group)):
                    if i < window - 1:
                        hist_features = group_features[:i+1]
                        padding = np.zeros((window - i - 1, len(available_features)))
                        hist_features = np.vstack([padding, hist_features])
                    else:
                        hist_features = group_features[i-window+1:i+1]
                    
                    batch_features.append(hist_features.flatten())
                    batch_labels.append(int(group_labels[i]))
                    all_labels.append(int(group_labels[i]))
                    
                    if len(batch_features) >= batch_size:
                        save_buffer()
            
            del df, feature_df
        
        save_buffer()
        
        if not buffer_files:
            raise ValueError("No data processed")
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            for buffer_file in buffer_files:
                f.write(f'{buffer_file}\n')
        
        total_samples = sum([xgb.DMatrix(bf).num_row() for bf in buffer_files])
        print(f"Saved {len(buffer_files)} binary DMatrix blocks to {meta_path} ({total_samples} total samples)")
        
        return all_labels
    
    def create_features_batch_generator(self, df, label_name, window=100, batch_size=10000):
        """
        分批生成特征和标签，避免内存溢出
        """
        df = df.sort_values(['sym', 'date', 'time']).reset_index(drop=True)
        available_features = [col for col in self.feature_columns if col in df.columns]
        feature_df = df[available_features].copy()
        
        for col in feature_df.columns:
            feature_df[col] = feature_df[col].fillna(0)
        
        grouped = df.groupby(['sym', 'date'])
        batch_features = []
        batch_labels = []
        
        for (sym, date), group in grouped:
            group_features = feature_df.loc[group.index].values
            group_labels = group[label_name].fillna(1).astype(int).values
            
            for i in range(len(group)):
                if i < window - 1:
                    hist_features = group_features[:i+1]
                    padding = np.zeros((window - i - 1, len(available_features)))
                    hist_features = np.vstack([padding, hist_features])
                else:
                    hist_features = group_features[i-window+1:i+1]
                
                batch_features.append(hist_features.flatten())
                batch_labels.append(group_labels[i])
                
                if len(batch_features) >= batch_size:
                    yield np.array(batch_features), np.array(batch_labels)
                    batch_features = []
                    batch_labels = []
        
        if batch_features:
            yield np.array(batch_features), np.array(batch_labels)
    
    def get_train_test_split_batch(self, label_name='label_5', test_size=0.2, window=100, batch_size=10000):
        """
        分批版本：返回生成器而不是完整的DMatrix，避免一次性加载所有数据
        """
        if label_name not in self.label_columns:
            raise ValueError(f"Label {label_name} not found. Available labels: {self.label_columns}")
        
        train_files, val_files = self._split_files(test_size)
        
        # 创建生成器，分批加载文件
        train_gen = self._load_and_process_files_generator(train_files, label_name, window, batch_size)
        val_gen = self._load_and_process_files_generator(val_files, label_name, window, batch_size)
        
        return train_gen, val_gen
    
    def _get_labels_only_generator(self, csv_files, label_name):
        """
        轻量级生成器：只读取标签，不创建特征，用于计算类别权重
        """
        for file in csv_files:
            df = pd.read_csv(file)
            grouped = df.groupby(['sym', 'date'])
            for (sym, date), group in grouped:
                labels = group[label_name].fillna(1).astype(int).values
                yield labels
            del df
    
    def _load_and_process_files_generator(self, csv_files, label_name, window, batch_size):
        """
        分批加载文件并生成特征，避免一次性加载所有数据
        """
        if not csv_files:
            return
        
        available_features = [col for col in self.feature_columns if col in pd.read_csv(csv_files[0], nrows=1).columns]
        batch_features = []
        batch_labels = []
        
        for file in tqdm(csv_files, desc="Processing files", leave=False, ncols=80):
            df = pd.read_csv(file)
            df = df.sort_values(['sym', 'date', 'time']).reset_index(drop=True)
            
            feature_df = df[available_features].copy()
            for col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(0)
            
            grouped = df.groupby(['sym', 'date'])
            
            for (sym, date), group in grouped:
                group_features = feature_df.loc[group.index].values
                group_labels = group[label_name].fillna(1).astype(int).values
                
                for i in range(len(group)):
                    if i < window - 1:
                        hist_features = group_features[:i+1]
                        padding = np.zeros((window - i - 1, len(available_features)))
                        hist_features = np.vstack([padding, hist_features])
                    else:
                        hist_features = group_features[i-window+1:i+1]
                    
                    batch_features.append(hist_features.flatten())
                    batch_labels.append(group_labels[i])
                    
                    if len(batch_features) >= batch_size:
                        yield np.array(batch_features), np.array(batch_labels)
                        batch_features = []
                        batch_labels = []
            
            del df, feature_df
        
        if batch_features:
            yield np.array(batch_features), np.array(batch_labels)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test data processing module')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--label', type=str, default='label_5', help='Label name to use')
    
    args = parser.parse_args()
    
    processor = DataProcessor(args.data_dir)
    
    print(f"Processing data from: {args.data_dir}")
    print(f"Using label: {args.label}")
    
    dtrain = processor.process_data(args.label)
    print(f"Total data shape: {dtrain.num_row()} samples, {dtrain.num_col()} features")
    
    train_meta, val_meta, train_labels = processor.get_train_test_split(args.label)
    print(f"Binary blocks listed in: {train_meta} (train) and {val_meta} (val)")
    print(f"Cached train rows (approx.): {len(train_labels)}")
    
    print("Data processing test completed successfully!")


if __name__ == '__main__':
    main()

