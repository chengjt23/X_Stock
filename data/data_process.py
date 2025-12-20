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
            'n_ask4', 'n_asize4', 'n_ask5', 'n_asize5',
            'spread_1', 'spread_3', 'spread_5',
            'mid_price_1', 'mid_price_3', 'mid_price_5',
            'relative_bid_density_1', 'relative_ask_density_1',
            'relative_bid_density_3', 'relative_ask_density_3',
            'weighted_ab_1', 'weighted_ab_3',
            'vol1_rel_diff', 'vol3_rel_diff', 'vol5_rel_diff',
            'amount_normalized',
            'log_bsize1', 'log_asize1', 'log_bsize3', 'log_asize3', 'log_bsize5', 'log_asize5',
            'close_delta', 'bid1_delta', 'ask1_delta', 'midprice_delta',
            'close_mean', 'close_std', 'close_vs_mean',
            'bid1_mean', 'bid1_std', 'bid1_vs_mean',
            'bid3_mean', 'bid3_std', 'bid3_vs_mean',
            'bid5_mean', 'bid5_std', 'bid5_vs_mean',
            'ask1_mean', 'ask1_std', 'ask1_vs_mean',
            'ask3_mean', 'ask3_std', 'ask3_vs_mean',
            'ask5_mean', 'ask5_std', 'ask5_vs_mean',
            'bsize1_mean', 'bsize1_std', 'bsize1_vs_mean',
            'bsize3_mean', 'bsize3_std', 'bsize3_vs_mean',
            'bsize5_mean', 'bsize5_std', 'bsize5_vs_mean',
            'asize1_mean', 'asize1_std', 'asize1_vs_mean',
            'asize3_mean', 'asize3_std', 'asize3_vs_mean',
            'asize5_mean', 'asize5_std', 'asize5_vs_mean',
            'midprice_mean', 'midprice_std',
            'mid_price_1_mean', 'mid_price_1_std',
            'mid_price_3_mean', 'mid_price_3_std',
            'mid_price_5_mean', 'mid_price_5_std',
            'time_seconds', 'time_interval',
            'bid1_plus1', 'bid3_plus1', 'bid5_plus1',
            'ask1_plus1', 'ask3_plus1', 'ask5_plus1',
            'cross_weighted_1', 'cross_weighted_2',
            'midprice_ma5',
            'volatility_5', 'volatility_10', 'volatility_20', 'volatility_40', 'volatility_60',
            'macd_dif', 'macd_dea', 'macd_bar',
            'kdj_k', 'kdj_d', 'kdj_j'
        ]
        self.label_columns = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
    
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
    
    def _add_derived_features(self, df):
        def time_to_seconds(time_str):
            parts = str(time_str).split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        
        df['time_seconds'] = df['time'].apply(time_to_seconds)
        df['time_interval'] = df['time_seconds'].apply(lambda x: min(int((x - 34200) / 1800), 7) if x >= 34200 else 0)
        
        for i in [1, 3, 5]:
            df[f'spread_{i}'] = df[f'n_ask{i}'] - df[f'n_bid{i}']
            df[f'mid_price_{i}'] = (df[f'n_ask{i}'] + df[f'n_bid{i}']) / 2
            total_size = df[f'n_bsize{i}'] + df[f'n_asize{i}']
            df[f'relative_bid_density_{i}'] = df[f'n_bsize{i}'] / (total_size + 1e-10)
            df[f'relative_ask_density_{i}'] = df[f'n_asize{i}'] / (total_size + 1e-10)
        
        for i in [1, 3]:
            df[f'weighted_ab_{i}'] = (df[f'n_bid{i}'] * df[f'n_asize{i}'] + df[f'n_ask{i}'] * df[f'n_bsize{i}']) / (df[f'n_bsize{i}'] + df[f'n_asize{i}'] + 1e-10)
        
        df['vol1_rel_diff'] = (df['n_bsize1'] - df['n_asize1']) / (df['n_bsize1'] + df['n_asize1'] + 1e-10)
        df['vol3_rel_diff'] = (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] - df['n_asize1'] - df['n_asize2'] - df['n_asize3']) / (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] + df['n_asize1'] + df['n_asize2'] + df['n_asize3'] + 1e-10)
        df['vol5_rel_diff'] = (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] + df['n_bsize4'] + df['n_bsize5'] - df['n_asize1'] - df['n_asize2'] - df['n_asize3'] - df['n_asize4'] - df['n_asize5']) / (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] + df['n_bsize4'] + df['n_bsize5'] + df['n_asize1'] + df['n_asize2'] + df['n_asize3'] + df['n_asize4'] + df['n_asize5'] + 1e-10)
        
        df['amount_normalized'] = np.log1p(df['amount_delta'] / (1 + df['n_midprice']))
        
        for i in [1, 3, 5]:
            df[f'log_bsize{i}'] = np.log1p(df[f'n_bsize{i}'])
            df[f'log_asize{i}'] = np.log1p(df[f'n_asize{i}'])
        
        grouped = df.groupby(['sym', 'date'])
        df['close_delta'] = grouped['n_close'].diff()
        df['bid1_delta'] = grouped['n_bid1'].diff()
        df['ask1_delta'] = grouped['n_ask1'].diff()
        df['midprice_delta'] = grouped['n_midprice'].diff()
        
        df['close_mean'] = grouped['n_close'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        df['close_std'] = grouped['n_close'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
        df['close_vs_mean'] = df['n_close'] / (df['close_mean'] + 1e-10)
        
        for i in [1, 3, 5]:
            df[f'bid{i}_mean'] = grouped[f'n_bid{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            df[f'bid{i}_std'] = grouped[f'n_bid{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            df[f'bid{i}_vs_mean'] = df[f'n_bid{i}'] / (df[f'bid{i}_mean'] + 1e-10)
            
            df[f'ask{i}_mean'] = grouped[f'n_ask{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            df[f'ask{i}_std'] = grouped[f'n_ask{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            df[f'ask{i}_vs_mean'] = df[f'n_ask{i}'] / (df[f'ask{i}_mean'] + 1e-10)
            
            df[f'bsize{i}_mean'] = grouped[f'n_bsize{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            df[f'bsize{i}_std'] = grouped[f'n_bsize{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            df[f'bsize{i}_vs_mean'] = df[f'n_bsize{i}'] / (df[f'bsize{i}_mean'] + 1e-10)
            
            df[f'asize{i}_mean'] = grouped[f'n_asize{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            df[f'asize{i}_std'] = grouped[f'n_asize{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            df[f'asize{i}_vs_mean'] = df[f'n_asize{i}'] / (df[f'asize{i}_mean'] + 1e-10)
            
            df[f'mid_price_{i}_mean'] = grouped[f'mid_price_{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            df[f'mid_price_{i}_std'] = grouped[f'mid_price_{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
        
        df['midprice_mean'] = grouped['n_midprice'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        df['midprice_std'] = grouped['n_midprice'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
        
        for i in [1, 3, 5]:
            df[f'bid{i}_plus1'] = df[f'n_bid{i}'] + 1
            df[f'ask{i}_plus1'] = df[f'n_ask{i}'] + 1
        
        df['cross_weighted_1'] = (df['n_ask1'] * df['n_bsize2'] + df['n_ask2'] * df['n_bsize1']) / (df['n_bsize1'] + df['n_bsize2'] + 1e-10)
        df['cross_weighted_2'] = (df['n_bid1'] * df['n_asize2'] + df['n_bid2'] * df['n_asize1']) / (df['n_asize1'] + df['n_asize2'] + 1e-10)
        
        df['midprice_ma5'] = grouped['n_midprice'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        
        for period in [5, 10, 20, 40, 60]:
            vol_values = []
            for (sym, date), group in grouped:
                for i in range(len(group)):
                    if i < period:
                        vol_values.append(0)
                    else:
                        current_idx = group.index[i]
                        before_idx = group.index[i-period]
                        rate = (2 + df.loc[current_idx, 'n_ask1'] + df.loc[current_idx, 'n_bid1']) / (2 + df.loc[before_idx, 'n_ask1'] + df.loc[before_idx, 'n_bid1'] + 1e-10) - 1
                        vol_values.append(rate)
            df[f'volatility_{period}'] = vol_values
        
        ema12 = grouped['n_midprice'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema26 = grouped['n_midprice'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df['macd_dif'] = ema12 - ema26
        df['macd_dea'] = grouped['macd_dif'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df['macd_bar'] = df['macd_dif'] - df['macd_dea']
        
        kdj_k_list = []
        kdj_d_list = []
        kdj_j_list = []
        
        for (sym, date), group in grouped:
            close = group['n_midprice'].values
            high = group['n_ask1'].values
            low = group['n_bid1'].values
            
            low_9 = pd.Series(low).rolling(window=9, min_periods=1).min().values
            high_9 = pd.Series(high).rolling(window=9, min_periods=1).max().values
            
            rsv = 100 * (close - low_9) / (high_9 - low_9 + 1e-10)
            k = pd.Series(rsv).ewm(alpha=1/3, adjust=False).mean().values
            d = pd.Series(k).ewm(alpha=1/3, adjust=False).mean().values
            j = 3 * k - 2 * d
            
            kdj_k_list.extend(k)
            kdj_d_list.extend(d)
            kdj_j_list.extend(j)
        
        df['kdj_k'] = kdj_k_list
        df['kdj_d'] = kdj_d_list
        df['kdj_j'] = kdj_j_list
        
        # print(f"\n=== Feature Quality Check ===")
        # sample_features = ['n_close', 'spread_1', 'volatility_5', 'macd_bar', 'kdj_k', 
        #                   'close_mean', 'bid1_mean', 'vol1_rel_diff', 'amount_normalized']
        # for col in sample_features:
        #     if col in df.columns:
        #         non_zero = (df[col] != 0).sum()
        #         nan_count = df[col].isna().sum()
        #         inf_count = np.isinf(df[col]).sum()
        #         mean_val = df[col].replace([np.inf, -np.inf], np.nan).mean()
        #         print(f"  {col:20s}: non-zero={non_zero:5d}/{len(df):5d} ({100*non_zero/len(df):5.1f}%), "
        #               f"NaN={nan_count:4d}, inf={inf_count:4d}, mean={mean_val:.6f}")
        # print("=============================\n")
        
        return df
    
    def create_features(self, df, window=100):
        df = df.sort_values(['sym', 'date', 'time']).reset_index(drop=True)
        df = self._add_derived_features(df)
        
        available_features = [col for col in self.feature_columns if col in df.columns]
        feature_df = df[available_features].copy()
        
        for col in feature_df.columns:
            feature_df[col] = feature_df[col].replace([np.inf, -np.inf], 0).fillna(0)
        
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
        sample_df = pd.read_csv(sample_file, nrows=100)
        sample_df = self._add_derived_features(sample_df)
        available_features = [col for col in self.feature_columns if col in sample_df.columns]
        
        print(f"\n=== Feature Availability Check ===")
        print(f"Total defined features: {len(self.feature_columns)}")
        print(f"Available features after derivation: {len(available_features)}")
        if len(available_features) < len(self.feature_columns):
            missing_features = set(self.feature_columns) - set(available_features)
            print(f"Missing features ({len(missing_features)}): {list(missing_features)[:10]}")
        print("====================================\n")
        print("Processing training data and saving to binary DMatrix blocks...")
        train_label_counts = self._process_and_save_binary_blocks(train_files, label_name, window, available_features, train_meta, cache_dir, batch_size)
        
        print("Processing validation data and saving to binary DMatrix blocks...")
        self._process_and_save_binary_blocks(val_files, label_name, window, available_features, val_meta, cache_dir, batch_size)
        
        return train_meta, val_meta, train_label_counts
    
    def _split_files(self, test_size):
        pattern = os.path.join(self.data_dir, 'snapshot_sym*_date*_*.csv')
        csv_files = sorted(glob.glob(pattern))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        def _parse(file_path):
            name = os.path.basename(file_path)
            # name: snapshot_symX_dateY_am/pm.csv
            try:
                sym_part = name.split('_')[1]
                date_part = name.split('_')[2] 
                session_part = name.split('_')[3].split('.')[0]
                sym_idx = int(sym_part.replace('sym', '').replace('sym', ''))
                date_idx = int(date_part.replace('date', ''))
                session_idx = 0 if 'am' in session_part else 1
                return (date_idx, session_idx, sym_idx)
            except Exception:
                return (0, 0, 0)
        
        csv_files = sorted(csv_files, key=_parse)
        
        split_file_idx = int(len(csv_files) * (1 - test_size))
        train_files = csv_files[:split_file_idx]
        val_files = csv_files[split_file_idx:]
        return train_files, val_files

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
        label_counts = {0: 0, 1: 0, 2: 0}
        buffer_files = []
        buffer_idx = 0
        total_samples = 0
        
        base_name = os.path.splitext(os.path.basename(meta_path))[0]
        
        def save_buffer():
            nonlocal buffer_idx, batch_features, batch_labels, total_samples
            if not batch_features:
                return 0
            X_batch = np.array(batch_features)
            y_batch = np.array(batch_labels)
            buffer_file = os.path.join(cache_dir, f'{base_name}_{buffer_idx}.buffer')
            dtrain_batch = xgb.DMatrix(X_batch, label=y_batch)
            dtrain_batch.save_binary(buffer_file)
            num_samples = len(batch_labels)
            buffer_files.append(buffer_file)
            batch_features = []
            batch_labels = []
            buffer_idx += 1
            return num_samples
        
        for file in tqdm(csv_files, desc="Processing files", ncols=80):
            df = pd.read_csv(file)
            df = df.sort_values(['sym', 'date', 'time']).reset_index(drop=True)
            df = self._add_derived_features(df)
            
            feature_df = df[available_features].copy()
            for col in feature_df.columns:
                feature_df[col] = feature_df[col].replace([np.inf, -np.inf], 0).fillna(0)
            
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
                    label = int(group_labels[i])
                    batch_labels.append(label)
                    label_counts[label] = label_counts.get(label, 0) + 1
                    
                    if len(batch_features) >= batch_size:
                        total_samples += save_buffer()
            
            del df, feature_df
        
        total_samples += save_buffer()
        
        if not buffer_files:
            raise ValueError("No data processed")
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            for buffer_file in buffer_files:
                f.write(f'{buffer_file}\n')
        
        print(f"Saved {len(buffer_files)} binary DMatrix blocks to {meta_path} ({total_samples} total samples)")
        
        return label_counts
    
    def create_features_batch_generator(self, df, label_name, window=100, batch_size=10000):
        """
        分批生成特征和标签，避免内存溢出
        """
        df = df.sort_values(['sym', 'date', 'time']).reset_index(drop=True)
        df = self._add_derived_features(df)
        available_features = [col for col in self.feature_columns if col in df.columns]
        feature_df = df[available_features].copy()
        
        for col in feature_df.columns:
            feature_df[col] = feature_df[col].replace([np.inf, -np.inf], 0).fillna(0)
        
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
            df = self._add_derived_features(df)
            
            feature_df = df[available_features].copy()
            for col in feature_df.columns:
                feature_df[col] = feature_df[col].replace([np.inf, -np.inf], 0).fillna(0)
            
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
    
    dtrain_split, dtest_split = processor.get_train_test_split(args.label)
    print(f"Train set: {dtrain_split.num_row()} samples")
    print(f"Test set: {dtest_split.num_row()} samples")
    
    print("Data processing test completed successfully!")


if __name__ == '__main__':
    main()

