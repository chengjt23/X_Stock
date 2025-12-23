import argparse
import os
import sys
import numpy as np
import xgboost as xgb
import joblib
import scipy.sparse as sp
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.data_process import DataProcessor

def tprint(*args, **kwargs):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}]', *args, **kwargs)

def evaluate_model(model_path, meta_path, data_dir, eval_threshold=0.88):
    tprint(f"Starting evaluation...")
    tprint(f"Model: {model_path}")
    tprint(f"Final Report Threshold: {eval_threshold}")

    # 1. 加载模型
    try:
        loaded_obj = joblib.load(model_path)
        bst = loaded_obj.model if hasattr(loaded_obj, 'model') else loaded_obj
        tprint("Model loaded successfully.")
    except Exception as e:
        tprint(f"Error loading model: {e}"); return

    # 2. 加载数据与同步的价格差文件
    if not os.path.exists(meta_path):
        tprint(f"Error: Meta file {meta_path} not found."); return
        
    with open(meta_path, 'r') as f:
        buffer_files = [line.strip() for line in f if line.strip()]
    
    tprint(f"Loading {len(buffer_files)} blocks and fetching real price deltas...")
    
    xs, ys, price_deltas = [], [], []
    for bf in buffer_files:
        dm = xgb.DMatrix(bf)
        xs.append(dm.get_data())
        ys.append(dm.get_label())
        
        # 加载 DataProcessor 生成的 .price.npy 文件
        pf = bf.replace('.buffer', '.price.npy')
        if os.path.exists(pf):
            price_deltas.append(np.load(pf))
        else:
            tprint(f"Warning: Price file {pf} missing! Using zeros.")
            price_deltas.append(np.zeros(int(dm.num_row())))
    
    X_all = sp.vstack(xs) if sp.issparse(xs[0]) else np.vstack(xs)
    y_true = np.concatenate(ys).astype(int)
    p_diff_raw = np.concatenate(price_deltas) # P(t+n) - P(t)
    dtest = xgb.DMatrix(X_all, label=y_true)
    
    tprint(f"Total samples for testing: {len(y_true)}")

    # 3. 执行预测
    pred_proba = bst.predict(dtest) 

    # 4. 置信度阈值扫描
    tprint("\n" + "="*135)
    tprint(f"{'Threshold':<8} | {'Signal %':<9} | {'Trades':<8} | {'Prec':<8} | {'Recall':<8} | {'Total PnL':<14} | {'Avg PnL':<12} | {'F0.5'}")
    tprint("-" * 135)

    actual_moves_mask = (y_true == 0) | (y_true == 2)
    total_actual_moves = np.sum(actual_moves_mask)
    thresholds = [0.0, 0.35, 0.40, 0.50, 0.60,0.65, 0.70, 0.75, 0.80,0.81, 0.82, 0.83,0.84, 0.85, 0.88,0.90, 0.95]
    
    for thr in thresholds:
        mask_down = pred_proba[:, 0] > thr
        mask_up = pred_proba[:, 2] > thr
        
        if thr == 0:
            final_pred_tmp = pred_proba.argmax(axis=1)
            mask_signal = (final_pred_tmp == 0) | (final_pred_tmp == 2)
        else:
            mask_signal = mask_down | mask_up
            
        num_signals = np.sum(mask_signal)
        
        if num_signals > 0:
            signal_pct = 100 * num_signals / len(y_true)
            preds_in_mask = pred_proba[mask_signal].argmax(axis=1)
            labels_in_mask = y_true[mask_signal]
            diffs_in_mask = p_diff_raw[mask_signal]
            
            # 精准率
            correct = np.sum(preds_in_mask == labels_in_mask)
            precision = correct / num_signals
            recall = correct / (total_actual_moves + 1e-10)
            
            # --- 收益率计算逻辑 ---
            # 预测上涨(2): 收益 = P(t+n) - P(t) [即 diffs_in_mask]
            # 预测下跌(0): 收益 = P(t) - P(t+n) [即 -diffs_in_mask]
            trade_pnls = np.where(preds_in_mask == 2, diffs_in_mask, -diffs_in_mask)
            total_pnl = np.sum(trade_pnls)
            avg_pnl = total_pnl / num_signals
            
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall + 1e-10)
            
            tprint(f"{thr:<8.2f} | {signal_pct:<8.2f}% | {num_signals:<8} | {precision:<8.4f} | {recall:<8.4f} | {total_pnl:<14.6f} | {avg_pnl:<12.6f} | {f05:.4f}")
        else:
            tprint(f"{thr:<8.2f} | 0.00%     | 0        | N/A      | N/A      | 0.000000       | 0.000000     | 0.0000")

    tprint("="*135)

    # 5. 指定阈值下的最终报告
    tprint(f"\nFinal Summary Report at Threshold {eval_threshold}:")
    y_pred_final = np.ones(len(y_true))
    y_pred_final[pred_proba[:, 0] > eval_threshold] = 0
    y_pred_final[pred_proba[:, 2] > eval_threshold] = 2
    
    print(classification_report(y_true, y_pred_final, target_names=['Down', 'Unchanged', 'Up'], digits=4))
    
    final_mask = (y_pred_final == 0) | (y_pred_final == 2)
    if np.sum(final_mask) > 0:
        f_preds = y_pred_final[final_mask]
        f_diffs = p_diff_raw[final_mask]
        f_pnls = np.where(f_preds == 2, f_diffs, -f_diffs)
        tprint(f"Results for Threshold {eval_threshold}:")
        tprint(f"  - Total PnL: {np.sum(f_pnls):.6f}")
        tprint(f"  - Avg PnL:   {np.mean(f_pnls):.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='/mnt/data/cache')
    parser.add_argument('--label', type=str, default='label_5')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.88)
    args = parser.parse_args()
    
    val_meta_path = os.path.join(args.cache_dir, f'val_{args.label}.meta')
    evaluate_model(args.model_path, val_meta_path, args.data_dir, eval_threshold=args.threshold)