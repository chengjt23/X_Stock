import pickle

import numpy as np
import xgboost as xgb


class XGBoostModel:
    def __init__(self, **kwargs):
        self.model = None 
        self.params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_boost_round': 100,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'nthread': 8,
            'tree_method': 'hist'
        }
        self.params.update(kwargs)
    
    def fit(self, dtrain, dval=None, num_boost_round=None, early_stopping_rounds=10):
        params = self.params.copy()
        if num_boost_round is None:
            num_boost_round = params.pop('num_boost_round', 100)
        else:
            params.pop('num_boost_round', None)
        
        if dval is not None:
            # 使用early stopping防止过拟合
            self.model = xgb.train(
                params, dtrain, 
                num_boost_round=num_boost_round, 
                evals=[(dval, 'eval')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=20
            )
        else:
            self.model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        return self
    
    def fit_incremental(self, train_gen, val_gen=None, rounds_per_batch=5, early_stopping_rounds=10):
        """
        增量训练：分批加载数据，逐步更新模型
        train_gen: 生成器，每次yield (X, y) 或 (X, y, weights)
        """
        params = self.params.copy()
        params.pop('num_boost_round', None)
        
        self.model = None

        def _batch_to_dmatrix(batch):
            if len(batch) == 3:
                X, y, weights = batch
                return xgb.DMatrix(X, label=y, weight=weights)
            X, y = batch
            return xgb.DMatrix(X, label=y)

        def _build_validation_matrix(generator):
            if generator is None:
                return None
            val_features = []
            val_labels = []
            for payload in generator:
                X_val, y_val = payload[:2]
                val_features.append(X_val)
                val_labels.append(y_val)
            if not val_features:
                return None
            X_stack = np.vstack(val_features)
            y_stack = np.concatenate(val_labels)
            dmatrix = xgb.DMatrix(X_stack, label=y_stack)
            del val_features, val_labels, X_stack, y_stack
            return dmatrix
        
        try:
            first_batch = next(train_gen)
            dtrain = _batch_to_dmatrix(first_batch)
            dval = _build_validation_matrix(val_gen)
            
            if dval is not None:
                self.model = xgb.train(
                    params, dtrain,
                    num_boost_round=rounds_per_batch,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=20
                )
            else:
                self.model = xgb.train(params, dtrain, num_boost_round=rounds_per_batch)
            
            batch_count = 1
            for batch_data in train_gen:
                dtrain_batch = _batch_to_dmatrix(batch_data)
                
                if dval is not None:
                    self.model = xgb.train(
                        params, dtrain_batch,
                        num_boost_round=rounds_per_batch,
                        xgb_model=self.model,
                        evals=[(dval, 'eval')],
                        verbose_eval=False
                    )
                else:
                    self.model = xgb.train(
                        params, dtrain_batch,
                        num_boost_round=rounds_per_batch,
                        xgb_model=self.model
                    )
                batch_count += 1
            
            return self
        except StopIteration:
            raise ValueError("Training generator is empty")
    
    def predict(self, dtest):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(dtest)
    
    def predict_proba(self, dtest):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        pred = self.model.predict(dtest)
        return pred.reshape(-1, 3)
    
    def save_model(self, path):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        # 使用pickle保存整个模型对象为.pth格式
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, path):
        # 从.pth文件加载模型
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        return self

