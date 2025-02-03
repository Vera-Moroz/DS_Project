# Modules/feature_engineering.py
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureEngineering:
    def select_k_best_features(self, data: pd.DataFrame, target: pd.Series, k=10):
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(data, target)
        selected_features = selector.get_support(indices=True)
        return data.iloc[:, selected_features]
