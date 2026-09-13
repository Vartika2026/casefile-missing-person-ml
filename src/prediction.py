from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


class MissingPersonLocationModel:
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f'Unsupported model type: {self.model_type}')

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds, average='weighted', zero_division=0),
            'recall': recall_score(y_test, preds, average='weighted', zero_division=0),
            'f1': f1_score(y_test, preds, average='weighted', zero_division=0),
        }
        return metrics

    def predict_probabilities(self, X: pd.DataFrame):
        if self.model is None:
            raise ValueError('Model has not been trained yet.')
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame):
        if self.model is None:
            raise ValueError('Model has not been trained yet.')
        return self.model.predict(X)


def build_probability_ranking(prediction_probs: np.ndarray, class_labels: list[str]) -> pd.DataFrame:
    """Return ranking dataframe with probabilities and priority."""
    chosen_labels = list(class_labels)
    ranking = pd.DataFrame(prediction_probs, columns=chosen_labels)
    ranking['top_label'] = ranking.idxmax(axis=1)
    ranking['top_probability'] = ranking.max(axis=1)
    ranking['rank'] = ranking['top_probability'].rank(method='first', ascending=False)

    ranked_data = ranking[['top_label', 'top_probability']].sort_values('top_probability', ascending=False)
    ranked_data = ranked_data.reset_index(drop=True)
    ranked_data.index += 1
    ranked_data.columns = ['predicted_area', 'probability']
    ranked_data['probability'] = ranked_data['probability'] * 100
    ranked_data['probability'] = ranked_data['probability'].round(2)
    ranked_data['priority'] = ranked_data['probability'].apply(
        lambda x: 'Very High' if x >= 80 else 'High' if x >= 60 else 'Medium' if x >= 30 else 'Low'
    )
    return ranked_data
