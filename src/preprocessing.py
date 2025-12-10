import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

class DataPreprocessor:
    def __init__(self):
        self.encoders = {}
        self.scaler = MinMaxScaler()
        # Define categorical and numerical columns based on UNSW-NB15 schema
        self.cat_cols = ['proto', 'service', 'state']
        self.drop_cols = ['id'] # Irrelevant for pattern detection
        self.label_encoder = LabelEncoder()
        
    def fit(self, df):
        """Learn valid patterns from training data."""
        # Fit Label Encoders for categorical features
        for col in self.cat_cols:
            le = LabelEncoder()
            # Handle unknown categories in future by using a 'unknown' token approach or simple fit
            le.fit(df[col].astype(str))
            self.encoders[col] = le
            
        # Fit Scaler for numerical data
        num_cols = df.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c not in self.drop_cols + ['label']]
        self.scaler.fit(df[num_cols])
        self.num_cols = num_cols

        # Fit Target Label Encoder (e.g., 'Normal' -> 0, 'DoS' -> 1)
        if 'attack_cat' in df.columns:
            self.label_encoder.fit(df['attack_cat'])

    def transform(self, df):
        """Transform data for the model."""
        df = df.copy()
        
        # Drop ID
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        # Encode Categoricals
        for col in self.cat_cols:
            le = self.encoders[col]
            # Map unseen labels to a default or handle gracefully
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0]) 
            df[col] = le.transform(df[col].astype(str))
            
        # Scale Numerics
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        
        return df

    def transform_labels(self, y):
        return self.label_encoder.transform(y)
    
    def save(self, path="artifacts/preprocessor.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="artifacts/preprocessor.pkl"):
        return joblib.load(path)