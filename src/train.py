import pandas as pd
import os
from src.preprocessing import DataPreprocessor
from src.model import MalwareDetector

def train_pipeline():
    # 1. Load Data
    print("📂 Loading UNSW-NB15 Dataset...")
    train_df = pd.read_csv('data/UNSW_NB15_training-set.csv')
    test_df = pd.read_csv('data/UNSW_NB15_testing-set.csv')

    # 2. Preprocessing
    print("⚙️  Preprocessing Data...")
    preprocessor = DataPreprocessor()
    
    # Fit on training data
    preprocessor.fit(train_df)
    
    # Transform both sets
    X_train = preprocessor.transform(train_df.drop(columns=['label', 'attack_cat']))
    y_train = preprocessor.transform_labels(train_df['attack_cat'])
    
    X_test = preprocessor.transform(test_df.drop(columns=['label', 'attack_cat']))
    y_test = preprocessor.transform_labels(test_df['attack_cat'])
    
    # Save preprocessor for later use in scanning
    if not os.path.exists('artifacts'): os.makedirs('artifacts')
    preprocessor.save('artifacts/preprocessor.pkl')

    # 3. Model Training
    detector = MalwareDetector()
    detector.train(X_train, y_train)
    
    # 4. Evaluation
    class_names = preprocessor.label_encoder.classes_
    detector.evaluate(X_test, y_test, class_names)
    
    # 5. Save Model
    detector.save('artifacts/model.pkl')
    print("💾 Model saved to artifacts/model.json")

if __name__ == "__main__":
    train_pipeline()