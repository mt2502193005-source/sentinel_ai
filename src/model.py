import xgboost as xgb
import joblib  # Import joblib
from sklearn.metrics import classification_report, accuracy_score

class MalwareDetector:
    def __init__(self):
        # XGBoost parameters optimized for multi-class classification
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False
        )

    def train(self, X_train, y_train):
        print("⚡ Training Sentinel AI Model (XGBoost)...")
        self.model.fit(X_train, y_train)
        print("✅ Training Complete.")

    def evaluate(self, X_test, y_test, label_names):
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n🎯 Model Accuracy: {acc:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, preds, target_names=label_names))

    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    # --- CHANGED SECTIONS BELOW ---
    
    def save(self, path="artifacts/model.pkl"):
        # We use joblib instead of save_model to avoid the "estimator_type" error
        joblib.dump(self.model, path)

    def load(self, path="artifacts/model.pkl"):
        self.model = joblib.load(path)