import time
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor
from src.model import MalwareDetector
from colorama import Fore, Style, init

init(autoreset=True)

class NetworkScanner:
    def __init__(self):
        self.preprocessor = DataPreprocessor.load("artifacts/preprocessor.pkl")
        self.model = MalwareDetector()
        self.model.load("artifacts/model.pkl")
        self.classes = self.preprocessor.label_encoder.classes_

    def mock_feature_extraction(self, raw_packet):
        """
        In a real production system, this converts raw bytes to the 40+ features 
        UNSW-NB15 uses (dur, sbytes, dbytes, sttl, etc.).
        Here, we mock it by sampling from the test set for demonstration.
        """
        # Load a random row from test set to simulate a 'processed' packet
        # In real usage, you would calculate these metrics from the `raw_packet` object
        mock_data = pd.read_csv('data/UNSW_NB15_testing-set.csv').sample(1)
        true_label = mock_data['attack_cat'].values[0]
        features = mock_data.drop(columns=['id', 'label', 'attack_cat'])
        return features, true_label

    def scan(self):
        print(f"{Fore.CYAN}📡 Sentinel AI Scanner Active - Monitoring Network Traffic...{Style.RESET_ALL}")
        print("---------------------------------------------------------------")
        
        try:
            while True:
                # 1. Simulate capturing a packet
                time.sleep(1) # Scan every second
                
                # 2. Extract features
                features_df, true_label = self.mock_feature_extraction(None)
                
                # 3. Preprocess features
                processed_features = self.preprocessor.transform(features_df)
                
                # 4. AI Prediction
                pred_idx = self.model.predict(processed_features)[0]
                pred_label = self.classes[pred_idx]
                confidence = np.max(self.model.predict_proba(processed_features))

                # 5. Alerting
                self.alert(pred_label, confidence, true_label)
                
        except KeyboardInterrupt:
            print("\n🔴 Scanner stopped.")

    def alert(self, pred_label, confidence, true_label):
        timestamp = time.strftime("%H:%M:%S")
        
        if pred_label == "Normal":
            print(f"[{timestamp}] {Fore.GREEN}✓ Traffic Clean{Style.RESET_ALL} (Conf: {confidence:.2f})")
        else:
            print(f"[{timestamp}] {Fore.RED}⚠ MALWARE DETECTED: {pred_label.upper()}{Style.RESET_ALL}")
            print(f"            Confidence: {confidence:.2f} | Validated against: {true_label}")

if __name__ == "__main__":
    scanner = NetworkScanner()
    scanner.scan()