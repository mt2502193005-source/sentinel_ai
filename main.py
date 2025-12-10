import sys
from src.train import train_pipeline
from src.scanner import NetworkScanner

if __name__ == "__main__":
    print("Sentinel AI - Windows Network Traffic Malware Detector")
    print("1. Train Model")
    print("2. Start Scanner")
    
    choice = input("Select mode (1/2): ")
    
    if choice == '1':
        train_pipeline()
    elif choice == '2':
        try:
            scanner = NetworkScanner()
            scanner.scan()
        except FileNotFoundError:
            print("❌ Error: Model not found. Please run 'Train Model' first.")
    else:
        print("Invalid choice.")