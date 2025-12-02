# Project 3 - PCB Detection
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("\n" + "="*50)
print("PCB Component Detection")
print("="*50)
print("1. Train Model")
print("2. Evaluate Model")
print("3. Run Predictions")
print("4. Extract Motherboard")
print("5. Run Everything")
print("6. Exit")

choice = input("\nChoice: ")

try:
    if choice == '1':
        import train_model
    elif choice == '2':
        import evaluate_model
    elif choice == '3':
        import predict
    elif choice == '4':
        import pcb_pipeline
    elif choice == '5':
        print("\nTraining...")
        import train_model
        print("\nEvaluating...")
        import evaluate_model
        print("\nPredicting...")
        import predict
        print("\nDone!")
    elif choice == '6':
        print("Bye!")
    else:
        print("Pick 1-6")
except Exception as e:
    print(f"Error: {e}")
