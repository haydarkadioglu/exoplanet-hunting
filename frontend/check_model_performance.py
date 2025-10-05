import joblib
import json
import os

def check_model_performance():
    """Check and extract model performance metrics from joblib files"""
    
    results = {}
    model_dirs = ['models/k2', 'models/kepler', 'models/tess']
    
    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue
            
        model_name = model_dir.split('/')[-1]
        model_info = {}
        
        # Check for model results file
        results_file = None
        for file in os.listdir(model_dir):
            if 'model_results' in file and file.endswith('.joblib'):
                results_file = os.path.join(model_dir, file)
                break
        
        if results_file and os.path.exists(results_file):
            try:
                results_data = joblib.load(results_file)
                print(f"\n=== {model_name.upper()} Model Performance ===")
                
                if isinstance(results_data, dict):
                    # Find the best performing model
                    best_model = None
                    best_f1 = 0
                    
                    for name, metrics in results_data.items():
                        if isinstance(metrics, dict) and 'f1_score' in metrics:
                            if metrics['f1_score'] > best_f1:
                                best_f1 = metrics['f1_score']
                                best_model = name
                                model_info = {
                                    'model_name': name,
                                    'accuracy': metrics.get('accuracy', 0),
                                    'precision': metrics.get('precision', 0),
                                    'recall': metrics.get('recall', 0),
                                    'f1_score': metrics.get('f1_score', 0),
                                    'roc_auc': metrics.get('roc_auc', 0),
                                    'cv_mean': metrics.get('cv_mean', 0),
                                    'cv_std': metrics.get('cv_std', 0)
                                }
                    
                    print(f"Best Model: {best_model}")
                    print(f"Accuracy: {model_info['accuracy']:.4f} ({model_info['accuracy']*100:.2f}%)")
                    print(f"Precision: {model_info['precision']:.4f} ({model_info['precision']*100:.2f}%)")
                    print(f"Recall: {model_info['recall']:.4f} ({model_info['recall']*100:.2f}%)")
                    print(f"F1-Score: {model_info['f1_score']:.4f} ({model_info['f1_score']*100:.2f}%)")
                    print(f"ROC AUC: {model_info['roc_auc']:.4f} ({model_info['roc_auc']*100:.2f}%)")
                    print(f"CV Mean: {model_info['cv_mean']:.4f} ± {model_info['cv_std']:.4f}")
                    
                    results[model_name] = model_info
                    
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
        
        # Check other files
        files_in_dir = os.listdir(model_dir)
        print(f"\nFiles in {model_dir}:")
        for file in sorted(files_in_dir):
            print(f"  - {file}")
    
    # Save summary to JSON
    with open('model_performance_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Summary saved to model_performance_summary.json ===")
    return results

if __name__ == "__main__":
    check_model_performance()