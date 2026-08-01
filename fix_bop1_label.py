# fix_bop1_label.py — run once from your color-histogram-analysis folder
import joblib, torch, numpy as np

# XGBoost bundle
b = joblib.load('results/models/traditional_xgboost.joblib')
classes = list(b['label_encoder'].classes_)
classes[classes.index('bop')] = 'bop1'
b['label_encoder'].classes_ = np.array(classes)
joblib.dump(b, 'results/models/traditional_xgboost.joblib')
print('xgboost bundle fixed:', classes)

# CNN checkpoints
for name in ['resnet18', 'mobilenet_v3_small']:
    path = f'results/models/cnn_{name}_best.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    ck['classes'] = [('bop1' if c == 'bop' else c) for c in ck['classes']]
    torch.save(ck, path)
    print(name, 'fixed:', ck['classes'])