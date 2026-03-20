import numpy as np

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, steps=500):
    import numpy as np
    
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1, 1)
    
    N, d = X.shape
    
    w = np.zeros((d, 1))
    b = 0.0
    
    for step in range(steps):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        
        epsilon = 1e-8
        loss = -np.mean(
            y * np.log(p + epsilon) + 
            (1 - y) * np.log(1 - p + epsilon)
        )
        
        dw = (1/N) * (X.T @ (p - y))
        db = (1/N) * np.sum(p - y)
        
        w -= lr * dw
        b -= lr * db
    
    # 🔥 FIX HERE
    return w.flatten().tolist(), float(b)