import numpy as np

class LatentClassAnalysis:
    def __init__(self, n_classes, n_iterations=100, tol=1e-4):
        self.n_classes = n_classes
        self.n_iterations = n_iterations
        self.tol = tol
        
    def _e_step(self, data):
        responsibilities = np.zeros((len(data), self.n_classes))
        print(responsibilities.shape)
        
        for i in range(len(data)):
            for k in range(self.n_classes):
                responsibilities[i, k] = self.class_probs[k] * np.prod(self.cond_probs[k][data[i]])
                
            responsibilities[i, :] /= np.sum(responsibilities[i, :])
        
        return responsibilities
    
    def _m_step(self, data, responsibilities):
        for k in range(self.n_classes):
            self.class_probs[k] = np.mean(responsibilities[:, k])
            
            for j in range(self.n_categories):
                self.cond_probs[k][j] = np.sum(responsibilities[:, k] * (data == j)) / np.sum(responsibilities[:, k])
    
    def fit(self, data):
        self.n_samples, self.n_categories = data.shape
        self.class_probs = np.ones(self.n_classes) / self.n_classes
        self.cond_probs = np.random.rand(self.n_classes, self.n_categories)
        
        for iteration in range(self.n_iterations):
            old_cond_probs = np.copy(self.cond_probs)
            
            responsibilities = self._e_step(data)
            self._m_step(data, responsibilities)
            
            if np.max(np.abs(self.cond_probs - old_cond_probs)) < self.tol:
                print(f"Converged after {iteration + 1} iterations.")
                break
    
    def get_class_probabilities(self):
        return self.class_probs
    
    def get_conditional_probabilities(self):
        return self.cond_probs

# Example usage
data = np.array([[0, 1, 0, 1],
                 [1, 0, 1, 0],
                 [0, 1, 1, 0],
                 [1, 0, 0, 1],
                 [1, 1, 1, 0],
                 [0, 0, 1, 1]])
"""
data = np.array([[0, 1, 0, 1],
                 [1, 0, 1, 0],
                 [0, 1, 1, 0],
                 [1, 0, 0, 1]])
"""

n_classes = 2
lca = LatentClassAnalysis(n_classes=n_classes)
lca.fit(data)

class_probs = lca.get_class_probabilities()
cond_probs = lca.get_conditional_probabilities()

print("Estimated Class Probabilities:", class_probs)
print("Estimated Conditional Probabilities:")
for k in range(n_classes):
    print(f"Class {k + 1}:", cond_probs[k])
