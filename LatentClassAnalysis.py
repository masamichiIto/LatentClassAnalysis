import numpy as np

class LatentClassAnalysis:
    """
    n_classes: number of latent classes
    n_iteration: maximum limit for EM iteration
    tol: parameter cotrolling precesion of converge
    random_state: seed for random generator
    responsibilities: Z = (z_{ik}) := P(k|x_i) = P(x_i|k)P(k)/(\sum_g P(x_i|g)P(g))
    class_probs: \pi = (\pi_1, ..., \pi_K)
    cond_probs: P = (p_{jk}) = P(x_i|k) = p_{jk}^{x_ij}(1-p_{jk})^{1-x_{ij}}
    """
    def __init__(self, n_classes, n_iterations=100, tol=1e-4, random_state=None):
        self.n_classes = n_classes
        self.n_iterations = n_iterations
        self.tol = tol
        self.random_state = random_state

    def _calc_loglike(self, data):
        ll = 0
        for i in range(len(data)):
            ll2 = 0
            for k in range(self.n_classes):
                ll2 += self.class_probs[k] * self._calc_bernoulli(data[i,:], k)
            ll += np.log(ll2)
        return ll
    
    def _calc_comp_ll(self, data, responsibilities):
        

    
    def _calc_bernoulli(self, data_i, k):
        temp_cond_prob = self.cond_probs[:,k]
        temp_cond_prob[data_i == 0] = 1 - temp_cond_prob[data_i == 0]
        return np.prod(temp_cond_prob)

    def _e_step(self, data):
        responsibilities = np.zeros((len(data), self.n_classes))
        print(responsibilities.shape)
        
        for i in range(len(data)):
            for k in range(self.n_classes):
                responsibilities[i, k] = self.class_probs[k] * self._calc_bernoulli(data[i,:], k)
                
            responsibilities[i, :] /= np.sum(responsibilities[i, :])
        
        return responsibilities
    
    def _m_step(self, data, responsibilities):
        for k in range(self.n_classes):
            self.class_probs[k] = np.mean(responsibilities[:, k])
            
            for j in range(self.n_categories):
                self.cond_probs[j][k] = np.sum(responsibilities[:, k] * (data[:,j])) / np.sum(responsibilities[:, k])
    
    def fit(self, data):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.n_samples, self.n_categories = data.shape
        self.class_probs = np.ones(self.n_classes) / self.n_classes
        self.cond_probs = np.random.rand(self.n_categories, self.n_classes)
        ll = -np.Inf
        
        for iteration in range(self.n_iterations):
            old_cond_probs = np.copy(self.cond_probs)
            
            responsibilities = self._e_step(data)
            self._m_step(data, responsibilities)
            print("{0} : {1}".format(iteration,self._calc_loglike(data)))
            if np.max(np.abs(self.cond_probs - old_cond_probs)) < self.tol:
                print(f"Converged after {iteration + 1} iterations.")
                break
    
    def get_class_probabilities(self):
        return self.class_probs
    
    def get_conditional_probabilities(self):
        return self.cond_probs
    
    def calc_posterior_class(self, data):
        return self._e_step(data)

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

