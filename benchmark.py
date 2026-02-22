import numpy as np
import time

def viterbi_orig(n_samples, n_states, lambda_penalty):
    X_arr = np.random.randn(n_samples, 5)
    means = np.random.randn(n_states, 5)
    
    distances = np.zeros((n_samples, n_states))
    for k in range(n_states):
        distances[:, k] = 0.5 * np.sum((X_arr - means[k])**2, axis=1)
        
    cost_matrix = np.zeros((n_samples, n_states))
    back_pointers = np.zeros((n_samples, n_states), dtype=int)
    cost_matrix[0] = distances[0]
    
    t0 = time.time()
    for t in range(1, n_samples):
        for k in range(n_states):
            trans_costs = cost_matrix[t-1] + lambda_penalty * (np.arange(n_states) != k)
            best_prev_state = np.argmin(trans_costs)
            cost_matrix[t, k] = trans_costs[best_prev_state] + distances[t, k]
            back_pointers[t, k] = best_prev_state
    t1 = time.time()
    return t1 - t0

def viterbi_matrix(n_samples, n_states, lambda_penalty):
    X_arr = np.random.randn(n_samples, 5)
    means = np.random.randn(n_states, 5)
    
    distances = 0.5 * np.sum((X_arr[:, None, :] - means[None, :, :])**2, axis=2)
        
    cost_matrix = np.zeros((n_samples, n_states))
    back_pointers = np.zeros((n_samples, n_states), dtype=int)
    cost_matrix[0] = distances[0]
    
    t0 = time.time()
    penalty_matrix = lambda_penalty * (1 - np.eye(n_states))
    
    for t in range(1, n_samples):
        trans_costs = cost_matrix[t-1, :, None] + penalty_matrix
        best_prev_states = np.argmin(trans_costs, axis=0)
        cost_matrix[t] = trans_costs[best_prev_states, np.arange(n_states)] + distances[t]
        back_pointers[t] = best_prev_states
        
    t1 = time.time()
    return t1 - t0

def viterbi_scalar(n_samples, n_states, lambda_penalty):
    X_arr = np.random.randn(n_samples, 5)
    means = np.random.randn(n_states, 5)
    
    distances = np.zeros((n_samples, n_states))
    for k in range(n_states):
        distances[:, k] = 0.5 * np.sum((X_arr - means[k])**2, axis=1)
        
    cost_matrix = np.zeros((n_samples, n_states))
    back_pointers = np.zeros((n_samples, n_states), dtype=int)
    cost_matrix[0] = distances[0]
    
    t0 = time.time()
    
    for t in range(1, n_samples):
        # specifically optimized for 2 states
        c00 = cost_matrix[t-1, 0]
        c10 = cost_matrix[t-1, 1] + lambda_penalty
        if c00 < c10:
            best_prev_0 = 0
            cost_0 = c00
        else:
            best_prev_0 = 1
            cost_0 = c10
            
        c01 = cost_matrix[t-1, 0] + lambda_penalty
        c11 = cost_matrix[t-1, 1]
        if c01 < c11:
            best_prev_1 = 0
            cost_1 = c01
        else:
            best_prev_1 = 1
            cost_1 = c11
            
        cost_matrix[t, 0] = cost_0 + distances[t, 0]
        back_pointers[t, 0] = best_prev_0
        cost_matrix[t, 1] = cost_1 + distances[t, 1]
        back_pointers[t, 1] = best_prev_1

    t1 = time.time()
    return t1 - t0

n = 2772
print(f"Orig: {viterbi_orig(n, 2, 10.0)*1000:.2f} ms")
print(f"Matrix: {viterbi_matrix(n, 2, 10.0)*1000:.2f} ms")
print(f"Scalar (2-state): {viterbi_scalar(n, 2, 10.0)*1000:.2f} ms")

