import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
import scipy.sparse as sp
from numpy import linalg as LA
from math import sqrt

class LipSDP:
    def __init__(self, weights, pair):
        self.weight_mats = weights
        self.layers=len(weights)
        self.weight_dims = [0] * self.layers
        for i in range(self.layers):
            self.weight_dims[i] = self.weight_mats[i].shape[1]
        self.construct_mat()
        self.pair = pair

    def solve_sdp(self, verbose):
        if len(self.pair) == 2:
           final_weight = self.weight_mats[-1][self.pair[0],:] - self.weight_mats[-1][self.pair[1],:]
        else:
           final_weight = self.weight_mats[-1][self.pair[0],:]
        final_weight = np.expand_dims(final_weight, axis=0)
        L_sq = cp.Variable((1,1), nonneg=True)
        D = cp.Variable((1, self.n_hidden_vars), nonneg=True)
        T = cp.diag(D)
        Q = cp.bmat([[0*T, T],[T, -2*T]])
        const_matrix = self.A_on_B.transpose() @ Q @ self.A_on_B
        #Another Matrix
        M = np.zeros([self.constraint_size,self.constraint_size], dtype=float)
        M[self.constraint_size-self.weight_dims[-1]:, self.constraint_size-self.weight_dims[-1]:] = -final_weight.transpose()*final_weight
        
        sparse_vars = cp.vstack([L_sq]*self.weight_dims[0])
        positions = []
        for i in range(self.weight_dims[0]):
            positions.append([i, i])
        V = np.ones(self.weight_dims[0])
        I = []
        J = []
        for idx, (row, col) in enumerate(positions):
            I.append(row + col*self.constraint_size)
            J.append(idx)
        reshape_mat = sp.coo_matrix((V, (I,J)), shape=(self.constraint_size*self.constraint_size, self.weight_dims[0]))
        N = cp.reshape(reshape_mat @ sparse_vars, (self.constraint_size,self.constraint_size))
        #The CVX optimization program
        prob = cp.Problem(cp.Minimize(L_sq), [const_matrix - M - N<< 0])
        #Verbose: False if not want to print out the progress from the solver
        prob.solve(solver=cp.MOSEK, verbose=verbose)
        return sqrt(prob.value)
        

    def construct_mat(self):
        #n_hidden_vars is the number of hidden nodes
        self.n_hidden_vars = sum(self.weight_dims[1:])
        #print(self.n_hidden_vars)
        #Constructing some auxilary matrices for the SDP program
        weights = block_diag(*self.weight_mats[:-1])
        zeros_col = np.zeros((weights.shape[0], self.weight_dims[-1]))
        A = np.concatenate((weights, zeros_col), axis=1)
        eyes = np.identity(A.shape[0])
        init_col = np.zeros((eyes.shape[0], self.weight_dims[0]))
        B = np.concatenate((init_col, eyes), axis=1)
        A_on_B = np.concatenate((A, B), axis = 0)
        self.constraint_size = A_on_B.shape[1]
        self.A_on_B = A_on_B

class ReduntSDP:
    def __init__(self, weights, pair):
        self.weight_mats = weights
        self.layers=len(weights)
        self.weight_dims = [0] * self.layers
        for i in range(self.layers):
            self.weight_dims[i] = self.weight_mats[i].shape[1]
        #nClasses is the number of classification classes, e.g., nClasses=10 for CIFAR10
        self.l2_norm()
        self.construct_mat()
        self.pair = pair

    def l2_norm(self):
        self.weight_norms = []
        self.layer_matrix_norms = []
        for i in range(self.layers-1):
          weight = self.weight_mats[i]
          prev_norm = 1
          if i > 0:
            #prev_norms = self.weight_norms[-1]
            #prev_norm = LA.norm(prev_norms,ord=2)
            prev_norm = self.layer_matrix_norms[-1]
          w_norm = []
          for j in range(weight.shape[0]):
            w_norm.append((LA.norm(weight[j,:],ord=2)**2)*prev_norm)
          self.weight_norms.append(w_norm)
          self.layer_matrix_norms.append(prev_norm * (LA.norm(weight,ord=2)**2))

    def construct_mat(self):
        #n_hidden_vars is the number of hidden nodes
        self.n_hidden_vars = sum(self.weight_dims[1:])
        #Constructing some auxilary matrices for the SDP program
        weights = block_diag(*self.weight_mats[:-1])
        zeros_col = np.zeros((weights.shape[0], self.weight_dims[-1]))
        A = np.concatenate((weights, zeros_col), axis=1)
        eyes = np.identity(A.shape[0])
        init_col = np.zeros((eyes.shape[0], self.weight_dims[0]))
        B = np.concatenate((init_col, eyes), axis=1)
        A_on_B = np.concatenate((A, B), axis = 0)
        extra_col = np.zeros((A_on_B.shape[0], 1))
        self.mult = np.concatenate((extra_col, A_on_B), axis=1)
        self.constraint_size = self.mult.shape[1]


    def solve_sdp(self):
        final_weight = self.weight_mats[-1][self.pair[0],:] - self.weight_mats[-1][self.pair[1],:]
        final_weight = np.expand_dims(final_weight, axis=0)
        L_sq = cp.Variable((1,1), nonneg=True)
        l2v = cp.Variable((1,1), nonneg=True)
        D = cp.Variable((1, self.n_hidden_vars), nonneg=True)
        DR = cp.Variable((self.n_hidden_vars, 1), nonneg=True)
        T = cp.diag(D)
        Q = cp.bmat([[0*T, T],[T, -2*T]])
        #const_matrix = self.mult.transpose() @ Q @ self.mult
        w_norms = []
        for weight_norm in self.weight_norms:
          w_norms += weight_norm
        #print("w_norms size:", len(w_norms))
        #Create Sparse Diagonal Variable Matrix
        l2_norm_array = np.expand_dims(np.array(w_norms), axis=0)
        #print(l2_norm_array.shape)
        #print(DR.shape)

        obj_term = L_sq-l2_norm_array@DR - l2v
        #obj_term = L_sq- l2v
        #print(obj_term.shape)
        sparse_vars = cp.vstack([obj_term] + [l2v]*self.weight_dims[0]+[DR])
        positions = []
        for i in range(self.constraint_size):
            positions.append([i, i])
        #assert len(range(1, self.weight_dims[0]+1)) == self.weight_dims[0]
        V = np.ones(self.constraint_size)
        I = []
        J = []
        for idx, (row, col) in enumerate(positions):
            I.append(row + col*self.constraint_size)
            J.append(idx)
        reshape_mat = sp.coo_matrix((V, (I,J)), shape=(self.constraint_size*self.constraint_size, self.constraint_size))
        M = cp.reshape(reshape_mat @ sparse_vars, (self.constraint_size,self.constraint_size))
        #Another Matrix
        N = np.zeros([self.constraint_size,self.constraint_size])
        N[0, self.constraint_size-self.weight_dims[-1]:] = -final_weight
        N[self.constraint_size-self.weight_dims[-1]:, 0] = -final_weight
        #The CVX optimization program
        prob = cp.Problem(cp.Minimize(L_sq), [(self.mult.transpose() @ Q @ self.mult) - M - N << 0])
        #Verbose: False if not want to print out the progress from the solver
        #prob.solve(solver=getattr(cp, 'SCS'), verbose=True, **{'gpu': True, 'use_indirect': True, 'eps_abs':1.0, 'max_iters':500})
        prob.solve(solver=cp.MOSEK, verbose=False)
        return prob.value/2, [L_sq.value, l2v.value, D.value, DR.value]

class EigSDP:
    def __init__(self, weights, pair):
        self.weight_mats = weights
        self.layers=len(weights)
        self.weight_dims = [0] * self.layers
        for i in range(self.layers):
            self.weight_dims[i] = self.weight_mats[i].shape[1]
        #nClasses is the number of classification classes, e.g., nClasses=10 for CIFAR10
        self.l2_norm()
        self.normalize_weights()
        self.construct_mat()
        self.pair = pair

    def l2_norm(self):
        weight1 = self.weight_mats[0]
        self.weight_norms = []
        for i in range(self.layers-1):
          weight = self.weight_mats[i]
          prev_norm = 1
          if i > 0:
            prev_norms = self.weight_norms[-1]
            prev_norm = LA.norm(prev_norms,ord=2)
          w_norm = []
          for j in range(weight.shape[0]):
            w_norm.append(LA.norm(weight[j,:],ord=2)*prev_norm)
          self.weight_norms.append(w_norm)

    def normalize_weights(self):
        self.normalized_weights = []
        for i in range(self.layers):
          mat = self.weight_mats[i]
          if i==0:
            self.normalized_weights.append(mat)
          else:
            norm = self.weight_norms[i-1]
            normalized_weight = np.matmul(mat,np.diag(np.array(norm)))
            self.normalized_weights.append(normalized_weight)

    def construct_mat(self):
        #n_hidden_vars is the number of hidden nodes
        self.n_hidden_vars = sum(self.weight_dims[1:])
        #Constructing some auxilary matrices for the SDP program
        weights = block_diag(*self.normalized_weights[:-1])
        zeros_col = np.zeros((weights.shape[0], self.weight_dims[-1]))
        A = np.concatenate((weights, zeros_col), axis=1)
        #print("eyes shape", A.shape[0])
        #eyes = np.identity(A.shape[0])
        w_norms = []
        for weight_norm in self.weight_norms:
          w_norms += weight_norm
        #print("weight shape", len(w_norms))
        eyes = np.diag(np.array(w_norms))
        init_col = np.zeros((eyes.shape[0], self.weight_dims[0]))
        B = np.concatenate((init_col, eyes), axis=1)
        A_on_B = np.concatenate((A, B), axis = 0)
        extra_col = np.zeros((A_on_B.shape[0], 1))
        self.mult = np.concatenate((extra_col, A_on_B), axis=1)
        self.constraint_size = self.mult.shape[1]


    def solve_sdp(self):
        final_weight = self.normalized_weights[-1][self.pair[0],:] - self.normalized_weights[-1][self.pair[1],:]
        D = cp.Variable((1, self.n_hidden_vars), nonneg=True)
        DR1 = cp.Variable((1, 1), nonneg=True)
        DR2 = cp.Variable((1, 1), nonneg=True)
        DR3 = cp.Variable((self.n_hidden_vars,1), nonneg=True)
        T = cp.diag(D)
        Q = cp.bmat([[0*T, T],[T, -2*T]])
        const_matrix = self.mult.transpose() @ Q @ self.mult
        #Create Sparse Diagonal Variable Matrix
        #l2_norm_array = np.array(self.weight_norms[-1])
        #normed_weight = final_weight * l2_norm_array
        sparse_vars = cp.vstack([DR1] + [DR2]*self.weight_dims[0]+[DR3])
        positions = []
        for i in range(self.constraint_size):
            positions.append([i, i])
        #assert len(range(1, self.weight_dims[0]+1)) == self.weight_dims[0]
        V = np.ones(self.constraint_size)
        I = []
        J = []
        for idx, (row, col) in enumerate(positions):
            I.append(row + col*self.constraint_size)
            J.append(idx)
        reshape_mat = sp.coo_matrix((V, (I,J)), shape=(self.constraint_size*self.constraint_size, self.constraint_size))
        M = cp.reshape(reshape_mat @ sparse_vars, (self.constraint_size,self.constraint_size))
        #Another Matrix
        N = np.zeros([self.constraint_size,self.constraint_size])
        N[0, self.constraint_size-self.weight_dims[-1]:] = -final_weight
        N[self.constraint_size-self.weight_dims[-1]:, 0] = -final_weight
        #The CVX optimization program
        prob = cp.Problem(cp.Minimize(DR1+DR2+cp.sum(DR3)), [M - (self.mult.transpose() @ Q @ self.mult) - N >> 0])
        #Verbose: False if not want to print out the progress from the solver
        #prob.solve(solver=getattr(cp, 'SCS'), verbose=True, **{'gpu': True, 'use_indirect': True, 'eps_abs':1.0, 'max_iters':500})
        prob.solve(solver=cp.MOSEK, verbose=False)
        return prob.value