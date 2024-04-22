import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from numpy import linalg as LA
from utils import max_eigv, flatten, power_iteration
import torch.optim.lr_scheduler as lr_scheduler
import os
from SDP_solver import ReduntSDP

class FOSDP:
    def __init__(self, weights, pair, device, epochs=1000, random_init=True, weight_types=None, params=None, memory_eff=False):
        self.device=device
        self.weight_mats = weights
        self.weight_types = weight_types
        self.params = params
        self.pair = pair
        self.layers=len(weights)
        self.weight_dims = [0] * self.layers
        for i in range(self.layers):
            self.weight_dims[i] = self.weight_mats[i].shape[1]
        self.constraint_size = sum(self.weight_dims)+1
        self.n_hidden_vars = sum(self.weight_dims[1:])
        print("the constraint size of the SDP is:", self.constraint_size)
        self.l2_norm()
        self.hidden_idx()
        self.create_weight_tensor()
        #self.normalize_weights()
        #self.u_final = self.weight_mats[-1][pair[0],:] - self.weight_mats[-1][pair[1],:]
        if len(pair) == 2:
            self.final_weight = self.weights_tensor[-1][pair[0],:] - self.weights_tensor[-1][pair[1],:]
        else:
            self.final_weight = self.weights_tensor[-1][pair[0],:]
        self.memory_eff = memory_eff
        if not memory_eff:
            #This step creates the multiplier that is used to build the whole matrix, which is not necessary
            self.construct_mat()
        self.epochs = epochs
        self.random_init = random_init

    def create_weight_tensor(self):
        self.weights_tensor = []
        for w in self.weight_mats:
            self.weights_tensor.append(torch.tensor(w).to(self.device))

    def hidden_idx(self):
        starting = 0
        pox = [starting]
        total_pox = [starting, self.weight_dims[0]]
        for i in range(1, self.layers):
            starting += self.weight_dims[i]
            pox.append(starting)
            total_pox.append(starting+self.weight_dims[0])
        self.hidden_idx = pox
        self.total_idx = total_pox

    def reverse_norm_prod(self):
        self.reverse_layer_norms = []
        prod = 1
        for w_norm in reversed(self.single_layer_norms):
            self.reverse_layer_norms.append(prod)
            prod *= w_norm
        self.reverse_layer_norms.append(prod)

    def l2_norm(self):
        self.weight_norms = []
        self.layer_matrix_norms = []
        self.single_layer_norms = []
        for i in range(self.layers-1):
          weight = self.weight_mats[i]
          single_norm = LA.norm(weight,ord=2)
          self.single_layer_norms.append(single_norm)
          prev_norm = 1
          if i > 0:
            prev_norm = self.layer_matrix_norms[-1]
          w_norm = []
          for j in range(weight.shape[0]):
            w_norm.append((LA.norm(weight[j,:],ord=2)**2)*prev_norm)
          self.weight_norms += w_norm
          self.layer_matrix_norms.append(prev_norm * (single_norm**2))
        self.reverse_norm_prod()
        self.weight_sum = np.sum(self.weight_norms).item()

    def construct_mat(self):
        #n_hidden_vars is the number of hidden nodes
        #Constructing some auxilary matrices for the SDP program
        weights = block_diag(*self.weight_mats[:-1])
        zeros_col = np.zeros((weights.shape[0], self.weight_dims[-1]))
        A = np.concatenate((weights, zeros_col), axis=1)
        eyes = np.identity(A.shape[0])
        init_col = np.zeros((eyes.shape[0], self.weight_dims[0]))
        B = np.concatenate((init_col, eyes), axis=1)
        A_on_B = np.concatenate((A, B), axis = 0)
        extra_col = np.zeros((A_on_B.shape[0], 1))
        self.mult = torch.tensor(np.concatenate((extra_col, A_on_B), axis=1)).to(self.device)
        #self.constraint_size = self.mult.shape[1]

    def lanc_iter_op(self, op, size, rv):
        #We generate a smaller tridiagonal matrix with lanczos algorithm from mat
        #Notice that mat is only used in mat*q
        lan_subm = torch.zeros(size, size).to(self.device)
        rand_vec = rv/torch.linalg.norm(rv)
        b = 1
        q_old = 0
        l_min = float('inf')
        for i in range(size):
            q = rand_vec/b
            mat_q = op(q) #torch.matmul(mat, q)
            a = torch.matmul(torch.t(q), mat_q)
            rand_vec = mat_q - a*q - b*q_old
            b_old = b
            b = torch.linalg.norm(rand_vec)
            q_old=q
            lan_subm[i,i] = a
            l_min = min(l_min, a-abs(b_old)-abs(b))
            if b==0 or i== size-1:
                break
            lan_subm[i, i+1] = b
            lan_subm[i+1, i] = b
        return lan_subm        

    def lanc_iter(self, mat, size, rv):
        #We generate a smaller tridiagonal matrix with lanczos algorithm from mat
        #Notice that mat is only used in mat*q
        lan_subm = torch.zeros(size, size).to(self.device)
        rand_vec = rv/torch.linalg.norm(rv)
        b = 1
        q_old = 0
        l_min = float('inf')
        for i in range(size):
            q = rand_vec/b
            mat_q = torch.matmul(mat, q)
            a = torch.matmul(torch.t(q), mat_q)
            rand_vec = mat_q - a*q - b*q_old
            b_old = b
            b = torch.linalg.norm(rand_vec)
            q_old=q
            lan_subm[i,i] = a
            l_min = min(l_min, a-abs(b_old)-abs(b))
            if b==0 or i== size-1:
                break
            lan_subm[i, i+1] = b
            lan_subm[i+1, i] = b
        return lan_subm
    
    def mat_mul(self, W, x, shape=None, weight_trans=False, module=False, type="linear", kernel_mat=[0, 0], stride=0, padding=0):
        #weight trans denotes whether it is of the type W^T * x or W * x
        #Recall that W * x = (x^T * W^T)^T = (F.linear(x^T, W))^T
        if module:
            if type == "linear":
                if weight_trans:
                    W = W.T
                out = F.linear(x.T, weight=W).T
            if type == "conv2d":
                x = x.reshape(shape).unsqueeze(0)
                if weight_trans:
                    out = F.conv_transpose2d(x, kernel_mat, stride=stride, padding=padding)
                else:
                    out = F.conv2d(x, kernel_mat, stride=stride, padding=padding)
                out = flatten(out).T
            return out
        else:
            if weight_trans:
                W = W.T
            return torch.matmul(W, x)

    def eig_mat_op(self, kappa, dual, q, module):
        prod = torch.zeros(q.shape[0], 1).to(self.device)
        ###First element
        weight_mult = torch.tensor(self.weight_norms).float().to(self.device)
        ##Input layer
        prod[0] = (torch.inner(weight_mult, kappa[2:])+kappa[1]-kappa[0])*q[0] + torch.matmul(q[self.total_idx[self.layers-1]+1:, :].T,  self.final_weight) #torch.matmul(q[self.total_idx[self.layers-1]+1:, :].T, self.final_weight)
        D = torch.diag(dual[self.hidden_idx[0]:self.hidden_idx[1]])
        prod[1:1+self.total_idx[1]] = -kappa[1]*q[1:1+self.total_idx[1]] + \
            self.mat_mul(self.weights_tensor[0],torch.matmul(D, q[1+self.total_idx[1]:1+self.total_idx[2], :]), weight_trans=True, module=module, type=self.weight_types[0], shape=[self.params[1][0]]+self.params[1][4], kernel_mat=self.params[1][1], stride=self.params[1][2], padding=self.params[1][3]) 
        ###Inner Hidden layers
        for i in range(1, self.layers-1):
            D_prev = D
            D = torch.diag(dual[self.hidden_idx[i]:self.hidden_idx[i+1]])
            D_k = torch.diag(kappa[2+self.hidden_idx[i-1]:2+self.hidden_idx[i]])
            prod[1+self.total_idx[i]:1+self.total_idx[i+1]] = torch.matmul(-2*D_prev-D_k,q[1+self.total_idx[i]:1+self.total_idx[i+1]])+\
                self.mat_mul(self.weights_tensor[i],torch.matmul(D, q[1+self.total_idx[i+1]:1+self.total_idx[i+2], :]), weight_trans=True, module=module,type=self.weight_types[i], shape=[self.params[i+1][0]]+self.params[i+1][4], kernel_mat=self.params[i+1][1], stride=self.params[i+1][2], padding=self.params[i+1][3])+\
                    torch.matmul(D_prev,self.mat_mul(self.weights_tensor[i-1], q[1+self.total_idx[i-1]:1+self.total_idx[i], :], module=module, type=self.weight_types[i-1], shape=[self.params[i-1][0]]+self.params[i-1][4], kernel_mat=self.params[i][1], stride=self.params[i][2], padding=self.params[i][3]))
        ###Final layer
        D_prev = D
        D_k = torch.diag(kappa[2+self.hidden_idx[self.layers-2]:2+self.hidden_idx[self.layers-1]])
        prod[1+self.total_idx[self.layers-1]:1+self.total_idx[self.layers]] = torch.matmul(-2*D_prev-D_k,q[1+self.total_idx[self.layers-1]:])+\
            torch.matmul(D_prev, self.mat_mul(self.weights_tensor[self.layers-2], q[1+self.total_idx[self.layers-2]:self.total_idx[self.layers-1]+1, :], module=module, type=self.weight_types[self.layers-2], shape=[self.params[self.layers-2][0]]+self.params[self.layers-2][4], kernel_mat=self.params[self.layers-1][1], stride=self.params[self.layers-1][2], padding=self.params[self.layers-1][3]))+\
                torch.unsqueeze(q[0]*self.final_weight, axis=1)

        return prod

    def eig_mat(self, kappa, dual):
        T = torch.diag(dual)
        objv = kappa[0]
        in_var = kappa[1]
        hid_vars = kappa[2:]
        upper = torch.cat([0*T,T], dim=1)
        lower = torch.cat([T, -2*T], dim=1)
        Q = torch.cat([upper,lower],dim=0).double()
        const_matrix = torch.matmul(torch.matmul(torch.transpose(self.mult,0,1), Q), self.mult)
        const_matrix[0, self.constraint_size-self.weight_dims[-1]:] = const_matrix[0, self.constraint_size-self.weight_dims[-1]:] \
            +self.final_weight
        const_matrix[self.constraint_size-self.weight_dims[-1]:, 0] = const_matrix[self.constraint_size-self.weight_dims[-1]:, 0] \
            +self.final_weight
        #Done LipSDP except for the input constraint

        const_matrix[self.weight_dims[0]+1:,self.weight_dims[0]+1:] = const_matrix[self.weight_dims[0]+1:,self.weight_dims[0]+1:] \
            - torch.diag(hid_vars)
        const_matrix[1:self.weight_dims[0]+1,1:self.weight_dims[0]+1] = const_matrix[1:self.weight_dims[0]+1,1:self.weight_dims[0]+1] \
            - torch.eye(self.weight_dims[0]).to(self.device)*in_var
        const_matrix[0,0] = torch.inner(torch.tensor(self.weight_norms).float().to(self.device), hid_vars)+in_var-objv
        return const_matrix


    def init_vars(self, sol, init):
        kappa = torch.zeros(self.n_hidden_vars+2,device=self.device)
        dual = torch.zeros(self.n_hidden_vars, device=self.device)
        if init=="schur":
            v_norm = torch.linalg.vector_norm(self.final_weight, ord=2)
            c = v_norm/self.reverse_layer_norms[-1]
            kappa[0] = 2*self.reverse_layer_norms[-1]*v_norm
            kappa[1] = self.reverse_layer_norms[-1]*v_norm
            dvs = []
            for j in range(self.layers-1):
                dv = [c*(self.reverse_layer_norms[self.layers-2-j]**2)]*self.weight_dims[j+1]
                dvs += dv
            dual = torch.tensor(dvs, device=self.device)
        elif init == "diag":
            kappa[self.n_hidden_vars+2-self.weight_dims[-1]:] = torch.absolute(self.final_weight)
            kappa[0] = torch.sum(torch.absolute(self.final_weight))
        elif init == "analytical":
            solver = ReduntSDP(self.weight_mats, self.pair)
            opt, vals = solver.solve()
            print("sdp analytical sol is", opt)
            dual_sol = vals[2]
            ####
            vs = []
            for v in vals[0][0]:
                vs.append(v)
            for v in vals[1][0]:
                vs.append(v)
            for v in vals[3]:
                vs.append(v[0])
            ###
            dual = torch.tensor(dual_sol,device=self.device)
            kappa = torch.tensor(vs, device=self.device)
        else:
            kappa = torch.rand(self.n_hidden_vars+2, device=self.device)
            dual = torch.rand(self.n_hidden_vars, device=self.device)
        dual.requires_grad=True
        kappa.requires_grad=True
        if not sol == None:
            dual, kappa = sol[0].clone().detach().requires_grad_(True), sol[1].clone().detach().requires_grad_(True)
        return kappa, dual

    def lanc_eig_init(self, kappa, dual, sparse=False, module=False):
        eig_m = self.eig_mat(kappa, dual)
        _, eig_vec = eigv_init(eig_m, self.device)
        eig_vec = eig_vec.detach()
        return eig_vec
    
    def lanc_iter_warm_op(self, op, size, init_vec):
        #We generate a smaller tridiagonal matrix with lanczos algorithm from mat
        #Notice that mat is only used in mat*q
        lan_subm = torch.zeros(size, size).to(self.device)
        Q_vecs = []
        rand_vec = init_vec/torch.linalg.norm(init_vec)
        b = 1
        q_old = 0
        eigv = None
        #print("init vec", init_vec, "mat", mat)
        for i in range(size):
            q = rand_vec/b
            Q_vecs.append(q)
            mat_q = op(q) #torch.matmul(mat, q)
            a = torch.matmul(torch.t(q), mat_q)
            rand_vec = mat_q - a*q - b*q_old
            b = torch.linalg.norm(rand_vec)
            q_old=q
            lan_subm[i,i] = a
            #Now let's do a comparison
            if eigv is None:
                eigv = a
            else:
                temp_subm = lan_subm[:i+1, :i+1]
                Ls, Qs = torch.linalg.eigh(temp_subm)
                temp_eigv = Ls[-1]
                if abs(temp_eigv - eigv)/abs(temp_eigv) < 1e-6:
                    Q_mat = torch.cat(Q_vecs,dim=1)
                    eig_vec = torch.matmul(Q_mat, Qs[:,-1]).unsqueeze(1)
                    return Ls[-1], eig_vec
                else:
                    eigv = temp_eigv
            if b==0 or i== size-1:
                break
            lan_subm[i, i+1] = b
            lan_subm[i+1, i] = b
        Q_mat = torch.cat(Q_vecs,dim=1)
        Ls, Qs = torch.linalg.eigh(lan_subm)
        eig_vec = torch.matmul(Q_mat, Qs[:,-1]).unsqueeze(1)
        return Ls[-1], eig_vec

    def lanc_eig_loss(self, kappa, dual, iterations, eig_vec, sparse=False, lanc=False, lan_steps=30, module=False):
        if sparse:
            #rv = torch.rand(self.constraint_size, 1).to(self.device)
            def op(q):
                return self.eig_mat_op(kappa, dual, q, module=module)
            eigv, updated_eig_vec = self.lanc_iter_warm_op(op, size=lan_steps, init_vec=eig_vec)
            updated_eig_vec = updated_eig_vec.detach()
            #L = torch.linalg.eigvalsh(eig_m)
        else:
            eig_m = self.eig_mat(kappa, dual)
            #rv = torch.rand(self.constraint_size, 1).double().to(self.device)
            eigv, updated_eig_vec = self.lanc_iter_warm(eig_m, size=lan_steps, init_vec=eig_vec)
            updated_eig_vec = updated_eig_vec.detach()
            #L = torch.linalg.eigvalsh(eig_m)
        return eigv, updated_eig_vec #L[-1] 

    def eig_loss(self, kappa, dual, iterations, sparse=False, lanc=False, lan_steps=30, module=False):
        if lanc:
            if sparse:
                rv = torch.rand(self.constraint_size, 1).to(self.device)
                def op(q):
                    return self.eig_mat_op(kappa, dual, q, module=module)
                eig_m = self.lanc_iter_op(op, lan_steps, rv)
                L = torch.linalg.eigvalsh(eig_m)
            else:
                eig_m = self.eig_mat(kappa, dual)
                rv = torch.rand(self.constraint_size, 1).double().to(self.device)
                eig_m = self.lanc_iter(eig_m, lan_steps, rv)
                L = torch.linalg.eigvalsh(eig_m)
        else:
            eig_m = self.eig_mat(kappa, dual)
            L = torch.linalg.eigvalsh(eig_m)
        return L[-1]

    def solve_eig(self, sol=None, lr=0.1, verbose=False, lanc=False, lan_steps=20, ratio=5, group_val=2, sparse=False, init="random", module=False):
        #The initialization strategy for initializing the SDP, random, schur, diag
        groups = group_val
        group_its = int(self.epochs/groups)
        best_kappa, best_dual = self.init_vars(sol, init)
        eig_vec = self.lanc_eig_init(best_kappa, best_dual)
        stepsize = int(group_its/50)
        def lr_lambda(epoch):
            max_lr = 1.0
            min_lr = 1e-1
            decay = 0.95
            weight = decay**(int(epoch/stepsize))
            return max_lr*weight + (1-weight)*min_lr
        
        min_loss = float('inf')
        total_its = 0
        for gp in range(groups):
            dual, kappa = best_dual, best_kappa
            opt1 = torch.optim.Adam([dual],lr=lr)
            opt2 = torch.optim.Adam([kappa],lr=lr*ratio)
            scheduler1 = lr_scheduler.LambdaLR(opt1, lr_lambda)
            scheduler2 = lr_scheduler.LambdaLR(opt2, lr_lambda)

            for it in range(group_its):
                opt1.zero_grad()
                opt2.zero_grad()

                pos_dual = torch.absolute(dual)
                pos_kappa = torch.absolute(kappa)
                if it > group_its-5 or it%100==0:
                    if self.memory_eff:
                        lanc_val=lanc
                        lan_steps_val = lan_steps+20
                    else:
                        lanc_val=False
                        if verbose:
                            print(f"No Lanc at {it} th iteration")
                else:
                    lanc_val=lanc
                lan_steps_val = lan_steps
                eigv = self.eig_loss(pos_kappa, pos_dual, it, lanc=lanc_val, lan_steps=lan_steps_val, sparse=sparse, module=module)
                loss = (self.weight_sum+2)*torch.nn.functional.relu(eigv)+pos_kappa[0]
                if lanc_val==False:
                    if loss < min_loss:
                        min_loss = loss
                        best_kappa = kappa
                        best_dual = dual
                    if (it > group_its-5 or it%100==0) and verbose:
                        print("True loss")
                if (it%50==0 or it > group_its-5) and verbose:
                    print(f'{total_its+it} iteration with loss {loss/2}')
                loss.backward()
                opt1.step()
                opt2.step()
                scheduler1.step()
                scheduler2.step()
            total_its += group_its
        return min_loss/2