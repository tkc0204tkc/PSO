from particle import *
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import time
import sys
from tqdm import tqdm
import pickle
from portfolio_objectives import stocks_returns_mean, covariance_matrix
class pso:
    '''
    c_param -> personal parameter
    s_param -> sozial parameter
    v_weight -> weight for old velocity
    swarm -> all particles
    comp_swarm -> swarm + g_best + p_best
    l_bound -> lower bounds
    u_bound -> upper bounds
    integer -> Integer constraint for every attribute
    vmax -> max velocity
    g_best -> best particle of the swarm
    multi -> multi objective or single objective
    '''
    
    def __init__(self,att,l_b,u_b,obj_func,constraints=[],c=2.1304,s=1.0575,w=0.4091,pop=100,vm=np.nan,integer=False):
        if np.isnan(vm):
            vm = np.array([u_b[i]-l_b[i] for i in range(att)])
        if type(vm) != np.ndarray and type(vm) != list:
            vm = np.array([vm for i in range(att)])
        if len(vm)!=att:
            np.append(vm, [vm[len(vm)-1]for i in range(len(l_b),att)])
        if type(integer) != list:
            integer = np.array([integer for i in range(att)])
        self.c_param = c
        self.s_param = s
        self.v_weight = w
        self.l_bound = l_b
        self.u_bound = u_b
        self.integer = integer
        self.vmax = vm
        # self.swarm = [particle_multi(obj_func,att,constraints,vm,l_b,u_b,integer) for i in range(pop)]
        self.swarm = []
        for i in range(pop):
            if i < att:
                self.swarm.append(particle_multi(obj_func, att, constraints, vm, l_b, u_b, integer, init_method=2, extreme_asset=i))
            elif att <= i < att*2:
                self.swarm.append(particle_multi(obj_func, att, constraints, vm, l_b, u_b, integer, init_method=1))
            else:
                self.swarm.append(particle_multi(obj_func, att, constraints, vm, l_b, u_b, integer))

        self.comp_swarm = self.swarm
        self.non_dom_sort()
        for part in self.swarm:
            part.init_p_best()
        self.set_g_best()

        
    def non_dom_sort(self):
        # fast non domination sort
        F = []
        F1 = []
        for p in self.comp_swarm:
            Sp = []
            n_p = 0
            for q in self.comp_swarm:
                if p.dominates(q):
                    Sp.append(q)
                elif q.dominates(p):
                    n_p +=1
            p.S = Sp
            p.n = n_p
            if n_p==0:
                F1.append(p)
                p.rank = 0
        F.append(F1)
        i=0
        while F[i]:
            H = []
            for p in F[i]:
                for q in p.S:
                    q.n -= 1
                    if q.n == 0:
                        H.append(q)
                        q.rank = i+1
            i +=1
            F.append(H)
        F.pop()
        # crowding distance
        for Fi in F:
            l = len(Fi)
            for parti in Fi:
                parti.distance = 0
            for m in range(len(Fi[0].obj_functions)):
                m_obj = [x.obj_values[m] for x in Fi]
                if l>1:
                    Fi_sorted = [Fi[j] for j in np.argsort(m_obj)]
                else:
                    Fi_sorted = Fi
                Fi_sorted[0].distance = np.inf
                Fi_sorted[-1].distance = np.inf
                for i in range(2,l-1):
                    Fi_sorted[i].distance = Fi_sorted[i].distance + Fi_sorted[i+1].obj_values[m] - Fi_sorted[i-1].obj_values[m]
                    
    def set_g_best(self):
        # sys.setrecursionlimit(5000)
        self.g_best = self.swarm[0]
        for part in self.swarm[1:]:
            if part.compare(self.g_best):
                self.g_best = copy.deepcopy(part)
        
    def plot(self, best_p=True, x_coord=0, y_coord=1):
        '''
        best_p = True or False -> want to plot the personal best or the actual position of all particles
        x_coord = 0,1,... -> for single: which position variable should be plotted ; for multi: which objective value should be plotted on the y axis
        y_coord = 0,1,... -> for single: not relevant ; for multi: which objective value should be plottet on the y axis
        '''
        for partic in self.swarm:
            partic.plot(best_p,x_coord,y_coord)

        plt.xlabel(r'$f_%2i(x_1,x_2,\dots)$' % x_coord)
        plt.ylabel(r'$f_%2i(x_1,x_2,\dots)$' % y_coord)
        plt.title('Pareto Front')
        plt.show()

    def plot_3d(self, best_p=True, x_coord =0, y_coord=1, z_coord=2):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        sequence_containing_x_vals = []
        sequence_containing_y_vals = []
        sequence_containing_z_vals = []
        for partic in self.swarm:
            if best_p:
                sequence_containing_x_vals.append(partic.best_p.obj_values[x_coord])
                sequence_containing_y_vals.append(partic.best_p.obj_values[y_coord])
                sequence_containing_z_vals.append(partic.best_p.obj_values[z_coord])

        ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
        plt.xlabel("Expected Return")
        plt.xlabel("Risk")
        plt.show()



        
    def moving(self,steps, time_termination):
        t0 = time.time()
        sys.setrecursionlimit(50000)
        for i in tqdm(range(steps)):
            if time_termination != -1 and time.time()-t0 > time_termination:
                break

            self.comp_swarm=[]
            self.comp_swarm.append(self.g_best)
            for part in self.swarm:
                #calc new velocity
                r1 = random.random()
                r2 = random.random()
                new_v = self.v_weight*part.velocity + self.c_param*r1*(part.best_p.position-part.position) + self.s_param*r2*(self.g_best.position-part.position)
                # control vmax
                new_v = np.array([new_v[i] if new_v[i]>-self.vmax[i] else -self.vmax[i] for i in range(len(new_v))])
                new_v = np.array([new_v[i] if new_v[i]<self.vmax[i] else self.vmax[i] for i in range(len(new_v))])                
                
                #calc new position
                new_p = part.position + new_v
                for i in range(len(new_p)):
                    if self.integer[i]:
                        new_p[i] = int(new_p[i])
                
                # stick to bound
                new_p = np.array([new_p[i] if new_p[i]>self.l_bound[i] else self.u_bound[i] - abs(self.l_bound[i]-new_p[i])%(self.u_bound[i]-self.l_bound[i]) for i in range(len(new_p))])
                new_p = np.array([new_p[i] if new_p[i]<self.u_bound[i] else self.l_bound[i] + abs(self.u_bound[i]-new_p[i])%(self.u_bound[i]-self.l_bound[i]) for i in range(len(new_p))])
                
                part.set_velocity(new_v)
                part.set_position(new_p)
                
                #add to comp_swarm
                self.comp_swarm.append(part)
                self.comp_swarm.append(part.best_p)
                

            self.non_dom_sort()
            # set g_best with new rank and distance
            # print("testing 1")
            self.g_best = copy.deepcopy(self.comp_swarm[0])
            # print("testing 2")
            j=1
            new_swarm = self.comp_swarm[1:-1:2]
            self.swarm = copy.deepcopy(new_swarm)
            # print("testing 3")
            j+=1
                
            for part in self.swarm:

                #set swarm and p_best with new rank and distance
                part.best_p = copy.deepcopy(self.comp_swarm[j])
                j += 2
                #compare part with g_best
                if part.compare(self.g_best):
                    self.g_best = copy.deepcopy(part)
                #update p_best
                part.compare_p_best()

            # Pareto_Front, Optimal_Solutions = self.get_solution()
            # print(f"Pareto Front in round {steps} : {Pareto_Front}")
            # for Optimal_Solution in Optimal_Solutions:
            #     print(Optimal_Solution.position)
            #     print(f"expected return = {stocks_returns_mean.T.dot(Optimal_Solution.position)}")
            #     print(f"objective return = {Optimal_Solution.obj_values[1]}")
            #     print(f"risk = {(np.array(Optimal_Solution.position).T).dot(covariance_matrix).dot(np.array(Optimal_Solution.position))}")
            #     print(f"objective risk = {Optimal_Solution.obj_values[0]}")
            #     print("----------------------------------------------------------------")



                
                
    def get_solution(self,whole_particle=False):
        solution = []
        particle = []
        # if self.multi:
        for part in self.swarm:
            if part.rank == 0:
                if whole_particle:
                    solution.append(part)
                else:
                    solution.append(part.get_obj_value())
                    particle.append(part)
            if part.best_p.rank == 0 and all(part.position != part.best_p.position):
                if whole_particle:
                    solution.append(part.best_p)
                else:
                    solution.append(part.best_p.get_obj_value())
                    particle.append(part.best_p)
                
        return solution, particle
