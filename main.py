from pso import pso
from portfolio_objectives import objective, stocks_returns_mean, covariance_matrix
import numpy as np


V, l_bound, u_bound, f, constr = objective()

print(f"V = {V}, l_bound = {l_bound}, u_bound = {u_bound}")


P = pso(V,l_bound,u_bound,f,constraints=constr,c=2.13,s=1.05,w=0.41,pop=200)
# P.plot()
P.moving(100 ,1000)
P.plot_3d()
Pareto_Front, Optimal_Solutions = P.get_solution()
print(f"Pareto Front : {Pareto_Front}")
for Optimal_Solution in Optimal_Solutions:
    print(Optimal_Solution.position)
    print(f"expected return = {stocks_returns_mean.T.dot(Optimal_Solution.position)}")
    print(f"objective return = {Optimal_Solution.obj_values[1]}")
    print(f"risk = {(np.array(Optimal_Solution.position).T).dot(covariance_matrix).dot(np.array(Optimal_Solution.position))}")
    print(f"objective risk = {Optimal_Solution.obj_values[0]}")