import matplotlib.pyplot as plt
from numpy import eye, heaviside, diag, pi, array, zeros, conjugate, linalg, sum as npsum
from .param import *

# LAPLACIANO 1D DISCRETIZADO
def laplaciano(dx, N):
    return -(1 / (dx ** 2)) * (2 * eye(N) - eye(N, k=-1) - eye(N, k=1))

#ENERGIA CINÉTICA
def energia_cinetica(m, dx, N):
    return -(0.5/m)*laplaciano(dx,N)

# DEFINIÇÃO DO POTENCIAL ESTRUTURAL
def poco_quadrado_finito(profundidade, xmin, xmax, x):
    return -profundidade * heaviside(x-xmin, 1) * heaviside(xmax-x, 1) + profundidade

# POTENCIAL DE HARTREE 
def V_hartree(n_3d, k, dx, N, x):
  L=int(dx*(N+1)) 
  V_antes = linalg.solve(laplaciano(dx, N), (-4*pi/k)*n_3d)
  alpha = (V_antes[int((N-3)/4)] - V_antes[int(((3*N)-1)/4)])/(.5*L)
  beta = - V_antes[int((N-3)/4)]
  # x = arange(1, N+1)*dx
  reta_correcao = alpha*(x-(L/4)) + beta
  return V_antes + reta_correcao # V_hartree com as bordas sempre em V=0 !!

# POTENCIAL DE HARTREE TEMPERADO
def V_hartree_temperado(tempero, V_1, V_3):
    return tempero*V_3 + (1-tempero)*V_1

# POTENCIAL TOTAL 
def V_total(V_estrutural, V_interacao):
    return V_estrutural + V_interacao

# HAMILTONIANO
def H(V_efetivo, m, dx, N):
    return energia_cinetica(m, dx, N) + diag(V_efetivo)

# E0_mu (soma de níveis ocupados)
def E0_mu(mu, E0_list):
    return npsum(E0_list, where=E0_list<=mu)

# Numero de níveis ocupados 
def N_levels(mu, E0_list):
    return npsum(E0_list<=mu)

# D.O.L 3D
def n_3d(phi_list, E0_list, mu, m, N): 
  vectors = array([(m*(mu-E0_list[j])/(2*pi*(h_bar**2)))*(conjugate(phi_list[j])*phi_list[j]) for j in range(N_levels(mu, E0_list))])
  if N_levels(mu, E0_list) > 1:
    return npsum(vectors, axis=0)
  elif N_levels(mu, E0_list) == 1:
    return vectors[0]
  return zeros(N)

# D.O.L 2D
def n_2d_params(mu, m, E0_list):
    return [npsum(E0_list<=mu)*m/(2*pi*(h_bar**2)),-E0_mu(mu, E0_list)*m/(2*pi*(h_bar**2))] #retorna os coeficientes linear e angular, n_2d_params[0]==a e n_2d_params[1]==b

# funcaozinha para inserir o range do eixo x, autovalores, autovetores e o numero de energias desejado na visualização; possivel inserir potencial para plotar junto
def plotEnPsi_finite_diff(x_values, en, psi, n, scale,ymin,ymax,xmin=0, xmax=800, potential=0):
    spot_size=1
    plt.scatter(x_values, potential, c="C1140", s=spot_size)
    for j in range(n):
        plt.hlines(
            en[j],
            xmin=x_values[0],
            xmax=x_values[-1],
            colors="C" + str(j),
            linestyles="--",
        )
        plt.scatter(x_values, en[j] + scale * psi[:, j],  c="C" + str(j), s=spot_size)
        y_min, y_max = (ymin, ymax)
        plt.ylim(y_min, y_max)
        x_min, x_max = (xmin, xmax)
        plt.xlim(x_min, x_max)
    plt.show()