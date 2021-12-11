from scipy.optimize import minimize
import numpy as np

def bestfit(params, *args):
    """ function to calculate the 1D or 2D least square between points """

    x0 = np.array(args[0])
    x_ref = np.array(args[1])
    type_of_bf = args[2] or '1d'

    # inicializando variável para cálculo do(s) valor a ser minimizado
    diff = []

    if (type_of_bf == '1d'):
        Ty = params[0]
        for i in range(len(x0)):
            xt = x0[i] + Ty
            diff.append(((x_ref[i]-xt)**2).sum())
    elif (type_of_bf == '2d'):
        Tx = params[0]
        Ty = params[1] 
        for i in range(len(x0[0])):
            xt = x0[0][i] + Tx
            yt = x0[1][i] + Ty
            diff.append(((x_ref[0][i]-xt)**2).sum())
            diff.append(((x_ref[1][i]-yt)**2).sum())
    return np.sqrt(np.sum(diff))

def calc_offset(pts, pts_ref, type_of_bf='1d'):
    """ calculates offset by the means of minimizing the least square between the 2 curves """

    # inicializando array com parâmetros a serem manipulados durante as iterações da minimização
    params = [0] if type_of_bf == '1d' else [1,0]

    # aplicando a operação de minimização para achar os parâmetros de transformação
    offset = minimize(fun=bestfit, x0=params, args=(pts, pts_ref, type_of_bf),method='SLSQP', options={'ftol': 1e-10, 'disp': False})['x']

    return offset