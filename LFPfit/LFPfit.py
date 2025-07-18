import numpy as np
import torch
from torch import autograd
import matplotlib.pyplot as plt
from .GibbsFE import solve_Theta_max
from .GibbsFE import GibbsFE_PDOS as GibbsFE
from .CommonTangent import _get_phase_boundaries 


global  _eps
_eps = 1e-7







# global settings
quaderature_points = 20
style = "Legendre"




g_i_list_Li = [2.20836568 , 0.58406430 , 1.28776312 , -0.53589582 ] # pretrained via fit_pure_substance_Cp
g_i_list_HM = [2.88982511, -4.33991003, 2.75108194, 0.85858440  ] # pretrained via fit_pure_substance_Cp
g_i_list_LiHM = [ 1.34505212, -0.47250998, -0.80727065, 0.45194003] # pretrained via fit_pure_substance_Cp


# convert pretrained value into torch.tensor
n_list = [9999.9, 6.0, 1.0]
g_i_list_Li = torch.tensor(g_i_list_Li) 
g_i_list_HM = torch.tensor(g_i_list_HM)
g_i_list_LiHM = torch.tensor(g_i_list_LiHM) 
Theta_max_Li = solve_Theta_max(g_i_list = g_i_list_Li, is_x_dependent = False, style = style)
Theta_max_Li = Theta_max_Li.detach()
Theta_max_HM = solve_Theta_max(g_i_list = g_i_list_HM, is_x_dependent = False, style = style)
Theta_max_HM = Theta_max_HM.detach()
Theta_max_LiHM = solve_Theta_max(g_i_list = g_i_list_LiHM, is_x_dependent = False, style = style)
Theta_max_LiHM = Theta_max_LiHM.detach()



# Below is the first OCV fit   (worked sort of well but did not capture low SOX)  
Omega_G0_list = [77468.8438, -62579.3047, -33789.5312, 13239.6562, -29306.2344, -193994.0781] 
S_config_params_list = [-7.80139732, 6.52273035, -2.74779415, 2.94423103, -1.16825449]
g_ij_list_LixHM = [[1.43152177, 0.41842270], [-1.60297871, 0.18667871], [0.95630181, -2.06099367], [3.77412558, 1.46913910]]   


total_params_list = [Omega_G0_list, S_config_params_list, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_HM, Theta_max_Li]

def LFP_OCP(x_measured, T, GibbsFE = GibbsFE, total_params_list = total_params_list, style="Legendre", quaderature_points=5):
    #E: currently x_measured is taken in as a numpy
    #x_measured = x_measured.astype("float32")
    x_measured = np.array(x_measured)
    T = np.array(T)
    x_measured  = torch.from_numpy(x_measured).double()

    phase_boundary_fixed_point = _get_phase_boundaries(GibbsFE, total_params_list, T, style = style, quaderature_points = quaderature_points, ngrid=99)
    # first predict the one before common tangent
    mu_pred = []
    for i in range(0, len(x_measured)):
        x_now = x_measured[i]
        x_now = x_now.requires_grad_()
        g_now = GibbsFE(x_now, T, total_params_list, style = style, quaderature_points = quaderature_points)
        mu_pred_now = autograd.grad(outputs=g_now, inputs=x_now, create_graph=True)[0]
        mu_pred.append(mu_pred_now.detach().numpy())
    mu_pred = np.array(mu_pred)
    # calculate mu after common tangent construction
    mu_pred_after_ct = []
    # see if x is inside any gaps
    def _is_inside_gaps(_x, _gaps_list):
        _is_inside = False
        _index = -99999
        for i in range(0, len(_gaps_list)):
            if _x >= _gaps_list[i][0] and _x <= _gaps_list[i][1]:
                _is_inside = True
                _index = i
                break
        return _is_inside, _index
    # pred
    for i in range(0, len(x_measured)):
        x_now = x_measured[i]
        is_inside, index = _is_inside_gaps(x_now, phase_boundary_fixed_point)
        if is_inside == False:
            # outside miscibility gap 
            mu_pred_after_ct.append(mu_pred[i])
        else: 
            # inside miscibility gap
            x_alpha = phase_boundary_fixed_point[index][0]
            x_beta = phase_boundary_fixed_point[index][1]
            ct_pred = (GibbsFE(x_alpha, T, total_params_list, style = style, quaderature_points = quaderature_points) - GibbsFE(x_beta, T, total_params_list, style = style, quaderature_points = quaderature_points))/(x_alpha - x_beta) 
            if torch.isnan(ct_pred) == False:
                mu_pred_after_ct.append(ct_pred.clone().detach().numpy()) 
            else:
                mu_pred_after_ct.append(mu_pred[i])
    mu_pred_after_ct = np.array(mu_pred_after_ct)
    U_pred_after_ct = mu_pred_after_ct/(-96485)
    return U_pred_after_ct


