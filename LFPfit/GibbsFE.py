import numpy as np
import torch






global  _eps
_eps = 1e-7





def legendre_poly_recurrence(x, n):
    """
    Compute the Legendre polynomials up to degree n 
    using the Bonnet's recursion formula (i+1)P_(i+1)(x) = (2i+1)xP_i(x) - iP_(i-1)(x)
    and return all n functions in a list
    """
    # P = [torch.ones_like(x), x]  # P_0(x) = 1, P_1(x) = x
    P = [1.0, x]  # P_0(x) = 1, P_1(x) = x
    for i in range(1, n):
        P_i_plus_one = ((2 * i + 1) * x * P[i] - i * P[i - 1]) / (i + 1)
        P.append(P_i_plus_one)
    return P


def chebyshev_poly_recurrence(x, n):
    """
    Compute the Chebyshev polynomials (first kind) up to degree n 
    using the recursion formula T_(n+1)(x) = 2xT_n(x) - T_(n-1)(x),
    and return all n functions in a list
    """
    # T = [torch.ones_like(x), x]  # T_0(x) = 1, T_1(x) = x
    T = [1.0, x]
    for i in range(1, n):
        T_i_plus_1 = 2*x*T[i] - T[i-1]
        T.append(T_i_plus_1)
    return T







def solve_Theta_max(g_ij_matrix = None, g_i_list = None, is_x_dependent = False, x = None, style = "Legendre"):
    """ 
    Given g_ij_matrix or g_i_list, solve Theta_max that satisfies constraint
    \int_0^Theta_max g(Theta,x) dTheta = 3
    Plug-in expression of g(Theta, x), we have 
    Theta_max/2 * 10**-2 * \sum_{i=0}^n g_i(x) \int_-1^1 ((y+1)/2)**2 * P_i(y) dy  = 3
    we need to evaluate \int_-1^1 ((y+1)/2)**2 * P_i(y) dy  analytically, and this can be done by decomposing ((y+1)/2)**2  into P_i s:
    ((y+1)/2)**2 = 1/6 P_2 + 1/2 P_1 + 1/3 P_0, where P_2 = 3/2 y**2 - 1/2, P_1 = y, P_0 = 1
    For Legendre polynomials, according to its orthonomality, we have
    \int_-1^1 P_n*P_m dy = 0 if n!=m,  or 2/(2*n+1) if n=m
    With this, we can re-write the integration 
    Theta_max/2 * 10**-2 * \sum_{i=0}^n g_i(x) \int_-1^1 ((y+1)/2)**2 * P_i(y) dy  = 3
    as 
    Theta_max/2 * 10**-2 * \int_-1^1 (1/6 P_2 + 1/2 P_1 + 1/3 P_0) * \sum_{i=0}^n g_i(x) P_i(y) dy  = 3
    Therefore
    Theta_max/2 * 10**-2 * \int_-1^1 (1/6*g_2(x)*P_2**2 + 1/2*g_1(x)*P_1**2 + 1/3*g_0(x)*P_0**2) dy  = 3
    We have
    \int_-1^1 P_0**2 dy = 2, \int_-1^1 P_1**2 dy = 2/3, \int_-1^1 P_2**2 dy = 2/5
    Therefore we have
    Theta_max/2 * 10**-2 *  (1/6*g_2(x)*2/5 + 1/2*g_1(x)*2/3 + 1/3*g_0(x)*2)   = 3
    i.e.
    Theta_max/2 * 10**-2 *  (1/15*g_2(x) + 1/3*g_1(x) + 2/3*g_0(x)) = 3
    i.e.  we can solve Theta_max analytically as
    Theta_max  = 600/( 1/15*g_2(x) + 1/3*g_1(x) + 2/3*g_0(x) ) 
    
    """
    if is_x_dependent == True:
        # convert g_ij_matrix into g_i_list given x
        g_i_list = []
        n_i = len(g_ij_matrix)
        n_j = len(g_ij_matrix[0])
        for i in range(0, n_i):  
            _t = 2*x-1
            if style == "Legendre":
                Pn_values = legendre_poly_recurrence(_t, n_j-1) 
            elif style == "Chebyshev":
                Pn_values = chebyshev_poly_recurrence(_t, n_j-1)
            g_i = 0.0
            for j in range(0, n_j):
                g_i = g_i + g_ij_matrix[i][j]*Pn_values[j]
            g_i_list.append(g_i)
    else:
        pass
    # Now see if g_i_list is long enough
    if len(g_i_list) == 1:
        # only have g_0 term, we say g_1 and g_2 are 0
        g_i_list.append(0.0)
        g_i_list.append(0.0)
    elif len(g_i_list) == 2:
        # only have g_0 and g_1 term, we say g_2 is 0
        g_i_list.append(0.0)
    ## Now we can do integration
    Theta_max = 600.0/( 1/15*g_i_list[2] + 1/3*g_i_list[1] + 2/3*g_i_list[0] ) 
    return Theta_max




def _PDOS_evaluator(Theta, g_ij_matrix = None, g_i_list = None, Theta_max = None, is_x_dependent = False, x = None, style = "Legendre"):
    """ 
    evaluate \sum_{i=0}^n g_i(x) P_i(2*Theta/Theta_max-1)  value at given Theta & x
    
    PDOS expression is 
    g(Theta, x) = 10**-2 * (Theta/Theta_max)**2 * \sum_{i=0}^n (\sum_{j=0}^m g_ij P_j(2*x-1)) P_i(2*Theta/Theta_max-1)
    When PDOS is x-independent, it becomes
    g(Theta) = 10**-2 * (Theta/Theta_max)**2 * \sum_{i=0}^n g_i P_i(2*Theta/Theta_max-1)
    
    Explaination: 
    PDOS is orignally 
    g(omega) = (omega/omega_max)**2 * \sum_{i=0}^n g_i(x) P_i(2*omega/omega_max-1), 0<omega<omega_max, 
    the (omega/omega_max)**2 prefactor ensures g(omega) \propto omega**2 at low frequencies,
    the 10**-2 *  prefactor ensures g_i is on the order of O(1)
    and the constraint to g(omega) is \int_0^omega_max g(omega) domega = 3N_A
    Now, define Theta = h_bar/kB*omega, where h_bar = h/2pi = 6.62607015*10**-34, kB = 1.380649*10**-23
    we have 
    kB/h_bar * \int_0^omega_max 10**-2 * (Theta/Theta_max)**2 * \sum_{i=0}^n g_i(x) P_i(2*Theta/Theta_max-1) dh_bar/kB*omega = 3N_A
    kB/(h_bar*N_A) * \int_0^Theta_max (Theta/Theta_max)**2 * \sum_{i=0}^n g_i(x) P_i(2*Theta/Theta_max-1) dTheta = 3
    redefine g_i(x) = kB/(h_bar*N_A) * g_i(x), i.e. absorbing kB/(h_bar*N_A) into g_i, 
    we have the constraint to be 
    \int_0^Theta_max 10**-2 * (Theta/Theta_max)**2 * \sum_{i=0}^n g_i(x) P_i(2*Theta/Theta_max-1) dTheta = 3
    ________________
    Input Parameters:
    Theta: value of frequency, scaled by h/k_B (i.e. Theta = h*omega/kB)
    g_ij_matrix: contains all params for g_ij (only if when is_x_dependent = True)
    g_i_list: contains all params for g_i (only if when is_x_dependent = False)
    Theta_max: maximum value of Theta, subject to constraint \int_0^Theta_max g(Theta) dTheta = 3. Beyond which, we assume g(Theta) is always 0
    is_x_dependent: whether the PDOS expression is Li-filling-fraction (x) dependent
    x: value of Li-filling-fraction if PDOS is x dependent
    style: polynomial style of P_j if PDOS is x dependent, can be Legendre (default) or Chebyshev
    """
    if is_x_dependent == False:
        """ e.g. Li metal anode """
        _t = 2*Theta/Theta_max -1 # scale [0, Theta_max] to [-1,1]
        if style == "Legendre":
            Pn_values = legendre_poly_recurrence(_t, len(g_i_list)-1) 
        elif style == "Chebyshev":
            Pn_values = chebyshev_poly_recurrence(_t, len(g_i_list)-1)
        # calculate g(Theta)
        g = 0.0
        # print(Theta_Li)
        for i in range(0, len(g_i_list)):
            g = g + g_i_list[i] *Pn_values[i]   
    else:
        """ LixHM """
        n_i = len(g_ij_matrix)
        n_j = len(g_ij_matrix[0])
        # first calculate the value of g_i, i.e. summing over j
        g_i_value_list = []        
        for i in range(0, n_i):  
            _t = 2*x -1
            if style == "Legendre":
                Pn_values = legendre_poly_recurrence(_t, n_j-1) 
            elif style == "Chebyshev":
                Pn_values = chebyshev_poly_recurrence(_t, n_j-1)
            g_i = 0.0
            for j in range(0, n_j):
                g_i = g_i + g_ij_matrix[i][j]*Pn_values[j]
            g_i_value_list.append(g_i)
        # now summing over i
        g = 0.0
        _t = 2*Theta/Theta_max -1
        if style == "Legendre":
            Pn_values = legendre_poly_recurrence(_t, n_i-1) 
        elif style == "Chebyshev":
            Pn_values = chebyshev_poly_recurrence(_t, n_i-1)
        # calculate g(Theta)
        g = 0.0
        for i in range(0, n_i):
            g = g + g_i_value_list[i] *Pn_values[i]   
    ## ensuring that at low freqencies, g is propotional to omega squared
    g = g *1.0
    return g





def _S_vib_PDOS_quaderature_model(g_ij_matrix = None, g_i_list = None, Theta_max = None, quaderature_points=10, is_x_dependent = False, x = None, T = 320, style = "Legendre"):
    """ 
    calculate S_vib given the expression of g(Theta) (define Theta = h*omega/kB)
    S_vib = k_B \int_0^\infty [Theta/T * 1/(exp(Theta/T) -1) - log(1-exp(-Theta/T)) ] g(omega) domega
    where g(omega) is PDOS
    We use Theta_max to approximate \infty, i.e. beyond Theta_max g(Theta) is always 0
    S_vib = R* \int_0^\Theta_max [Theta/T * 1/(exp(Theta/T) -1) - log(1-exp(-Theta/T)) ] g(Theta) dTheta
    S_vib = R* \int_0^\Theta_max [Theta/T * 1/(exp(Theta/T) -1) - log(1-exp(-Theta/T)) ] * 10**-2 *  (Theta/Theta_max)**2 * \sum_{i=0}^n g_i P_i(2*Theta/Theta_max-1)  dTheta
    we use Gauss-Legendre quaderature to do this integration, 
    define y = 2*Theta/Theta_max - 1, i.e. Theta = Theta_max*(y+1)/2
    we have 
    S_vib = R* Theta_max/2 * 10**-2 *  \int_-1^1 [Theta_max*(y+1)/(2*T) * 1/(exp(Theta_max*(y+1)/(2*T)) -1) - log(1-exp(-Theta_max*(y+1)/(2*T))) ] * ((y+1)/2)**2 * \sum_{i=0}^n g_i P_i(y)  dy
    The sum of g_i*Pi part of g(Theta) can be evaluated with _PDOS_evaluator
    Theta_max can be evaluated with solve_Theta_max 

    
    """
    ys, weights = np.polynomial.legendre.leggauss(quaderature_points)
    ys = torch.from_numpy(ys.astype("float32"))
    weights = torch.from_numpy(weights.astype("float32"))
    H_vib = 0.0  
    for i in range(0, len(ys)):
        y_now = ys[i] 
        weight_now = weights[i]
        Theta = Theta_max/2*(1+y_now)
        if is_x_dependent == False:
            """ e.g. Li metal anode """
            g_omega_now = _PDOS_evaluator(Theta, g_i_list = g_i_list, Theta_max = Theta_max, is_x_dependent = False, style = style)
        else:
            """ LixHM """
            g_omega_now = _PDOS_evaluator(Theta, g_ij_matrix = g_ij_matrix, Theta_max = Theta_max, is_x_dependent = True, x=x, style = style)
        # S_vib = R* Theta_max/2 * 10**-2 *  \int_-1^1 [Theta_max*(y+1)/(2*T) * 1/(exp(Theta_max*(y+1)/(2*T)) -1) - log(1-exp(-Theta_max*(y+1)/(2*T))) ] * ((y+1)/2)**2 * \sum_{i=0}^n g_i P_i(y)  dy
        f_of_x = (Theta_max*(y_now+1)/(2*T) * 1/(torch.exp(Theta_max*(y_now+1)/(2*T)) -1) - torch.log(1-torch.exp(-Theta_max*(y_now+1)/(2*T)))) * ((y_now+1)/2)**2 * g_omega_now
        H_vib = H_vib + 8.314 * Theta_max/2* 10**-2 *  weight_now*f_of_x     
    return H_vib



def calculate_S_vib_total_PDOS(x, T, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_LixHM, Theta_max_HM, Theta_max_Li, quaderature_points = 20, style="Legendre"):
    """
    Corresponding S_vib of the reaction
    we need to satisfy dH/dT = TdS/dT, this is guaranteed by using the same set of PDOS params of LixHM, HM and Li
    
    x is the filling fraction
    T is the temperature
    n_list is how many moles of atoms are there in 1 mole of substance, the first element is wrong (should be deduced from the other two elements)
    g_ij_list_LixHM is the PDOS params for LixHM
    g_i_list_HM is the PDOS params for HM
    g_i_list_Li is the PDOS params for Li metal 
    Theta_max_LixHM, Theta_max_HM and Theta_max_Li should be solved outside and input here (Theta_max_Li and Theta_max_HM is always unchanged, so solve it at the beginning of training to save compute)
    quaderature_points: number of sampling points in Gauss-Legendre quaderatures 
    style: polynomial style, can be Legendre or Chebyshev
    """
    S_vib = 0.0
    ## we have s_excess = s_LixHM - s_HM - x*s_Li
    ## LixHM: there is 1 mole of HM, and x mole of Li
    S_vib = S_vib + (1.0*n_list[1] + x*n_list[2])* _S_vib_PDOS_quaderature_model(x=x, T=T, g_ij_matrix = g_ij_list_LixHM, Theta_max = Theta_max_LixHM, quaderature_points=quaderature_points, is_x_dependent = True, style = style)
    ## HM: there is 1 mole of HM
    S_vib = S_vib - (1.0*n_list[1])* _S_vib_PDOS_quaderature_model(T=T, g_i_list = g_i_list_HM, Theta_max = Theta_max_HM, quaderature_points=quaderature_points, is_x_dependent = False, style = style)
    ## Li: there is x mole of Li
    S_vib = S_vib - (x*n_list[2])*  _S_vib_PDOS_quaderature_model(T=T, g_i_list = g_i_list_Li, Theta_max = Theta_max_Li, quaderature_points=quaderature_points, is_x_dependent = False, style = style)
    return S_vib




def _H_vib_PDOS_quaderature_model(g_ij_matrix = None, g_i_list = None, Theta_max = None, quaderature_points=10, is_x_dependent = False, x = None, T = 320, style = "Legendre"):
    """" 
    calculate H_vib given the expression of g(Theta)
    H_vib = \int_0^\infty [0.5*h*omega + h*omega/(exp(h*omega/(kB*T)) -1) ] g(omega) domega
    where g(omega) is PDOS
    To use Gauss-Legendre Quad, we use Theta_max to approximate \infty, i.e. beyond Theta_max g(Theta) is always 0
    define Theta = h*omega/kB, we have
    H_vib = \int_0^\Theta_max k_B [0.5*h*omega/k_B + h*omega/k_B/(exp(h*omega/(kB*T)) -1) ] g(omega)*k_B/h dh*omega/k_B
    H_vib = \int_0^\Theta_max k_B [0.5*Theta + Theta/(exp(Theta/T) -1) ] g(omega)*k_B/h dTheta
    express PDOS using Theta, and absorb *k_B/(h*N_A) into g(omega) (i.e. scale g_ij) which is consistent with the definition of g(Theta): 
    H_vib = R * \int_0^\Theta_max [0.5*Theta + Theta/(exp(Theta/T) -1) ]* g(Theta) dTheta
    H_vib = R * \int_0^\Theta_max [0.5*Theta + Theta/(exp(Theta/T) -1) ]* 10**-2 *  (Theta/Theta_max)**2 * \sum_{i=0}^n g_i P_i(2*Theta/Theta_max-1)  dTheta
    we use Gauss-Legendre quaderature to do this integration, 
    define y = 2*Theta/Theta_max - 1, i.e. Theta = Theta_max*(y+1)/2
    we have 
    H_vib = R * Theta_max**2/2* 10**-2 *  \int_-1^1 [(y+1)/4 + (y+1)/2 * 1/(exp(Theta_max*(y+1)/(2*T)) -1) ]* ((y+1)/2)**2 * \sum_{i=0}^n g_i P_i(y)  dy
    The sum of g_i*Pi part of g(Theta) can be evaluated with _PDOS_evaluator
    Theta_max can be evaluated with solve_Theta_max 
    
    Input:
    g_ij_matrix: contains all params for g_ij (only if when is_x_dependent = True)
    g_i_list: contains all params for g_i (only if when is_x_dependent = False)
    Theta_max: solved Theta_max given PDOS parameters (solved outside). Beyond which, we assume g(Theta) is always 0
    is_x_dependent: whether the PDOS expression is Li-filling-fraction (x) dependent
    x: value of Li-filling-fraction if PDOS is x dependent
    T: temperature
    style: polynomial style of P_j if PDOS is x dependent, can be Legendre (default) or Chebyshev
    """
    ys, weights = np.polynomial.legendre.leggauss(quaderature_points)
    ys = torch.from_numpy(ys.astype("float32"))
    weights = torch.from_numpy(weights.astype("float32"))
    H_vib = 0.0  
    for i in range(0, len(ys)):
        y_now = ys[i] 
        weight_now = weights[i]
        Theta = Theta_max/2*(1+y_now)
        if is_x_dependent == False:
            """ e.g. Li metal anode """
            g_omega_now = _PDOS_evaluator(Theta, g_i_list = g_i_list, Theta_max = Theta_max, is_x_dependent = False, style = style)
        else:
            """ LixHM """
            g_omega_now = _PDOS_evaluator(Theta, g_ij_matrix = g_ij_matrix, Theta_max = Theta_max, is_x_dependent = True, x=x, style = style)
        # H_vib = R * Theta_max**2/2* 10**-2 *  \int_-1^1 [(y+1)/4 + (y+1)/2 * 1/(exp(Theta_max*(y+1)/(2*T)) -1) ]* ((y+1)/2)**2 * \sum_{i=0}^n g_i P_i(y)  dy
        f_of_x = ((y_now+1)/4 + (y_now+1)/2 * 1/(torch.exp(Theta_max*(y_now+1)/(2*T)) -1))* ((y_now+1)/2)**2 * g_omega_now
        H_vib = H_vib + 8.314 * Theta_max**2/2* 10**-2 *  weight_now*f_of_x     
    return H_vib


def calculate_H_vib_total_PDOS(x, T, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_LixHM, Theta_max_HM, Theta_max_Li, quaderature_points = 20, style="Legendre"):
    """
    Corresponding H_vib of the reaction
    we need to satisfy dH/dT = TdS/dT, this is guaranteed by using the same set of PDOS params of LixHM, HM and Li
    
    x is the filling fraction
    T is the temperature
    n_list is how many moles of atoms are there in 1 mole of substance, the first element is wrong (should be deduced from the other two elements)
    g_ij_list_LixHM is the PDOS params for LixHM
    g_i_list_HM is the PDOS params for HM
    g_i_list_Li is the PDOS params for Li metal 
    Theta_max_LixHM, Theta_max_HM and Theta_max_Li should be solved outside and input here (Theta_max_Li and Theta_max_HM is always unchanged, so solve it at the beginning of training to save compute)
    quaderature_points: number of sampling points in Gauss-Legendre quaderatures 
    style: polynomial style, can be Legendre or Chebyshev
    """
    H_vib = 0.0 
    ## we have h_excess = h_LixHM - h_HM - h*s_Li
    ## LixHM: there is 1 mole of HM, and x mole of Li
    H_vib = H_vib + (1.0*n_list[1] + x*n_list[2])* _H_vib_PDOS_quaderature_model(x=x, T=T, g_ij_matrix = g_ij_list_LixHM, Theta_max = Theta_max_LixHM, quaderature_points=quaderature_points, is_x_dependent = True, style = style)
    ## HM: there is 1 mole of HM
    H_vib = H_vib - (1.0*n_list[1])* _H_vib_PDOS_quaderature_model(T=T, g_i_list = g_i_list_HM, Theta_max = Theta_max_HM, quaderature_points=quaderature_points, is_x_dependent = False, style = style)
    ## Li: there is x mole of Li
    H_vib = H_vib - (x*n_list[2])*  _H_vib_PDOS_quaderature_model(T=T, g_i_list = g_i_list_Li, Theta_max = Theta_max_Li, quaderature_points=quaderature_points, is_x_dependent = False, style = style)
    return H_vib


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



def GibbsFE_PDOS(x, T, params_list = total_params_list, style = style, quaderature_points = 20):
    """
    Expression for Delta Gibbs Free Energy of charging / discharging process
    Delta_G = H_mix + H_vib - T(S_config + S_vib)
    _____
    Input params:
    x: Li-filling fraction
    T: temperature (Kelvin)
    params_list: in the sequence of [enthalpy_mixing_params_list, config_entropy_params_list, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_HM_value, Theta_max_Li_value]
    style: style of polynomial, can be "Legendre" (default) or "Chebyshev"
    quaderature_points: number of sampled points when doing Gauss-Legendre quaderature
    """
    if style == "Legendre":
        poly_eval_function = legendre_poly_recurrence
    elif style == "Chebyshev":
        poly_eval_function = chebyshev_poly_recurrence
    else:
        print("ERROR: polynomial style %s not supported." %(style))
        exit()
    enthalpy_mixing_params_list = params_list[0]
    config_entropy_params_list = params_list[1]
    n_list = params_list[2]
    g_ij_list_LixHM = params_list[3]
    g_i_list_HM = params_list[4]
    g_i_list_Li = params_list[5]
    Theta_max_HM = params_list[6]
    Theta_max_Li = params_list[7]
    # need to solve Theta_max_LixHM from g_ij_list_LixHM
    x = torch.clamp(x, min=_eps, max=1.0-_eps)
    Theta_max_LixHM = solve_Theta_max(g_ij_matrix = g_ij_list_LixHM, is_x_dependent = True, x = x, style = style)
    # H_mix
    G0 = enthalpy_mixing_params_list[-1]
    G = x*G0 + (1-x)*0.0 
    t = 2 * x -1 # Transform x to (2x-1) for legendre expansion
    Pn_values = poly_eval_function(t, len(enthalpy_mixing_params_list)-2)  # Compute Legendre polynomials up to degree len(coeffs) - 1 # don't need to get Pn(G0)
    for i in range(0, len(enthalpy_mixing_params_list)-1):
        G = G + x*(1-x)*(enthalpy_mixing_params_list[i]*Pn_values[i])
    # S_mix (S_ideal + S_excess)
    # S_ideal
    G = G - T*(-8.314)*(x*torch.log(x)+(1-x)*torch.log(1-x)) 
    # S_excess, i.e. excess configurational entropy
    t = 2 * x -1 # Transform x to (2x-1) for legendre expansion
    Pn_values = poly_eval_function(t, len(config_entropy_params_list)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1 
    for i in range(0, len(config_entropy_params_list)):
        G = G - T*(-8.314)*(x*torch.log(x)+(1-x)*torch.log(1-x))*(config_entropy_params_list[i]*Pn_values[i])
    # S_vib, i.e. vibrational entropy
    G = G - T*calculate_S_vib_total_PDOS(x, T, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_LixHM, Theta_max_HM, Theta_max_Li, quaderature_points = quaderature_points, style=style)
    # H_vib, i.e. vibrational enthalpy corresponds to the vibrational entropy
    # we need to satisfy dH/dT = T*dS/dT
    G = G + calculate_H_vib_total_PDOS(x, T, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_LixHM, Theta_max_HM, Theta_max_Li, quaderature_points = quaderature_points, style=style)
    return G