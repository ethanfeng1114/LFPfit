import numpy as np
import torch
import torch.nn as nn
from torch import autograd



global  _eps
_eps = 1e-7



def sampling(GibbsFunction, params_list, T, end_points, sampling_id, ngrid=99, requires_grad = False, is_PDOS_version_G = False, style = "Legendre", quaderature_points = 20):
    """
    Sampling a Gibbs free energy function (GibbsFunction)
    sampling_id is for recognition, must be a interger
    """
    x = np.concatenate((np.array([end_points[0]+_eps]),np.linspace(end_points[0]+1/(ngrid+1),end_points[1]-1/(ngrid+1),ngrid),np.array([end_points[1]-_eps]))) 
    x = torch.from_numpy(x.astype("float32"))
    x = x.requires_grad_()
    if is_PDOS_version_G == False:
        sample = torch.tensor([[x[i], GibbsFunction(x[i], params_list, T), sampling_id] for i in range(0, len(x))])
    else:
        sample = torch.tensor([[x[i], GibbsFunction(x[i], T, params_list, style = style, quaderature_points = quaderature_points), sampling_id] for i in range(0, len(x))])
    return sample



def convex_hull(sample, ngrid=99, tolerance = _eps):
    """
    Convex Hull Algorithm that provides the initial guess for common tangent
    Need NOT to be differentiable
    returning the initial guess for common tangent & corresponding phase id
    Adapted from Pinwe's Jax-Thermo with some modifications
    """
    # convex hull, starting from the furtest points at x=0 and 1 and find all pieces
    base = [[sample[0,:], sample[-1,:]]]
    current_base_length = len(base) # currently len(base) = 1
    new_base_length = 9999999
    base_working = base.copy()
    n_iter = 0
    while new_base_length != current_base_length:
        n_iter = n_iter + 1
        # save historical length of base, for comparison at the end
        current_base_length = len(base)
        # continue the convex hull pieces construction until we find all pieces
        base_working_new=base_working.copy()
        for i in range(len(base_working)):   # len(base_working) = 1 at first, but after iterations on n, the length of this list will be longer
            # the distance of sampling points to the hyperplane formed by base vector
            # sample[:,column]-h[column] calculates the x and y distance for all sample points to the base point h
            # 0:2 deletes the sampling_id
            # t[column]-h[column] is the vector along the hyperplane (line in 2D case)
            # dot product of torch.tensor([[0.0,-1.0],[1.0,0.0]]) and t[column]-h[column] calculates the normal vector of the hyperplane defined by t[column]-h[column]
            h = base_working[i][0]; t = base_working[i][1] # h is the sample point at left side, t is the sample point at right side
            _n = torch.matmul(torch.from_numpy(np.array([[0.0,-1.0],[1.0,0.0]]).astype("float32")), torch.reshape((t[0:2]-h[0:2]), (2,1)))
            # limit to those having x value between the x value of h and t
            left_id = torch.argmin(torch.abs(sample[:,0]-h[0])) + 1 # limiting the searching range within h and t
            right_id = torch.argmin(torch.abs(sample[:,0]-t[0]))
            if left_id == right_id: # it means this piece of convex hull is the shortest piece possible
                base_working_new.remove(base_working[i])
            else:
                # it means it's still possible to make this piece of convex hull shorter
                sample_current = sample[left_id:right_id, :] 
                _t = sample_current[:,0:2]-h[0:2]
                dists = torch.matmul(_t, _n).squeeze()
                if dists.shape == torch.Size([]): # in case that there is only 1 item in dists, .squeeze wil squeeze ALL dimension and make dists a 0-dim tensor
                    dists = torch.tensor([dists])
                # select those underneath the hyperplane
                outer = []
                for _ in range(0, sample_current.shape[0]):
                    if dists[_] < -_eps: 
                        outer.append(sample_current[_,:]) 
                # if there are points underneath the hyperplane, select the farthest one. If no outer points, then this set of working base is dead
                if len(outer):
                    pivot = sample_current[torch.argmin(dists)] # the furthest node below the hyperplane defined hy t[column]-h[column]
                    # after find the furthest node, we remove the current hyperplane and rebuild two new hyperplane
                    z = 0
                    while (z<=len(base)-1):
                        # i.e. finding the plane corresponding to the current working plane
                        diff = torch.max(  torch.abs(torch.cat((base[z][0], base[z][1])) - torch.cat((base_working[i][0], base_working[i][1])))  )
                        if diff < tolerance:
                            # remove this plane
                            base.pop(z) # The pop() method removes the item at the given index from the list and returns the removed item.
                        else:
                            z=z+1
                    # the furthest node below the hyperplane is picked up to build two new facets with the two corners 
                    base.append([h, pivot])
                    base.append([pivot, t])
                    # update the new working base
                    base_working_new.remove(base_working[i])
                    base_working_new.append([h, pivot])
                    base_working_new.append([pivot, t])
                else:
                    base_working_new.remove(base_working[i])
        base_working=base_working_new
        # update length of base
        new_base_length = len(base)
    # find the pieces longer than usual. If for a piece of convex hull, the length of it is longer than delta_x
    delta_x = 1.0/(ngrid+1.0) + tolerance
    miscibility_gap_x_left_and_right = []
    miscibility_gap_phase_left_and_right = []
    for i in range(0, len(base)):
        convex_hull_piece_now = base[i]
        if convex_hull_piece_now[1][0]-convex_hull_piece_now[0][0] > delta_x:
            miscibility_gap_x_left_and_right.append(torch.tensor([convex_hull_piece_now[0][0], convex_hull_piece_now[1][0]]))
            miscibility_gap_phase_left_and_right.append(torch.tensor([convex_hull_piece_now[0][2], convex_hull_piece_now[1][2]]))
    # sort the init guess of convex hull
    left_sides = torch.zeros(len(miscibility_gap_x_left_and_right))
    for i in range(0, len(miscibility_gap_x_left_and_right)):
        left_sides[i] = miscibility_gap_x_left_and_right[i][0]
    _, index =  torch.sort(left_sides)
    miscibility_gap_x_left_and_right_sorted = []
    miscibility_gap_phase_left_and_right_sorted = []
    for _ in range(0, len(index)):
        miscibility_gap_x_left_and_right_sorted.append(miscibility_gap_x_left_and_right[_])
        miscibility_gap_phase_left_and_right_sorted.append(miscibility_gap_phase_left_and_right[_])
    return miscibility_gap_x_left_and_right_sorted, miscibility_gap_phase_left_and_right_sorted    



class FixedPointOperation(nn.Module):
    def __init__(self, G, params_list, T = 300, is_PDOS_version_G = False, end_points = [0,1], scaling_alpha = 1e-5, style = "Legendre", quaderature_points = 20):
        """
        The fixed point operation used in the backward pass of common tangent approach. 
        Write the forward(self, x) function in such weird way so that it is differentiable
        G is the Gibbs free energy function 
        params_list: depend on whether is_PDOS_version_G, its content varies
        """
        super(FixedPointOperation, self).__init__()
        self.G = G
        self.params_list = params_list
        self.T = torch.tensor([T])
        self.is_PDOS_version_G = is_PDOS_version_G
        assert end_points[1] > end_points[0]
        self.end_points = end_points
        self.scaling_alpha = scaling_alpha
        self.style = style
        self.quaderature_points = quaderature_points
    def forward(self, x):
        """x[0] is the left limit of phase coexisting region, x[1] is the right limit"""
        x_alpha = x[0]
        x_beta = x[1]
        if self.is_PDOS_version_G == False:
            g_right = self.G(x_beta, self.params_list, self.T) 
            g_left = self.G(x_alpha, self.params_list, self.T)
        else:
            g_right = self.G(x_beta, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
            g_left = self.G(x_alpha, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
        mu_right = autograd.grad(outputs=g_right, inputs=x_beta, create_graph=True)[0]
        mu_left = autograd.grad(outputs=g_left, inputs=x_alpha, create_graph=True)[0]
        """ 
        Because  (g_right - g_left)/(x_beta-x_alpha) = mu_right = mu_left,
        we have 
        (g_right - g_left)/(x_beta-x_alpha) - mu_left = 0  ==>  x_alpha_new = x_alpha + (g_right - g_left)/(x_beta-x_alpha) - mu_left  (at fixedpoint, x_alpha_new = x_alpha)
        (g_right - g_left)/(x_beta-x_alpha) - mu_right = 0  ==>  x_beta_new = x_beta + (g_right - g_left)/(x_beta-x_alpha) - mu_right  (at fixedpoint, x_beta_new = x_beta)
        """
        x_alpha_new = x_alpha + self.scaling_alpha*((g_right - g_left)/(x_beta - x_alpha + _eps) - mu_left)
        x_beta_new = x_beta + self.scaling_alpha*((g_right - g_left)/(x_beta-x_alpha + _eps) - mu_right)
        ## old implementation
        # """ 
        # Because  (g_right - g_left)/(x_beta-x_alpha) = mu_right = mu_left,
        # we have 
        # (g_right - g_left)/(x_beta-x_alpha) = mu_right ==> x_alpha = x_beta  - (g_right - g_left)/mu_right
        # (g_right - g_left)/(x_beta-x_alpha) = mu_left  ==> x_beta  = x_alpha + (g_right - g_left)/mu_left   
        # """
        # x_alpha_new = x_beta - (g_right - g_left)/(mu_left + _eps)
        # x_beta_new = x_alpha + (g_right - g_left)/(mu_right + _eps)
        # if self.is_clamp: ## dont need to clamp here! clamped in the solver
        #     x_alpha_new = torch.clamp(x_alpha_new , min=self.end_points[0]+_eps, max=self.end_points[1]-_eps) # clamp
        #     x_beta_new = torch.clamp(x_beta_new , min=self.end_points[0]+_eps, max=self.end_points[1]-_eps) # clamp
        x_alpha_new = x_alpha_new.reshape(1)
        x_beta_new = x_beta_new.reshape(1)
        return torch.cat((x_alpha_new, x_beta_new))

    
def newton_raphson(func, x0, threshold=1e-6, end_points = [0,1], is_clamp = True):
    """
    x0: initial guess, with shape torch.Size([2])
    """
    error = 9999999.9
    x_now = x0.clone()
    # define g function for Newton-Raphson
    def g(x):
        return func(x) - x
    # iteration
    n_iter = -1
    while error > threshold and n_iter < 100:
        x_now = x_now.requires_grad_() # add grad track here for the sake of autograd.functional.jacobian
        f_now = g(x_now)
        J = autograd.functional.jacobian(g, x_now)
        f_now = torch.reshape(f_now, (2,1)) 
        x_new = x_now - torch.reshape(torch.linalg.pinv(J)@f_now, (2,)) 
        # detach for memory saving
        x_new = x_new.clone().detach() # detach for memory saving
        # clamp, actually it MUST be clamped
        if is_clamp:
            x_new[0] = torch.max(torch.tensor([end_points[0]+_eps, x_new[0]]))
            x_new[0] = torch.min(torch.tensor([end_points[1]-_eps, x_new[0]]))
            x_new[1] = torch.max(torch.tensor([end_points[0]+_eps, x_new[1]]))
            x_new[1] = torch.min(torch.tensor([end_points[1]-_eps, x_new[1]])) # +- 1e-6 is for the sake of torch.log. We don't want log function to blow up at x=0!
        x_now = x_now.clone().detach() # detach for memory saving
        # calculate error
        if torch.abs(x_new[0]-x_now[0]) < torch.abs(x_new[1]-x_now[1]):
            error = torch.abs(x_new[1]-x_now[1])
        else:
            error = torch.abs(x_new[0]-x_now[0])
        # step forward
        x_now = x_new.clone()
        n_iter = n_iter + 1
    if n_iter >= 99:
        print("Warning: Max iteration in Newton-Raphson solver reached.")
    return x_now





class FixedPointOperation1D(nn.Module):
    def __init__(self, G, params_list, T = 300, x_fixed_at_endpoint_ID = 0, is_PDOS_version_G = True, scaling_alpha = 1e-5, style = "Legendre", quaderature_points = 20):
        """
        The fixed point operation used in the backward pass of common tangent approach, but only one point is changing (one of them is endpoint and thus fixed)
        Write the forward(self, x) function in such weird way so that it is differentiable
        G is the Gibbs free energy function 
        params_list: depend on whether is_PDOS_version_G, its content varies
        x_fixed_at_endpoint_ID: the index of x that is fixed
        """
        super(FixedPointOperation1D, self).__init__()
        self.G = G
        self.params_list = params_list
        self.T = T
        self.is_PDOS_version_G = is_PDOS_version_G
        self.style = style
        self.quaderature_points = quaderature_points
        self.x_fixed_at_endpoint_ID = x_fixed_at_endpoint_ID
        self.scaling_alpha = scaling_alpha
        self.is_gradient_tracking_in_ct = False
    def forward(self, x):
        """x[self.x_fixed_at_endpoint_ID] is the fixed endpoint, the other one is the one can change"""
        x_fixed_at_end_point = x[self.x_fixed_at_endpoint_ID]
        x_can_move = x[1-self.x_fixed_at_endpoint_ID]
        if self.is_PDOS_version_G == False:
            raise NotImplementedError
        else:
            g_fixed_at_end_point = self.G(x_fixed_at_end_point, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
            g_c =                  self.G(x_can_move,           self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
        mu_c = autograd.grad(outputs=g_c, inputs=x_can_move, create_graph=True)[0]
        # """ we have xc = x_fixed_at_end_point - (G(x_fixed_at_end_point)-G(xc))/mu(xc)  """
        # x_moved = x_fixed_at_end_point - (g_fixed_at_end_point-g_c)/mu_c
        """ 
        or we just have xc = xc + 1e-5 * ( (G(x_fixed_at_end_point)-G(xc))/(x_fixed_at_end_point-x_c) - mu(x_c)) 
        1e-5 is for numeric stability TEST
        """
        x_moved = x_can_move +  self.scaling_alpha * ((g_fixed_at_end_point-g_c)/(x_fixed_at_end_point-x_can_move) - mu_c )
        if self.is_gradient_tracking_in_ct == False:
            x_fixed = x_fixed_at_end_point * 1.0
        else:
            """ 
            we need to track x_fixed, which can be given by re-writting common tangent
            if the quality of x_can_move is great, the calculated x_fixed should be basically the fixed point itself
            remember to clamp it so that it doesn't go over the region
            """
            x_fixed = x_fixed_at_end_point + self.scaling_alpha * ((g_fixed_at_end_point-g_c)/(x_fixed_at_end_point-x_can_move) - mu_c )
            self.end_points = [0.0, 1.0]
            x_fixed = torch.clamp(x_fixed , min=self.end_points[0]+_eps, max=self.end_points[1]-_eps) # clamp
        x_moved = x_moved.reshape(1)
        x_fixed = x_fixed.reshape(1)
        if self.x_fixed_at_endpoint_ID == 0:
            return torch.cat((x_fixed, x_moved))
        else:
            return torch.cat((x_moved, x_fixed))
    def forward_for_solver(self, x_moving_point):
        # first sef self.x_fixed_point_value outside, before using this class!
        if self.is_PDOS_version_G == False:
            raise NotImplementedError
        else:
            g_fixed_at_end_point = self.G(self.x_fixed_point_value, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
            g_c =                  self.G(x_moving_point,           self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
        mu_c = autograd.grad(outputs=g_c, inputs=x_moving_point, create_graph=True)[0]
        # """ we have xc = x_fixed_at_end_point - (G(x_fixed_at_end_point)-G(xc))/mu(xc)  """
        # x_moved = self.x_fixed_point_value - (g_fixed_at_end_point-g_c)/mu_c
        """ 
        or we just have xc = xc + 1e-5 * ( (G(x_fixed_at_end_point)-G(xc))/(x_fixed_at_end_point-x_c) - mu(x_c)) 
        1e-5 is for numeric stability TEST
        """
        x_moved = x_moving_point +  self.scaling_alpha * ((g_fixed_at_end_point-g_c)/(self.x_fixed_point_value-x_moving_point) - mu_c )
        return x_moved
    def forward_0D(self, x):
        """ reserved for the all concave case"""
        return x*1.0
    
    


def solver_1D(func, x0, ID_changing_one=0, threshold=1e-4, end_points=[0.0,1.0]):
    """
    For Solving the "common tangent" when one of the points are endpoint.
    In this case, we have 
    G
    |
    |       
    |           
    |       x           
    |   x     x                     b 
    |  a          x              x   
    |                   c
    |__________________________________ x
    In this case, connect a and c gives the convex hull, 
    i.e. we need to find the common tangent of the straight line ac and the G curve at point c
    i.e. the tangent of ac is the same as that of G curve at point c
    this is now a 1D solver, we need to solve xc, i.e.
    (G(xa)-G(xc))/(xa-xc) = mu(xc)  [Note that xa is fixed value]
    writing in fixed-point solving scheme:
    xc = xa - (G(xa)-G(xc))/mu(xc) 
    
    func: 
    x0: initial guess, with shape torch.Size([1])
    ID: which one is the changing one 
    """
    x_now = x0[ID_changing_one]*1.0
    x_fixed = x0[1-ID_changing_one]*1.0
    x_now = x_now.reshape(1)
    x_fixed = x_fixed.reshape(1)
    # define g function for Newton-Raphson
    def g(x):     
        func.x_fixed_point_value = x0[1-ID_changing_one]
        x_ = func.forward_for_solver(x)    
        return  x_ - x
    # iteration    
    n_iter_max = 1000
    is_reached_true_solution = False
    n_time_of_reaching_solution = 0
    while is_reached_true_solution == False: # sometimes it returns 
        # reset every intermediate variables
        n_iter = -1
        error = 9999999.9
        while error > threshold and n_iter < n_iter_max:
            x_now = x_now.requires_grad_() # add grad track here for the sake of autograd.functional.jacobian
            f_now = g(x_now)
            J = autograd.functional.jacobian(g, x_now)
            f_now = torch.reshape(f_now, (1,1)) 
            x_new = x_now - torch.reshape(torch.linalg.pinv(J)@f_now, (1,)) 
            # clamp, if the changing point goes outside the box
            while x_new>end_points[1] or x_new<end_points[0] or torch.isnan(x_new) or torch.isinf(x_new):
                import random
                randseed = random.uniform(-1,1)
                ## random perturbation around init point
                x_new = x0[ID_changing_one] + randseed*0.05 # real solution won't be far away from the init point
                x_new = x_new.reshape(1)
                x_new = x_new.requires_grad_()
                ## random perturbation within a half space
                # if x0[ID_changing_one]>=(end_points[0]+end_points[1])/2: 
                #     x_new = torch.tensor([end_points[1] - randseed*(end_points[0]+end_points[1])/2])
                #     x_new = x_new.requires_grad_()
                # else:
                #     x_new = torch.tensor([end_points[0] + randseed*(end_points[0]+end_points[1])/2])
                #     x_new = x_new.requires_grad_()
            # calculate error
            error = torch.abs(x_new-x_now)
            x_now = x_now.clone().detach() # detach for memory saving
            # step forward
            x_now = x_new.clone()
            n_iter = n_iter + 1
        # we have reached a solution
        n_time_of_reaching_solution = n_time_of_reaching_solution + 1
        # examine whether the solved x_now is close enough to x0
        ## don't check at all
        is_reached_true_solution = True
        # # check
        # if torch.abs(x_now-x0[ID_changing_one]) <= 0.05:
        #     is_reached_true_solution = True
        #     # print("Reached final solution. Now solution is %.4f, init sol is %.4f" %(x_now,x0[ID_changing_one]))
        # else:
        #     is_reached_true_solution = False
        #     print("%d times reach solution. Re-calculate 1D solution, now solution is %.4f, but init sol is %.4f" %(n_time_of_reaching_solution, x_now,x0[ID_changing_one]))
        #     # reset every intermediate variables
        #     n_iter = -1
        #     error = 9999999.9
        #     import random
        #     randseed = random.uniform(-1,1)
        #     x_now = x0[ID_changing_one] + randseed*0.05
        #     x_now = x_now.reshape(1)
        #     x_now = x_now.requires_grad_()
        #     print("Re-initialized as %.4f" %(x_now))
    # we find true solution now
    if ID_changing_one == 1:
        return torch.cat((x_fixed, x_now))
    else:
        return torch.cat((x_now, x_fixed))






class CommonTangent(nn.Module):
    """
    Common Tangent Approach for phase equilibrium boundary calculation
    """
    def __init__(self, G, params_list, T = 300, is_PDOS_version_G = False, end_points = [0.0,1.0], scaling_alpha = 1e-5, is_clamp = True, style = "Legendre", quaderature_points = 20, f_thres=1e-6):
        super(CommonTangent, self).__init__()
        # self.f_forward = FixedPointOperationForwardPass(G, params_list, T, is_PDOS_version_G=is_PDOS_version_G, end_points=end_points, style=style, quaderature_points = quaderature_points) # define forward operation here
        self.f = FixedPointOperation(G, params_list, T, is_PDOS_version_G=is_PDOS_version_G, end_points=end_points, scaling_alpha = scaling_alpha, style=style, quaderature_points = quaderature_points) # define backward operation here    
        self.solver = newton_raphson
        self.end_points = end_points
        self.scaling_alpha = scaling_alpha
        self.is_clamp = is_clamp
        self.f_thres = f_thres
        self.T = T
        self.G = G
        self.is_PDOS_version_G = is_PDOS_version_G
        self.style=style 
        self.quaderature_points = quaderature_points
        self.params_list = params_list
    def forward(self, x, **kwargs):
        """
        x is the initial guess provided by convex hull
        """
        if x[0]-self.end_points[0] >= _eps*2.0 and self.end_points[1]-x[1] >= _eps*2.0:
            """ 
            This means the miscibility gap does not start or end at endpoints, 
            G
            | x
            |  a    
            |      x     
            |        x          
            |         b                       x 
            |           x              x   
            |              x   x
            |__________________________________ x
            
            this is the normal situation where we can apply common tangent
            to solve the position of a and b
            """
            # Forward pass
            x_star = self.solver(self.f, x, threshold=self.f_thres, end_points = self.end_points, is_clamp=self.is_clamp) # use newton-raphson to get the fixed point
            # if torch.any(torch.isnan(x_star)) == True: # in case that the previous one doesn't work
            #     print("Fixpoint solver failed at T = %d. Use traditional approach instead" %(self.T))
            #     x_star = self.f_forward(x)
            # (Prepare for) Backward pass
            new_x_star = self.f(x_star.requires_grad_()) # go through the process again to get derivative
            # register hook, can do anything with the grad that passed in
            def backward_hook(grad):
                # we use this hook to calculate dz/dtheta, where z is the equilibrium and theta is the learnable params in the model
                if self.hook is not None:
                    self.hook.remove()
                    # torch.cuda.synchronize()   # To avoid infinite recursion
                """
                Compute the fixed point of y = yJ + grad, 
                where y is the new_grad, 
                J=J_f is the Jacobian of f at z_star, 
                grad is the input from the chain rule.
                From y = yJ + grad, we have (I-J)y = grad, so y = (I-J)^-1 grad
                """
                # # Original implementation by Shaojie Bai, DEQ https://github.com/locuslab/deq:
                # new_grad = self.solver(lambda y: autograd.grad(new_x_star, x_star, y, retain_graph=True)[0] + grad, torch.zeros_like(grad), threshold=self.f_thres, in_backward_hood=True)
                # AM Yao: use inverse jacobian as we aren't solving a large matrix here
                I_minus_J = torch.eye(2) - autograd.functional.jacobian(self.f, x_star)
                new_grad = torch.linalg.pinv(I_minus_J)@grad
                return new_grad
            # hook registration
            self.hook = new_x_star.register_hook(backward_hook)
            # all set! return 
            return new_x_star
        else:
            if x[0]-self.end_points[0] < _eps*2.0 and self.end_points[1]-x[1] < _eps*2.0:
                """ 
                this means this whole region is concave, 
                G
                |
                |       
                |             x
                |       x           x
                |   x                       x
                | a                             b
                |
                |__________________________________ x
                In this case, just connecting a and b will give the final convex hull (instead of getting common tangent for a and b)
                just return end_points and Gibbs free energy is minimized
                """
                print("WARNING: WHOLE REGION COMPLETELY CONCAVE, consider re-initialize parameters!!!")
                # pseudo-gradient here, it's all 0
                f_now = FixedPointOperation1D(self.G, self.params_list, self.T, x_fixed_at_endpoint_ID = -99999, is_PDOS_version_G=self.is_PDOS_version_G, style=self.style, quaderature_points = self.quaderature_points)
                new_x_star = f_now.forward_0D(torch.tensor(self.end_points).requires_grad_()) # go through the process again to get derivative
                # register hook, can do anything with the grad that passed in
                def backward_hook(grad):
                    # we use this hook to calculate dz/dtheta, where z is the equilibrium and theta is the learnable params in the model
                    if self.hook is not None:
                        self.hook.remove()
                        # torch.cuda.synchronize()   # To avoid infinite recursion
                    zeros = torch.eye(2) - torch.eye(2)
                    new_grad = zeros@grad
                    return new_grad
                # hook registration
                self.hook = new_x_star.register_hook(backward_hook)
                # all set! return 
                return new_x_star
            else:
                """ 
                when one of these things are end points, we do not have common tangent condition for g curve itself, 
                instead we just find out where is the gibbs free energy minimum place as one of the phase boundary points that is not endpoints
                G
                |
                |       
                |           
                |       x           
                |   x     x                     b 
                |  a          x              x   
                |                   c
                |__________________________________ x
                In this case, connect a and c gives the convex hull, instead of finding two points on g curve that give the same derivative value
                i.e. we need to find the common tangent of the straight line ac and the G curve at point c
                i.e. the tangent of ac is the same as that of G curve at point c
                this is now a 1D solver, we need to solve xc, i.e.
                (G(xa)-G(xc))/(xa-xc) = mu(xc)  [Note that xa is fixed value]
                writing in fixed-point solving scheme:
                xc = xa - (G(xa)-G(xc))/mu(xc)
                """
                #E: from .solver import solver_1D
                if x[0]-self.end_points[0] < _eps*2.0 and self.end_points[1]-x[1] >= _eps*2.0:
                    f_now = FixedPointOperation1D(self.G, self.params_list, self.T, x_fixed_at_endpoint_ID = 0, is_PDOS_version_G=self.is_PDOS_version_G, scaling_alpha = self.scaling_alpha, style=self.style, quaderature_points = self.quaderature_points)
                    x_star = solver_1D(f_now, x, ID_changing_one=1) # threshold cannot be too large, because our quaderature also has error!  
                else:
                    f_now = FixedPointOperation1D(self.G, self.params_list, self.T, x_fixed_at_endpoint_ID = 1, is_PDOS_version_G=self.is_PDOS_version_G, scaling_alpha = self.scaling_alpha, style=self.style, quaderature_points = self.quaderature_points) 
                    x_star = solver_1D(f_now, x, ID_changing_one=0) # threshold cannot be too large, because our quaderature also has error!
                # (Prepare for) Backward pass    
                f_now.is_gradient_tracking_in_ct = True
                new_x_star = f_now(x_star.requires_grad_()) # go through the process again to get derivative
                f_now.is_gradient_tracking_in_ct = False
                ## checking
                if x[0]-self.end_points[0] < _eps*2.0 and self.end_points[1]-x[1] >= _eps*2.0:
                    error = torch.abs(x_star[1] - new_x_star[1])
                else:
                    error = torch.abs(x_star[0] - new_x_star[0])
                if error >= 1e-2:
                    print("WARNING: 1D solution might be wrong, before fixedpoint is ", x_star, " after is ", new_x_star)
                # register hook
                def backward_hook(grad):
                    # we use this hook to calculate dz/dtheta, where z is the equilibrium and theta is the learnable params in the model
                    if self.hook is not None:
                        self.hook.remove()
                        # torch.cuda.synchronize()   # To avoid infinite recursion
                    I_minus_J = torch.eye(2) - autograd.functional.jacobian(self.f, x_star)
                    new_grad = torch.linalg.pinv(I_minus_J)@grad
                    return new_grad
                # hook registration
                self.hook = new_x_star.register_hook(backward_hook)
                # all set! return 
                return new_x_star

            






def _get_phase_boundaries(GibbsFE, total_params_list, T, end_points = [0,1], is_clamp = True, style = "Legendre", quaderature_points = 20, ngrid=99):
    """ 
    total_params_list: in the sequence of [enthalpy_mixing_params_list, config_entropy_params_list, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_HM, Theta_max_Li]
    """
    # sample the Gibbs free energy landscape
    sample = sampling(GibbsFE, total_params_list, T, end_points = end_points, sampling_id=1, is_PDOS_version_G = True, style = style, quaderature_points = quaderature_points, ngrid=ngrid)
    # give the initial guess of miscibility gap
    phase_boundarys_init, _ = convex_hull(sample) 
    # refinement & calculate loss
    if phase_boundarys_init != []:
        # There is at least one phase boundary predicted 
        phase_boundary_fixed_point = []
        for phase_boundary_init in phase_boundarys_init:
            common_tangent = CommonTangent(GibbsFE, total_params_list, T = T, is_PDOS_version_G = True, end_points= end_points, is_clamp = is_clamp, style = style, quaderature_points = quaderature_points) 
            phase_boundary_now = phase_boundary_init.requires_grad_()
            phase_boundary_fixed_point_now = common_tangent(phase_boundary_now) 
            phase_boundary_fixed_point.append(phase_boundary_fixed_point_now)
    else:
        # No boundary find.
        phase_boundary_fixed_point = []
    # check if there is any solved point that are not normal
    phase_boundary_fixed_point_checked = []
    for i in range(0, len(phase_boundary_fixed_point)):
        if phase_boundary_fixed_point[i][0] < phase_boundary_fixed_point[i][1] and phase_boundary_fixed_point[i][0]>=end_points[0] and phase_boundary_fixed_point[i][1] <= end_points[1]:
            phase_boundary_fixed_point_checked.append(phase_boundary_fixed_point[i])
        else:
            # if phase_boundary_fixed_point[i][0] > phase_boundary_fixed_point[i][1] and phase_boundary_fixed_point[i][1]>=end_points[0] and phase_boundary_fixed_point[i][0] <= end_points[1]:
            #     # sequence inversed
            #     phase_boundary_fixed_point_checked.append(phase_boundary_fixed_point[i])
            print("Abandoned abnormal solution ", phase_boundary_fixed_point[i])
    # print(phase_boundary_fixed_point_checked, "in _get_phase_boundaries after common tangent, AMYAO DEBUG")
    return phase_boundary_fixed_point_checked

