from cuqi.utilities import get_non_default_args
from cuqi.distribution import Distribution, Gaussian
from cuqi.solver import ProjectNonnegative, ProjectBox, ProximalL1, ProjectSimplex, ProjectL1Ball, ProjectL2Ball
from cuqi.geometry import Continuous1D, Continuous2D, Image2D

from cuqi.operator import FirstOrderFiniteDifference

import numpy as np
import scipy.sparse as sparse

class RegularizedGaussian(Distribution):
    """ Implicit Regularized Gaussian.

    Defines a so-called implicit prior based on a Gaussian distribution with implicit regularization.
    The regularization can be defined in the form of a proximal operator or a projector. 
    Alternatively, preset constraints and regularization can be used.

    For regularization of the form f(x), provide a single proximal operator.
    For regularization of the form sum_i f_i(L_i x), provide a list of proximal and linear operator pairs.
    
    Can be used as a prior in a posterior which can be sampled with the RegularizedLinearRTO sampler

    For more details on implicit regularized Gaussian see the following paper:

    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Sparse Bayesian inference with regularized
    Gaussian distributions." Inverse Problems 39.11 (2023): 115004.

    Parameters
    ----------
    mean
        See :class:`~cuqi.distribution.Gaussian` for details.

    cov
        See :class:`~cuqi.distribution.Gaussian` for details.

    prec
        See :class:`~cuqi.distribution.Gaussian` for details.

    sqrtcov
        See :class:`~cuqi.distribution.Gaussian` for details.

    sqrtprec
        See :class:`~cuqi.distribution.Gaussian` for details.

    proximal : callable f(x, scale), list of (callable f(x, scale), linear operator) or None
        Euclidean proximal operator f of the regularization function g, that is, a solver for the optimization problem
        min_z 0.5||x-z||_2^2+scale*g(x).

    projector : callable f(x) or None
        Euclidean projection onto the constraint C, that is, a solver for the optimization problem
        min_(z in C) 0.5||x-z||_2^2.

    constraint : string or None
        Preset constraints. Can be set to "nonnegativity" and "box". Required for use in Gibbs.
        For "box", the following additional parameters can be passed:
            lower_bound : array_like or None
                Lower bound of box, defaults to zero
            upper_bound : array_like
                Upper bound of box, defaults to one

    regularization : string or None
        Preset regularization. Can be set to "l1" or "TV". Required for use in Gibbs in future update.
        For "l1", the following additional parameters can be passed:
            strength : scalar
                Regularization parameter, i.e., strength*||x||_1 , defaults to one

        For "TV", the following additional parameters can be passed:
            strength : scalar
                Regularization parameter, i.e., strength*||x||_TV , defaults to one

    """
        
    def __init__(self, mean=None, cov=None, prec=None, sqrtcov=None, sqrtprec=None, proximal = None, projector = None, constraint = None, regularization = None, force_list = False, **kwargs):
        
        # Store regularization parameters and remove them from kwargs passed to Gaussian
        self.optional_regularization_parameters = {
            "lower_bound" : kwargs.pop("lower_bound", None), # Takes default of ProjectBox if None
            "upper_bound" : kwargs.pop("upper_bound", None), # Takes default of ProjectBox if None
            "radius" : kwargs.pop("radius", None),
            "strength" : kwargs.pop("strength", 1)
        }
        
        self._force_list = force_list

        # We init the underlying Gaussian first for geometry and dimensionality handling
        self._gaussian = Gaussian(mean=mean, cov=cov, prec=prec, sqrtcov=sqrtcov, sqrtprec=sqrtprec, **kwargs)
        kwargs.pop("geometry", None)

        # Init from abstract distribution class
        super().__init__(**kwargs)

        self._parse_regularization_input_arguments(proximal, projector, constraint, regularization, self.optional_regularization_parameters)

    def _parse_regularization_input_arguments(self, proximal, projector, constraint, regularization, optional_regularization_parameters):
        """ Parse regularization input arguments with guarding statements and store internal states """
   
        # Guards checking whether the regularization inputs are valid
        if (proximal is not None) + (projector is not None) + max((constraint is not None), (regularization is not None)) == 0:
            raise ValueError("At least some constraint or regularization has to be specified.")
            
        if (proximal is not None) + (projector is not None) == 2:
            raise ValueError("Only one of proximal or projector can be used.")

        if (proximal is not None) + (projector is not None) + max((constraint is not None), (regularization is not None)) > 1:
            raise ValueError("User-defined proximals an projectors cannot be combined with pre-defined constraints and regularization.")

        if proximal is not None:
            if callable(proximal):
                if len(get_non_default_args(proximal)) != 2:
                    raise ValueError("Proximal should take 2 arguments.")
            else:
                pass # TODO: Add error checking for list of regularizations
            
        if projector is not None:
            if callable(projector):
                if len(get_non_default_args(projector)) != 1:
                    raise ValueError("Projector should take 1 argument.")
            else:
                pass # TODO: Add error checking for list of regularizations
            

        # Set user-defined proximals or projectors
        if proximal is not None:
            self._preset = None
            self._proximal = proximal
            return
        
        if projector is not None:
            self._preset = None
            self._proximal = lambda z, gamma: projector(z)
            return
        

        # Set constraint and regularization presets for use with Gibbs
        self._preset = {"constraint": None,
                        "regularization": None}

        self._constraint_prox = None
        self._constraint_oper = None
        if constraint is not None:
            if not isinstance(constraint, str):
                raise ValueError("Constraint needs to be specified as a string.")
            
            c_lower = constraint.lower()
            if c_lower == "nonnegativity":
                self._constraint_prox = lambda z, gamma: ProjectNonnegative(z)
                self._preset["constraint"] = "nonnegativity"
            elif c_lower == "box":
                lower = optional_regularization_parameters["lower_bound"]
                upper = optional_regularization_parameters["upper_bound"]
                self._constraint_prox = lambda z, gamma: ProjectBox(z, lower, upper)
                self._preset["constraint"] = "box"
            elif c_lower == "simplex":
                radius = optional_regularization_parameters["radius"]
                self._constraint_prox = lambda z, gamma: ProjectSimplex(z, radius)
                self._preset["constraint"] = "simplex"
            elif c_lower == "l1":
                radius = optional_regularization_parameters["radius"]
                self._constraint_prox = lambda z, gamma: ProjectL1Ball(z, radius)
                self._preset["constraint"] = "l1"
            elif c_lower == "l2":
                radius = optional_regularization_parameters["radius"]
                self._constraint_prox = lambda z, gamma: ProjectL2Ball(z, radius)
                self._preset["constraint"] = "l2"
            else:
                raise ValueError("Constraint not supported.")
                

        self._regularization_prox = None
        self._regularization_oper = None
        if regularization is not None:
            if not isinstance(regularization, str):
                raise ValueError("Regularization needs to be specified as a string.")
                
            self._strength = optional_regularization_parameters["strength"]
            r_lower = regularization.lower()
            if r_lower == "l1":
                self._regularization_prox = lambda z, gamma: ProximalL1(z, gamma*self._strength)
                self._preset["regularization"] = "l1"
            elif r_lower == "tv":
                # Store the transformation to reuse when modifying the strength
                if isinstance(self.geometry, (Continuous1D)):
                    self._transformation = FirstOrderFiniteDifference(self.geometry.par_dim, bc_type='zero')
                elif isinstance(self.geometry, (Continuous2D, Image2D)):
                    self._transformation = FirstOrderFiniteDifference(self.geometry.fun_shape, bc_type='zero')
                else:
                    raise ValueError("Geometry not supported for total variation")
                self._regularization_prox = lambda z, gamma: ProximalL1(z, gamma*self._strength)
                self._regularization_oper = self._transformation
                self._preset["regularization"] = "TV"
            else:
                raise ValueError("Regularization not supported.")
                

        self._merge_predefined_option()


    def _merge_predefined_option(self):
        # Check whether it is a single proximal and hence FISTA could be used in RegularizedLinearRTO 
        if ((not self._force_list) and
            ((self._constraint_prox is not None) + (self._regularization_prox is not None) == 1) and
            ((self._constraint_oper is not None) + (self._regularization_oper is not None) == 0)):
                if self._constraint_prox is not None:
                    self._proximal = self._constraint_prox
                else:
                    self._proximal = self._regularization_prox 
                return

        # Merge regularization choices in list for use in ADMM by RegularizedLinearRTO
        self._proximal = []
        if self._constraint_prox is not None:
            self._proximal += [(self._constraint_prox, self._constraint_oper if self._constraint_oper is not None else sparse.eye(self.geometry.par_dim))]
        if self._regularization_prox is not None:
            self._proximal += [(self._regularization_prox, self._regularization_oper if self._regularization_oper is not None else sparse.eye(self.geometry.par_dim))]


    @property
    def transformation(self):
        return self._transformation
    
    @property
    def strength(self):
        return self._strength
        
    @strength.setter
    def strength(self, value):
        if self._preset is None or self._preset["regularization"] is None:
            raise TypeError("Strength is only used when the regularization is set to l1 or TV.")

        self._strength = value
        if self._preset["regularization"] in ["l1", "TV"]:        
            self._regularization_prox = lambda z, gamma: ProximalL1(z, gamma*self._strength)

        self._merge_predefined_option()


    # This is a getter only attribute for the underlying Gaussian
    # It also ensures that the name of the underlying Gaussian
    # matches the name of the implicit regularized Gaussian
    @property
    def gaussian(self):
        if self._name is not None:
            self._gaussian._name = self._name
        return self._gaussian
    
    @property
    def proximal(self):
        return self._proximal
    
    @property
    def preset(self):
        return self._preset

    def logpdf(self, x):
        return np.nan
        #raise ValueError(
        #    f"The logpdf of a implicit regularized Gaussian is not be defined.")
        
    def _sample(self, N, rng=None):
        raise ValueError("Cannot be sampled from.")
  
    @staticmethod
    def constraint_options():
        return ["nonnegativity", "box", "simplex", "l1", "l2"]

    @staticmethod
    def regularization_options():
        return ["l1", "TV"]


    # --- Defer behavior of the underlying Gaussian --- #
    @property
    def geometry(self):
        return self.gaussian.geometry
    
    @geometry.setter
    def geometry(self, value):
        self.gaussian.geometry = value
    
    @property
    def mean(self):
        return self.gaussian.mean
    
    @mean.setter
    def mean(self, value):
        self.gaussian.mean = value
    
    @property
    def cov(self):
        return self.gaussian.cov
    
    @cov.setter
    def cov(self, value):
        self.gaussian.cov = value
    
    @property
    def prec(self):
        return self.gaussian.prec
    
    @prec.setter
    def prec(self, value):
        self.gaussian.prec = value
    
    @property
    def sqrtprec(self):
        return self.gaussian.sqrtprec
    
    @sqrtprec.setter
    def sqrtprec(self, value):
        self.gaussian.sqrtprec = value
    
    @property
    def sqrtcov(self):
        return self.gaussian.sqrtcov
    
    @sqrtcov.setter
    def sqrtcov(self, value):
        self.gaussian.sqrtcov = value     
    
    def get_mutable_variables(self):
        add = []
        if self.preset is not None and self.preset['regularization'] in ["l1", "TV"]:
            add = ["strength"]
        return self.gaussian.get_mutable_variables() + add

    # Overwrite the condition method such that the underlying Gaussian is conditioned in general, except when conditioning on self.name
    # which means we convert Distribution to Likelihood or EvaluatedDensity.
    def _condition(self, *args, **kwargs):
        if self.preset is not None and self.preset['regularization'] in ["l1", "TV"]:
            return super()._condition(*args, **kwargs)

        # Handle positional arguments (similar code as in Distribution._condition)
        cond_vars = self.get_conditioning_variables()
        kwargs = self._parse_args_add_to_kwargs(cond_vars, *args, **kwargs)

        # When conditioning, we always do it on a copy to avoid unintentional side effects
        new_density = self._make_copy()

        # Check if self.name is in the provided keyword arguments.
        # If so, pop it and store its value.
        value = kwargs.pop(self.name, None)

        new_density._gaussian = self.gaussian._condition(**kwargs)

        # If self.name was provided, we convert to a likelihood or evaluated density
        if value is not None:
            new_density = new_density.to_likelihood(value)

        return new_density
    
    
class ConstrainedGaussian(RegularizedGaussian):
    
    def __init__(self, mean=None, cov=None, prec=None, sqrtcov=None,sqrtprec=None, projector=None, constraint=None, **kwargs):
        super().__init__(mean=mean, cov=cov, prec=prec, sqrtcov=sqrtcov,sqrtprec=sqrtprec, projector=projector, constraint=constraint, **kwargs)

        
class NonnegativeGaussian(RegularizedGaussian):
    
    def __init__(self, mean=None, cov=None, prec=None, sqrtcov=None,sqrtprec=None, **kwargs):
        super().__init__(mean=mean, cov=cov, prec=prec, sqrtcov=sqrtcov,sqrtprec=sqrtprec, constraint="nonnegativity", **kwargs)