from cuqi.implicitprior import RegularizedGaussian
from cuqi.distribution import Distribution, GMRF

class RegularizedGMRF(RegularizedGaussian):
    """ Implicit Regularized GMRF (Gaussian Markov Random Field). 

    Defines a so-called implicit prior based on a GMRF distribution with implicit regularization.
    The regularization can be defined in the form of a proximal operator or a projector.
    Alternatively, preset constraints and regularization can be used.

    For regularization of the form f(x), provide a single proximal operator.
    For regularization of the form sum_i f_i(L_i x), provide a list of proximal and linear operator pairs.

    Can be used as a prior in a posterior which can be sampled with the RegularizedLinearRTO sampler.

    For more details on implicit regularized Gaussian see the following paper:

    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Sparse Bayesian inference with regularized
    Gaussian distributions." Inverse Problems 39.11 (2023): 115004.

    Parameters
    ----------
    mean
        See :class:`~cuqi.distribution.GMRF` for details.
        
    prec
        See :class:`~cuqi.distribution.GMRF` for details.

    bc_type
        See :class:`~cuqi.distribution.GMRF` for details.

    order
        See :class:`~cuqi.distribution.GMRF` for details.

    proximal : callable f(x, scale) or None
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
        Preset regularization. Can be set to "l1". Required for use in Gibbs in future update.
        For "l1", the following additional parameters can be passed:
            strength : scalar
                Regularization parameter, i.e., strength*||x||_1 , defaults to one

        For "TV", the following additional parameters can be passed:
            strength : scalar
                Regularization parameter, i.e., strength*||x||_TV , defaults to one

    """
    def __init__(self, mean=None, prec=None, bc_type='zero', order=1, proximal = None, projector = None, constraint = None, regularization = None, force_list = False, **kwargs):
            
        self.optional_regularization_parameters = {"lower_bound" : kwargs.pop("lower_bound", None),
                "upper_bound" : kwargs.pop("upper_bound", None),
                "radius" : kwargs.pop("radius", None),
                "strength" : kwargs.pop("strength", None)}
        
        self._force_list = force_list
        
        # Underlying explicit GMRF
        self._gaussian = GMRF(mean, prec, bc_type=bc_type, order=order, **kwargs)
        kwargs.pop("geometry", None)

        # Init from abstract distribution class
        super(Distribution, self).__init__(**kwargs)

        self._parse_regularization_input_arguments(proximal, projector, constraint, regularization, self.optional_regularization_parameters)



class ConstrainedGMRF(RegularizedGMRF):
    
    def __init__(self, mean=None, prec=None, bc_type='zero', order=1, projector=None, constraint=None, **kwargs):
        super().__init__(mean=mean, prec=prec, bc_type=bc_type, order=order, projector=projector, constraint=constraint, **kwargs)

        
class NonnegativeGMRF(RegularizedGMRF):
    
    def __init__(self, mean=None, prec=None, bc_type='zero', order=1, **kwargs):
        super().__init__(mean=mean, prec=prec, bc_type=bc_type, order=order, constraint="nonnegativity", **kwargs)