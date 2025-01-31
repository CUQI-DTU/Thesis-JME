import numpy as np
import cuqi
from cuqi.experimental.mcmc import ProposalBasedSamplerNew
from cuqi.array import CUQIarray
from numbers import Number

class CWMHNew(ProposalBasedSamplerNew):
    """Component-wise Metropolis Hastings sampler.

    Allows sampling of a target distribution by a component-wise random-walk
    sampling of a proposal distribution along with an accept/reject step.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution` or lambda function
        The target distribution to sample. Custom logpdfs are supported by using
        a :class:`cuqi.distribution.UserDefinedDistribution`.
    
    proposal : `cuqi.distribution.Distribution` or callable method
        The proposal to sample from. If a callable method it should provide a
        single independent sample from proposal distribution. Defaults to a
        Gaussian proposal.  *Optional*.

    scale : float or ndarray
        Scale parameter used to define correlation between previous and proposed
        sample in random-walk.  *Optional*. If float, the same scale is used for
        all dimensions. If ndarray, a (possibly) different scale is used for
        each dimension.

    initial_point : ndarray
        Initial parameters. *Optional*

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is
        `callback(sample, sample_index)`, where `sample` is the current sample
        and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.

    kwargs : dict
        Additional keyword arguments to be passed to the base class 
        :class:`ProposalBasedSamplerNew`.

    Example
    -------
    .. code-block:: python
        import numpy as np
        import cuqi
        # Parameters
        dim = 5 # Dimension of distribution
        mu = np.arange(dim) # Mean of Gaussian
        std = 1 # standard deviation of Gaussian

        # Logpdf function
        logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)

        # Define distribution from logpdf as UserDefinedDistribution (sample
        # and gradients also supported as inputs to UserDefinedDistribution)
        target = cuqi.distribution.UserDefinedDistribution(
            dim=dim, logpdf_func=logpdf_func)

        # Set up sampler
        sampler = cuqi.experimental.mcmc.CWMHNew(target, scale=1)

        # Sample
        samples = sampler.sample(2000).get_samples()

    """

    _STATE_KEYS = ProposalBasedSamplerNew._STATE_KEYS.union(['_scale_temp'])

    def __init__(self, target:cuqi.density.Density=None, proposal=None, scale=1,
                 initial_point=None, **kwargs):
        super().__init__(target, proposal=proposal, scale=scale,
                         initial_point=initial_point, **kwargs)
        
    def _initialize(self):
        if isinstance(self.scale, Number):
            self.scale = np.ones(self.dim)*self.scale
        self._acc = [np.ones((self.dim))] # Overwrite acc from ProposalBasedSamplerNew with list of arrays

        # Handling of temporary scale parameter due to possible bug in old CWMH
        self._scale_temp = self.scale.copy()

    @property
    def scale(self):
        """ Get the scale parameter. """
        return self._scale

    @scale.setter
    def scale(self, value):
        """ Set the scale parameter. """
        if self._is_initialized and isinstance(value, Number):
            value = np.ones(self.dim)*value
        self._scale = value

    def validate_target(self):
        if not isinstance(self.target, cuqi.density.Density):
            raise ValueError(
                "Target should be an instance of "+\
                f"{cuqi.density.Density.__class__.__name__}")
        
    def validate_proposal(self):
        if not isinstance(self.proposal, cuqi.distribution.Distribution):
            raise ValueError("Proposal must be a cuqi.distribution.Distribution object")
        if not self.proposal.is_symmetric:
            raise ValueError("Proposal must be symmetric")
        
    @property
    def proposal(self):
        if self._proposal is None:
            self._proposal = cuqi.distribution.Normal(
                mean=lambda location: location,
                std=lambda scale: scale,
                geometry=self.dim,
            )
        return self._proposal
    
    @proposal.setter
    def proposal(self, value):
        self._proposal = value

    def step(self):
        # Initialize x_t which is used to store the current CWMH sample
        x_t = self.current_point.copy()

        # Initialize x_star which is used to store the proposed sample by
        # updating the current sample component-by-component
        x_star = self.current_point.copy()

        # Propose a sample x_all_components from the proposal distribution
        # for all the components
        target_eval_t = self.current_target_logd
        if isinstance(self.proposal,cuqi.distribution.Distribution):
            x_all_components = self.proposal(
                location= self.current_point, scale=self.scale).sample()
        else:
            x_all_components = self.proposal(self.current_point, self.scale)

        # Initialize acceptance rate
        acc = np.zeros(self.dim)

        # Loop over all the components of the sample and accept/reject
        # each component update.
        for j in range(self.dim):
            # propose state x_star by updating the j-th component
            x_star[j] = x_all_components[j]

            # evaluate target
            target_eval_star = self.target.logd(x_star)

            # compute Metropolis acceptance ratio
            alpha = min(0, target_eval_star - target_eval_t)

            # accept/reject
            u_theta = np.log(np.random.rand())
            if (u_theta <= alpha): # accept
                x_t[j] = x_all_components[j]
                target_eval_t = target_eval_star
                acc[j] = 1

            x_star = x_t.copy()

        self.current_target_logd = target_eval_t
        self.current_point = x_t

        return acc

    def tune(self, skip_len, update_count):
        # Store update_count in variable i for readability
        i = update_count

        # Optimal acceptance rate for CWMH
        star_acc = 0.21/self.dim + 0.23

        # Mean of acceptance rate over the last skip_len samples
        hat_acc = np.mean(self._acc[i*skip_len:(i+1)*skip_len], axis=0)

        # Compute new intermediate scaling parameter scale_temp
        # Factor zeta ensures that the variation of the scale update vanishes
        zeta = 1/np.sqrt(update_count+1)  
        scale_temp = np.exp(
            np.log(self._scale_temp) + zeta*(hat_acc-star_acc))

        # Update the scale parameter
        self.scale = np.minimum(scale_temp, np.ones(self.dim))
        self._scale_temp = scale_temp
