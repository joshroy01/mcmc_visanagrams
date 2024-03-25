import torch


class AnnealedUHASampler:
    """Implements UHA Sampling"""

    def __init__(self, num_steps, num_samples_per_step, step_sizes, damping_coeff, mass_diag_sqrt,
                 num_leapfrog_steps, gradient_function, sync_function):
        assert len(
            step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
        self._damping_coeff = damping_coeff
        self._mass_diag_sqrt = mass_diag_sqrt
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._num_leapfrog_steps = num_leapfrog_steps
        self._num_samples_per_step = num_samples_per_step
        self._gradient_function = gradient_function
        self._sync_function = sync_function

    def leapfrog_step(self, x, v, i, text_embeddings):
        step_size = self._step_sizes[i]
        return leapfrog_step(x, v, lambda _x: self._gradient_function(_x, i, text_embeddings),
                             step_size, self._mass_diag_sqrt[i], self._num_leapfrog_steps)

    def sample_step(self, x, t, text_embeddings):

        # Sample Momentum
        v = torch.randn_like(x) * self._mass_diag_sqrt[t]
        v = self._sync_function(v)

        for i in range(self._num_samples_per_step):

            # Partial Momentum Refreshment
            eps = torch.randn_like(x)
            eps = self._sync_function(eps)

            v = v * self._damping_coeff + np.sqrt(
                1. - self._damping_coeff**2) * eps * self._mass_diag_sqrt[t]

            x, v = self.leapfrog_step(x, v, t, text_embeddings)

        return x
