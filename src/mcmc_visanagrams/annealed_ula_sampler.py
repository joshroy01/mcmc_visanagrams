import torch


class AnnealedULASampler:
    """Implements AIS with ULA"""

    def __init__(self, num_steps, num_samples_per_step, step_sizes, gradient_function,
                 sync_function, noise_function):
        assert len(
            step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._num_samples_per_step = num_samples_per_step
        self._gradient_function = gradient_function
        self._sync_function = sync_function
        self._noise_function = noise_function

    def sample_step(self, x, t, text_embeddings):

        for i in range(self._num_samples_per_step):
            ss = self._step_sizes[t]
            std = (2 * ss)**.5
            grad = self._gradient_function(x, t, text_embeddings)
            noise = torch.randn_like(x) * std
            x = x + grad * ss + noise

        return x
