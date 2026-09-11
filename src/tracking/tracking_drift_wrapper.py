import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

class RandomDriftWrapper(BaseParallelWrapper):
    """
    PettingZoo wrapper for DSSE DroneSwarmSearch.
    Randomizes target drift direction (vector) on each reset() call.
    This prevents drones from memorizing a fixed spatial drift path
    and forces policy networks to process the 25x25 probability matrix.
    """

    def __init__(self, env, speed=1.0):
        super().__init__(env)
        self.speed = speed

    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        else:
            options = dict(options)

        # Generate a random angle theta in [0, 2*pi)
        if seed is not None:
            np.random.seed(seed)
        
        angle = np.random.uniform(0, 2 * np.pi)
        drift_vector = (
            float(self.speed * np.cos(angle)),
            float(self.speed * np.sin(angle))
        )
        options["vector"] = drift_vector

        return self.env.reset(seed=seed, options=options)
