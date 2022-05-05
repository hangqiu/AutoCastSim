import numpy as np
from AVR.autocast_agents.simple_agent import SimpleAgent


class NoisySimpleAgent(SimpleAgent):
    class OrnsteinUhlenbeckActionNoise:
        def __init__(self, mu=0, sigma=0.1, theta=.1, dt=0.05, x0=None):
            self.theta = theta
            self.mu = mu
            self.sigma = sigma
            self.dt = dt
            self.x0 = x0
            self.reset()
    
        def __call__(self):
            x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal()
            self.x_prev = x
            return x

        def reset(self):
            self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        
    def __init__(self, path_to_conf_file):
        super().__init__(path_to_conf_file)
        
        self.noiser = self.OrnsteinUhlenbeckActionNoise()
        self.steer_noise = 0.
        self._expert_control = None

    def run_step(self, input_data, timestamp):
        control = super().run_step(input_data, timestamp)
        self._expert_control = control
        self.steer_noise = self.noiser()
        
        # Apply OU noise
        control.steer += self.steer_noise

        control.steer = np.clip(control.steer, -1.0, 1.0)
        return control
