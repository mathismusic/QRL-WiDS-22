import numpy as np
from numpy.random import *

class stationary_bandit:
    def __init__(self, n: int, qa: list[float], optimistic: bool=False) -> None:
        self.n = n
        self.qa = qa
        self.Qt = np.zeros((1, n)) + 5*optimistic # optimistic initial values greatly increase accuracy by inducing initial exploration, especially if eps=0.
        self.Nt = np.zeros((1, n), dtype=int)
        self.P = ['Gaussian(0,1)']*self.n

    def get_best_move(self) -> int:
        return max(range(self.n), key=lambda x: self.Qt[0, x]) # argmax_x Qt(x)
    
    def get_move(self) -> int:
        return self.get_best_move()
    
    def make_move(self) -> float:
        move = self.get_move()
        reward = self.qa[0, move] + randn() if self.P[move] == 'Gaussian(0,1)' else 0
        self.Nt[0, move] += 1
        to_return = self.Qt[0, move] # our estimate of the value of the move we have taken

        # update the value of this move based on received reward.
        self.update_Qt(move, reward)
        return to_return

    def update_Qt(self, move: int, reward: float) -> None:
        pass # no uppdates in the base class

    def simulate(self, steps: int=1000) -> float:
        # return average reward over these `steps` steps. Of course, this value will (roughly) increase with the number of steps.
        return sum(self.make_move() for _ in range(steps))/steps

    @classmethod
    def test(cls, n: int, stepsize: int=200, epochs: int=10, samples: int=1000, verbose: int=0, **kwargs):
        # n = arms of the bandit. In general, (as expected), the avg value/step that we predict increases with n.
        
        qa = randn(samples, n) # randomly generated q(a) values for each of n arms for each of `samples` samples.
        avg_avg_reward_over_samples = np.zeros((1, epochs))

        optimal = sum(max(qa[t, i] for i in range(n)) for t in range(samples))/samples # average optimal value per step.
        print(f"average (over samples) optimal value/step: {optimal:.6f}")
            
        if 'eps' in kwargs: print("epsilon:", kwargs['eps'])
        if 'c' in kwargs: print("c:", kwargs['c'])
        for i in range(samples):
            b = cls(n, qa[i:i+1, :], **kwargs)
            for k in range(epochs):
                avg_avg_reward_over_samples[0, k] += b.simulate(stepsize)
        avg_avg_reward_over_samples[0, :] /= samples
        if verbose == 0:
            print(f'After {epochs*stepsize} steps, we (on average) predict a value/step of {avg_avg_reward_over_samples[0, epochs-1]:.6f}, ({avg_avg_reward_over_samples[0, epochs-1]/optimal * 100:.1f}% optimal)')
        elif verbose == 1:
            for k in range(epochs):
                print(f'After {(1 + k)*stepsize} steps, we (on average) predict a value/step of {avg_avg_reward_over_samples[0, k]:.6f}, ({avg_avg_reward_over_samples[0, k]/optimal * 100:.1f}% optimal)')

class eps_greedy_bandit(stationary_bandit):
    def __init__(self, n: int, qa: list[float], eps: float=1e-2, optimistic: bool=False, learn_type: str='constant-rate', alpha: float=0.2) -> None:
        super().__init__(n, qa, optimistic)
        self.eps = eps
        self.learn_type = learn_type

        if learn_type == 'constant-rate': # put the declarations for these special variables in **kwargs
            self.alpha = alpha
        
    def get_move(self):
        if rand() < self.eps:
            return randint(self.n)
        return self.get_best_move()
    
    def update_Qt(self, move: int, reward: float) -> None:
        if self.learn_type == 'sample-average':
            self.Qt[0, move] = (self.Qt[0, move] * (self.Nt[0, move] - 1) + reward)/self.Nt[0, move]
        elif self.learn_type == 'constant-rate':
            self.Qt[0, move] = self.Qt[0, move]  + self.alpha * (reward - self.Qt[0, move])
        else:
            pass

# bandit where move selection happens (most often) greedily but we allow exploration by adding a term. We inherit from greedy bandit to experiment with allowing eps-greedy here too - though we don't need to.
class ucb_bandit(eps_greedy_bandit):
    def __init__(self, n: int, qa: list[float], optimistic: bool=False, c: float=2, learn_type: str = 'constant-rate', alpha: float = 0.2, eps: float=0) -> None:
        super().__init__(n, qa, eps, optimistic, learn_type, alpha)
        self.c = c
        self.t = 0 # maintain a time variable.
    
    def get_best_move(self):
        return max(range(self.n), key=lambda x: np.inf if self.Nt[0, x] == 0 else self.Qt[0, x] + self.c * np.sqrt(np.log(self.t)/self.Nt[0, x]))

    def make_move(self):
        self.t += 1
        return super().make_move()
