import numpy as np
import plotly.graph_objects as go
import scipy

class CIR:
    def __init__(self, a=1, b=1, sigma=1, r_0=0, t_0=0, t_n=1, n_steps=1000,n_simulations=1):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r_0 = r_0
        if (self.a <= 0) or (self.b < 0) or (self.sigma <= 0) or (self.r_0 < 0):
            raise ValueError("The model parameters must satisfy a > 0, b >= 0, sigma > 0 and r_0 >= 0.")


        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = n_simulations
        self.t = np.linspace(t_0, t_n, n_steps + 1)
        self.dt = (t_n - t_0) / n_steps
        self.path = None

        self.nu = (4*self.a*self.b)/(self.sigma**2)
        self.factor = (4*self.a*np.exp(-self.a * self.dt))/((self.sigma**2)*(1-np.exp(-self.a * self.dt)))
        self.c = ((self.sigma**2)*(1-np.exp(-self.a * self.dt)))/(4*self.a)

    def drift(self,x,t):
        return self.a * (self.b-x)

    def diffusion(self,x,t):
        return self.sigma * np.sqrt(np.maximum(x,0))

    def simulate(self, method="exact"):
        if method == "euler-maruyama":
            from pystochastic.sde import EulerMaruyama
            if (2 * self.a * self.b < self.sigma ** 2):
                raise ValueError(
                    "The model parameters are inconsistent with the model. Please choose a, b and sigma such that 2*a*b >= sigma^2"
                )
            self.path = EulerMaruyama(self.drift,self.diffusion,self.r_0,self.t_0,self.t_n,self.n_steps,self.n_simulations).solve()
        elif method == "exact":
            self.path = np.zeros((self.n_simulations,self.n_steps+1, 1))
            self.path[:,0] = self.r_0
            for sim in range(self.n_simulations):
                for i in range(1,self.n_steps+1):
                    Y = scipy.stats.ncx2.rvs(df=self.nu, nc=self.path[sim,i-1] * self.factor)
                    self.path[sim,i] = self.c*Y
        else:
            raise ValueError("The method must be either 'euler-maruyama' or 'exact'.")
        return self.path

    def plot(self):
        if self.path is None:
            raise ValueError("The path has not been simulated yet. Please run the simulate method first.")
        fig = go.Figure()
        for sim in range(self.n_simulations):
            fig.add_trace(go.Scatter(x=self.t, y=self.path[sim,:,0], mode="lines", line=dict(width=2)))
        fig.show()
