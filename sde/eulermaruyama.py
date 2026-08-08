import numpy as np
import plotly.graph_objects as go
from ..processes import brownian
from ..utils import default_drift, default_diffusion

class EulerMaruyama:
    '''
    This class provide a way to solve an SDE using the Euler-Maruyama method.

    Parameters :
    - mu : scalar or vectorial drift function of the form mu(x,t) = f(x,t)
    - sigma : scalar or vectorial diffusion function of the form sigma(x,t) = g(x,t)
    - x_0 : initial value (line vector value)
    - t_0 : initial time
    - t_n : final time
    - n_steps : number of time steps
    - n_simulations : number of simulations
    '''
    def __init__(self, mu=default_drift,sigma=default_diffusion, x_0=np.array(0), t_0=0,t_n=1, n_steps=1000,n_simulations=1):
        x_0 = np.atleast_1d(x_0)
        self.mu = mu
        self.sigma = sigma
        self.x_0 = x_0
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = n_simulations
        self.dim = np.size(x_0)
        self.t = np.linspace(t_0,t_n,n_steps+1)
        self.dt = (t_n-t_0)/n_steps

    def solve(self, plot=False):

        Y = np.zeros((self.n_simulations,self.n_steps+1,self.dim))
        Y[:,0,:] = self.x_0
        fig = go.Figure()
        for sim in range(self.n_simulations):
            dW = brownian.Brownian(np.eye(self.dim), self.dim, self.t_0, self.t_n, self.n_steps).increments
            for i in range(1,self.n_steps+1):
                Y[sim,i,:] = Y[sim,i-1,:] + self.mu(Y[sim,i-1,:],self.t[i-1])*self.dt + self.sigma(Y[sim,i-1,:],self.t[i-1]) @ dW[i-1,:]
        if plot == True and self.dim <= 3:
            for sim in range(self.n_simulations):
                if self.dim == 1:
                    fig.add_trace(go.Scatter(x=self.t, y=Y[sim,:,0],mode="lines", line=dict(width=2)))

                elif self.dim == 2:
                    fig.add_trace(go.Scatter(x=Y[sim,:, 0], y=Y[sim,:, 1], mode="lines", line=dict(width=2)))
                else:
                    fig.add_trace(go.Scatter3d(x=Y[sim,:, 0],y=Y[sim,:, 1],z=Y[sim,:, 2],mode="lines",line=dict(width=2)))

            fig.show()
        elif plot == True and self.dim > 3:
                print("The path can be plotted only for 1D, 2D and 3D.")
        return Y
