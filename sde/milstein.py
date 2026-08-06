import numpy as np
import plotly.graph_objects as go
import processes.brownian


class Milstein:
    '''
    This class provide a way to solve an autonomous SDE using the Milstein method.

    Parameters :
    - mu : scalar or vectorial drift function of the form mu(x) = f(x)
    - sigma : scalar or vectorial diffusion function of the form sigma(x) = g(x)
    - x_0 : initial value (line vector value)
    - t_0 : initial time
    - t_n : final time
    - n_steps : number of time steps
    - n_simulations : number of simulations
    '''
    def __init__(self, mu,sigma, x_0=np.array(0), t_0=0,t_n=1, n_steps=1000,n_simulations=1):
        if np.size(x_0) != 1:
            raise NotImplementedError(
                "Milstein is currently implemented only for autonomous one-dimensional SDEs."
            )
        x_0 = np.array(x_0)
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

    def approx_derivative_diffusion(self,x, eps=1e-6):
        return (self.sigma(x + eps)- self.sigma(x - eps)) / (2 * eps)

    def solve(self, plot=True):

        Y = np.zeros((self.n_simulations,self.n_steps,self.dim))
        Y[:,0,:] = self.x_0
        fig = go.Figure()
        for sim in range(self.n_simulations):
            dW = processes.brownian.Brownian(np.eye(self.dim), self.dim, self.dt, self.n_steps).increments()
            for i in range(1,self.n_steps):
                Y[sim,i,:] = Y[sim,i-1,:] + self.mu(Y[sim,i-1,:])*self.dt + self.sigma(Y[sim,i-1,:]) * dW[i-1,:] + (1/2)*self.sigma(Y[sim,i-1,:])*self.approx_derivative_diffusion(Y[sim,i-1,:])*(dW[i-1,:]**2-self.dt)

        if plot == True:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter(x=self.t, y=Y[sim,:,0],mode="lines", line=dict(width=2)))
            fig.show()
        return Y