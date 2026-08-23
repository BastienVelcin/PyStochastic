from abc import abstractmethod, ABC
from pystochastic.processes.process import Process
class DiffusionProcess(Process,ABC):

    @abstractmethod
    def drift(self,x,t=None):

        """
        Drift function

        Evaluate the drift of the GBM at a given point x and time t.

        Parameters
        ----------
        x : np.ndarray
            Point at which the drift is evaluated.
        t : float
            Time at which the drift is evaluated.

        Returns
        -------
        float :
            Drift evaluated at x and t.
        """

        pass

    @abstractmethod
    def diffusion(self,x,t=None):

        """
        Diffusion function

        Evaluate the diffusion of the GBM at a given point x and time t.

        Parameters
        ----------
        x : np.ndarray
            Point at which the diffusion is evaluated.
        t : float
            Time at which the diffusion is evaluated.

        Returns
        -------
        float :
            Diffusion evaluated at x and t.
        """

        pass

    def simulate(self,n_simulations=1,method="euler-maruyama",plot=False,parallel=False,n_workers=None):

        """
        Simulate method.

        Simulate the process path using both the Euler-Maruyama, Milstein or Runge-Kutta methods and the explicit solution (if available).

        Parameters
        ----------
        n_simulations : int
            Number of trajectories to simulate.
        method : {"exact", "euler-maruyama", "milstein", "runge-kutta"}
            Simulation method to use.
        plot : bool
            Specify if the path should be plotted.
        parallel: bool
            In the case the vectorization doesn't work, the user can specify the usage of parallel computing.
        n_workers: int
            Number of workers to use in parallel computing.

        Returns
        -------
        np.ndarray
            Path of the simulated Geometric Brownian Motion of the form ``(n_simulations, steps + 1, dim)``.
        """
        self.n_simulations = n_simulations
        if method == "euler-maruyama":
            from pystochastic.sde import EulerMaruyama
            self.path = EulerMaruyama(self.drift,
                                      self.diffusion,
                                      self.initial,
                                      self.t_0,
                                      self.t_n,
                                      self.steps).solve(n_simulations=n_simulations,
                                                        plot=plot,
                                                        parallel=parallel,
                                                        n_workers=n_workers)

        elif method == "milstein":
            if self.dim > 1:
                raise ValueError(
                    "The Milstein method is only implemented for 1D processes."
                )

            from pystochastic.sde import Milstein
            self.path = Milstein(self.drift,
                                 self.diffusion,
                                 self.initial,
                                 self.t_0,
                                 self.t_n,
                                 self.steps).solve(n_simulations=n_simulations, plot=plot)

        elif method == "runge-kutta":
            if self.dim > 1:
                raise ValueError(
                    "The Runge-Kutta method is only implemented for 1D processes."
                )
            from pystochastic.sde import RungeKutta

            self.path = RungeKutta(self.drift,
                                   self.diffusion,
                                   self.initial,
                                   self.t_0,
                                   self.t_n,
                                   self.steps).solve(n_simulations, plot=plot)

        elif method == "exact":
            self.path = self._simulate_exact(n_simulations=n_simulations,plot= plot)
        else:
            raise ValueError(
                "The method must be either 'euler-maruyama', 'milstein', 'runge-kutta' or 'exact'."
            )
        return self.path

    def _simulate_exact(self, n_simulations):
        raise NotImplementedError(
            f"Exact simulation is not available for "
            f"{self.__class__.__name__}."
        )