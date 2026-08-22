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
