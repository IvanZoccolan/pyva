from abc import abstractmethod
from abc import ABCMeta
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


class BaseProcess(metaclass=ABCMeta):
    """ Base class for Levy stochastic processes {X(t): t >= 0}
        defined by their characteristic function.
    """
    @abstractmethod
    def __init__(self): pass

    @abstractmethod
    def characteristic(self, u, time): pass

    def density(self, x, time=1):
        """
        Probability density function of the X(t) variable of the process.

        Parameters
        ----------
        x: (float) x-coordinate.
        time: (float) time coordinate.

        Returns
        -------
        float: value of the pdf of X(t) at point x f(x, t).

        Notes
        -----
        The pdf is numerically calculated from the characteristic function
        by means of the Pelaez formula.
        """
        return 1 / np.pi * quad(lambda u: np.real(np.exp(-u * x * 1j) * self.characteristic(u, time)),
                                1e-15,
                                np.inf)[0]


class VGProcess(BaseProcess):
    """
    Class for the Variance Gamma stochastic process.

    Parameters
    ----------
    mu: (float) drift of the Brownian motion.
    sigma: (float) volatility of the Brownian motion.
    nu: (float) variance of the Gamma process
    """
    def __init__(self, mu=-0.3150, sigma=0.1301, nu=0.1753):
        super().__init__()
        self._mu = mu
        if sigma > 0.:
            self._sigma = sigma
        else:
            raise KeyError("sigma must be strictly greater than zero.")
        if nu > 0.:
            self._nu = nu
        else:
            raise KeyError("nu must be strictly greater than zero.")

    def characteristic(self, u, time=1):
        """
        Characteristic function Phi(u) of the X(t) variable of the Variance Gamma process.

        Parameters
        ----------
        u: (float) u-coordinate of the characteristic function.
        time: (float) time t coordinate of the process {X(t) t>=0}.

        Returns
        ------
        float: Phi(u) ordinate value mapping the u-coordinate.
        """
        const = -(time/self._nu)
        esp = np.log(1 - 1j * u * self._mu * self._nu + 0.5 * self._sigma ** 2 * u ** 2 * self._nu)
        return np.exp(const * esp)


class CGMYProcess(BaseProcess):
    """
    Class for the Carr, Geman, Madan, Yor (CGMY) stochastic process.

    Parameters
    ----------
    g: (float) right skewness parameter.
    m: (float) left skewness parameter.
    y: (float) parameter describing the Levy density near 0.
    c: (float) parameter measure of the overall level of activity.
    """
    def __init__(self, c=0.6817, g=18.0293, m=57.6750, y=0.8):
        super().__init__()
        if g >= 0.:
            self._g = g
        else:
            raise KeyError("g must be greater of equal zero.")
        if m >= 0.:
            self._m = m
        else:
            raise KeyError("m must be greater of equal zero.")
        if c > 0.:
            self._c = c
        else:
            raise KeyError("c must be strictly greater than zero.")
        if y < 2.:
            self._y = y
        else:
            raise KeyError("y must be strictly less than 2.")

    def characteristic(self, u, time=1):
        """
        Characteristic function Phi(u) of the X(t) variable of the CGMY process.

        Parameters
        ----------
        u: (float) u-coordinate of the characteristic function.
        time: (float) time t coordinate of the process {X(t) t>=0}.

        Returns
        ------
        float: Phi(u) ordinate value mapping the u-coordinate.
        """
        left = (self._m - 1j*u)**self._y - self._m**self._y
        right = (self._g + 1j*u)**self._y - self._g**self._y
        return np.exp(self._c * time * gamma(-self._y) * (left + right))


class VGProcess(BaseProcess):
    """
    Class for the Variance Gamma stochastic process.

    Parameters
    ----------
    mu: (float) drift of the Brownian motion.
    sigma: (float) volatility of the Brownian motion.
    nu: (float) variance of the Gamma process
    """
    def __init__(self, mu=-0.3150, sigma=0.1301, nu=0.1753):
        super().__init__()
        self._mu = mu
        if sigma > 0.:
            self._sigma = sigma
        else:
            raise KeyError("sigma must be strictly greater than zero.")
        if nu > 0.:
            self._nu = nu
        else:
            raise KeyError("nu must be strictly greater than zero.")

    def characteristic(self, u, time=1):
        """
        Characteristic function Phi(u) of the X(t) variable of the Variance Gamma process.

        Parameters
        ----------
        u: (float) u-coordinate of the characteristic function.
        time: (float) time t coordinate of the process {X(t) t>=0}.

        Returns
        ------
        float: Phi(u) ordinate value mapping the u-coordinate.
        """
        const = -(time/self._nu)
        esp = np.log(1 - 1j * u * self._mu * self._nu + 0.5 * self._sigma ** 2 * u ** 2 * self._nu)
        return np.exp(const * esp)
