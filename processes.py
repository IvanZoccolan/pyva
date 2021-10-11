from abc import abstractmethod
from abc import ABCMeta
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
from scipy.stats import ncx2


class LevyProcess(metaclass=ABCMeta):
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


class VGProcess(LevyProcess):
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


class CGMYProcess(LevyProcess):
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


class CirMortality:
    """
    CIR intensity of mortality process.

    Parameters
    ----------
    alpha: (float)
    theta: (float)
    sigma: (float) volatility of the process
    mu0: (float) initial value of the  process

    """

    def __init__(self, alpha=7.989275e-05, theta=0.1326157, sigma=0.007732111, mu0=0.002239457):
        self.alpha = alpha
        self.theta = theta
        self.sigma = sigma
        self.mu0 = mu0
        self.gamma = np.sqrt(theta**2 + 2*sigma**2)
        self.df = 4 * alpha / sigma ** 2

    def _c1(self, t):
        n = 2 * self.gamma * np.exp(0.5 * (self.gamma - self.theta) * t)
        d = (self.gamma - self.theta) * (np.exp(self.gamma * t) - 1) + 2 * self.gamma
        return (n/d)**((2*self.alpha)/(self.sigma**2))

    def _c2(self, t):
        n = 2 * (np.exp(self.gamma * t) - 1)
        d = (self.gamma - self.theta)*(np.exp(self.gamma*t) - 1) + 2*self.gamma
        return n/d

    def px(self, t, mu):
        """
         Survival function of the CIR process

        :param t: (float) time
        :param mu: (float) value of the intensity of mortality at time t = 0
        :return: (float) probability of survival at time t
        """
        assert t >= 0.0
        return self._c1(t) * np.exp(-self._c2(t) * mu)

    def qx(self, t, mu):
        """
        Death function of the CIR process

        :param t: (float) time
        :param mu: (float) value of the intensity of mortality at time t = 0
        :return: (float) probability of death at time t
        """

        assert t >= 0.0
        return 1 - self.px(t, mu)

    def cpdf(self, x, mu, dt):
        """
        Conditional probability density function of the CIR intensity of mortality.
        This is the pdf of mu(t) given mu(s), where dt = t - s.
        :param x: (float) x-coordinate
        :param mu: (float) intensity of mortality mu(s) to condition upon.
        :param dt: (float) time delta between intensities of mortality dt = t - s.
        :return: float y-coordinate of the cpdf.

        """
        k = 4 * self.theta / (self.sigma ** 2 * (np.exp(self.theta*dt) - 1))
        nc_par = k * mu * np.exp(self.theta*dt)
        return k * ncx2.pdf(x=k*x, df=self.df, nc=nc_par)

    def cond_moments(self, mu, dt):
        """
        Calculates the variance and mean of mu(t) given m(s)
        :param mu: (float) intensity of mortality mu(s) to condition upon.
        :param dt: (float) time delta between intensities of mortality dt = t - s.
        :return:  tuple of floats with the mean and variance of the random variable mu(t) | mu(t)
        """
        d1 = np.exp(2*self.theta*dt) - np.exp(self.theta*dt)
        d2 = (1 - np.exp(self.theta*dt))**2
        var = (self.sigma**2 / self.theta)*(mu*d1 + 0.5*d2)
        mean = mu * np.exp(self.theta*dt) - (self.alpha / self.theta) * (1 - np.exp(self.theta*dt))
        return mean, var

    def cond_support(self, mu, dt, verbose=False):
        """
        Estimates the support of the density of  mu(t) | mu(s)
        :param mu: (float) intensity of mortality mu(s) to condition upon.
        :param dt: (float) time delta between intensities of mortality dt = t - s.
        :param verbose: (boolean) default to False. If true it returns the integral of
        conditional density over the support
        :return: a tuple with the left and right extremes of the support.
        If verbose = True it returns a list with the support and the integral of the density over it
        The latter should sum to a value reasonably close to one and can be considered as a test
        the estimate of the support is good enough.
        """
        m, v = self.cond_moments(mu, dt=dt)
        res = max(m - np.ceil(self.df) * np.sqrt(v), 0), m + np.ceil(self.df) * np.sqrt(v)
        if verbose:
            xx, dx = np.linspace(res[0], res[1], 2**9+1, retstep=True)
            integr = np.apply_along_axis(lambda x: self.cpdf(x, mu, dt=dt), 0, xx)
            support_integr= np.trapz(integr, dx=dx)
            return {'support': res, 'integral': support_integr}
        else:
            return res




