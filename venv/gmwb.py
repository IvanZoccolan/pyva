
import numpy as np
from scipy import interpolate
from scipy.integrate import trapz


class Contract:
    """

    """
    def __init__(self, process,  **kwargs):
        # User defined data members
        self.process = process
        self.maturity = 20
        self.fee = 0.0043
        self.spot = 0.03
        self.premium = 100
        self.g_amount = 100
        self.withdraw_amount = 5
        self.div = 0.0
        self.penalty = 0.05
        # Steps to discretize time, guaranteed account and personal account
        self.t_step = 1  # time step
        self.g_step = 10  # guaranteed account step
        self.p_step = 10  # personal account step
        # Integration limits
        self.right_limit = 1.0
        self.left_limit = -2.0

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise KeyError(k)

        # Internal data members
        # Max values for the accounts
        self._max_g_account = self.premium  # max guaranteed account
        self._k = 5  # internal factor to calculate the max personal account
        self._max_p_account = self._k * self._max_g_account  # max personal account
        # Variables for the pricing algorithm
        self._H = int(self._max_g_account / self.g_step) + 1
        self._L = int(self._max_p_account / self.p_step) + 1
        self._N = int(self.maturity / self.t_step) + 1
        self._g_account = np.linspace(0, self._max_g_account, self._H)
        self._p_account = np.linspace(0, self._max_p_account, self._L)
        # constants for the integration (step 2. II)
        self._d = np.real(-np.log(self.process.characteristic(-1j)))
        self._beta = np.exp((self.spot - self.div + self._d)*self.t_step) * (1 - self.fee*self.t_step)
        self._m = np.exp(-self.spot*self.t_step)
        self._xx = np.linspace(self.left_limit, self.right_limit, num=2**8+1)
        self._dx = abs(self._xx[1] - self._xx[0])
        self._yy = self._beta * np.apply_along_axis(np.exp, 0, self._xx)
        self._density = np.apply_along_axis(np.vectorize(self.process.density), 0, self._xx)
        self._values = []  # list to gather all calculation steps.
        self._price = 0.0

    def set_fee(self, fee=0.0043):
        self.fee = fee
        self._beta = np.exp((self.spot - self.div + self._d) * self.t_step) * (1 - self.fee * self.t_step)
        self._yy = self._beta * np.apply_along_axis(np.exp, 0, self._xx)

    def set_spot(self, spot=0.03):
        self.spot = spot
        self._beta = np.exp((self.spot - self.div + self._d) * self.t_step) * (1 - self.fee * self.t_step)
        self._m = np.exp(-self.spot * self.t_step)
        self._yy = self._beta * np.apply_along_axis(np.exp, 0, self._xx)

    def set_gstep(self, gstep=1):
        self.g_step = gstep
        self._H = int(self._max_g_account / self.g_step) + 1
        self._g_account = np.linspace(0, self._max_g_account, self._H)

    def set_pstep(self, pstep=1):
        self.p_step = pstep
        self._L = int(self._max_p_account / self.p_step) + 1
        self._p_account = np.linspace(0, self._max_p_account, self._L)

    def price(self):
        # Initial contract value
        gg, pp = np.meshgrid(self._g_account, self._p_account)
        self._values.append(np.maximum(gg, pp))

        # Evaluate the contract value at each point of the grid
        @np.vectorize
        def value(a, w, ff, theta):
            cc = theta - self.penalty*max(theta - min(self.withdraw_amount, a), 0)
            if theta <= min(self.withdraw_amount, a):
                aa = a - theta
            else:
                aa = max(min(a - theta, a*(1-theta/w)), 0) if w != 0 else 0

            _yy = max(w - theta, 0) * self._yy
            _ff = np.array([ff(y, aa).item() for y in _yy])
            integrand = _ff * self._density
            return cc + self._m * trapz(integrand, dx=self._dx)

        # Static approach
        theta_val = self.withdraw_amount
        for t in np.arange(self._N - 2, 0, -self.t_step):
            # Step 2. I Interpolate the H x L triplets
            val_func = interpolate.RectBivariateSpline(self._p_account, self._g_account,  self._values[-1])
            # Step 2. II Compute the contract value at each time step.
            self._values.append(value(gg, pp, val_func, theta_val))

        # Contract value at inception
        val_func = interpolate.RectBivariateSpline(self._p_account, self._g_account, self._values[-1])
        _YY = self.premium*self._yy
        _FF = np.array([val_func(y, self.premium).item() for y in _YY])
        final_integrand = _FF * self._density
        self._price = self._m * trapz(final_integrand, dx=self._dx)
        return self._price
