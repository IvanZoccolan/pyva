
import numpy as np
from scipy import interpolate
from scipy import trapz
from scipy import optimize


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
        self.g_amount = self.premium
        self.withdraw_amount = self.premium / self.maturity
        self.div = 0.0
        self.penalty = 0.05
        # Steps to discretize time, guaranteed account and personal account
        self.t_step = 1  # time step
        self.g_step = self.withdraw_amount   # guaranteed account step
        self.p_step = 2 * self.withdraw_amount   # personal account step
        # Integration limits
        self.right_limit = 0.5
        self.left_limit = -1.5

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise KeyError(k)

        # Internal data members
        # Max values for the accounts
        self._max_g_account = self.premium  # max guaranteed account
        self._k = 10  # internal factor to calculate the max personal account
        self._max_p_account = self._k * self._max_g_account  # max personal account
        # Variables for the pricing algorithm
        self._H = int(self._max_g_account / self.g_step) + 1
        self._L = int(self._max_p_account / self.p_step) + 1
        self._N = int(self.maturity / self.t_step)
        self._g_account = np.linspace(0, self._max_g_account, self._H)
        self._p_account = np.linspace(0, self._max_p_account, self._L)
        self._theta_step = 0.25 * self.withdraw_amount
        # constants for the integration (step 2. II)
        self._d = np.real(-np.log(self.process.characteristic(-1j)))
        self._beta = np.exp((self.spot - self.div + self._d)*self.t_step) * (1 - self.fee*self.t_step)
        self._m = np.exp(-self.spot*self.t_step)
        self._xx = np.linspace(self.left_limit, self.right_limit, num=2**8+1)
        self._dx = abs(self._xx[1] - self._xx[0])
        self._yy = self._beta * np.apply_along_axis(np.exp, 0, self._xx)
        self._density = np.apply_along_axis(np.vectorize(self.process.density), 0, self._xx)
        self._price = 0.0

    def set_maturity(self, maturity=20):
        assert type(maturity) == int
        self.maturity = maturity
        self._N = int(self.maturity / self.t_step)

    def set_integration_limits(self, right=0.5, left=-1.5):
        self.right_limit = right
        self.left_limit = left
        self._xx = np.linspace(self.left_limit, self.right_limit, num=2 ** 8 + 1)
        self._dx = abs(self._xx[1] - self._xx[0])
        self._yy = self._beta * np.apply_along_axis(np.exp, 0, self._xx)
        self._density = np.apply_along_axis(np.vectorize(self.process.density), 0, self._xx)

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

    def price(self, method="static"):
        """

        :param method:
        :return:
        """

        @np.vectorize
        def value(a, w, ff):
            """
            Compute the contract value at a given point of the guaranteed account
            x personal account grid.
            It's vectorized via decorator so that we can pass the meshgrid to it.

            :param a:
            :param w:
            :param ff:
            :return:
            """
            if a == 0 and w == 0:
                return 0.0
            aorg = min(self.withdraw_amount, a)
            control_set = np.nditer(aorg)
            if method == "dynamic":
                ctrl_set_upper = max(aorg, w)
                ctrl_set_points = int(ctrl_set_upper / self._theta_step)
                control_set = np.linspace(0, ctrl_set_upper, num=ctrl_set_points)
            elif method == "mixed":
                control_set = np.nditer(np.array([aorg, w]))
            values = []
            for theta in control_set:
                cc = theta - self.penalty * max(theta - aorg, 0)
                if theta <= aorg:
                    aa = a - theta
                else:
                    aa = max(min(a - theta, a * (1 - theta / w)), 0) if w != 0 else 0

                _bb = max(w - theta, 0) * self._yy
                _aa = np.repeat(aa, len(_bb))
                integrand = ff.ev(_bb, _aa) * self._density
                # Integral calculated by means of the trapezoid method
                values.append(cc + self._m * trapz(integrand, dx=self._dx))
            return max(values)

        def calc_price(n, c_val):
            if n == 1:
                interp_func = interpolate.RectBivariateSpline(self._p_account, self._g_account, c_val)
                return value(gg, pp, interp_func)
            else:
                # Interpolate the H x L triplets
                interp_func = interpolate.RectBivariateSpline(self._p_account, self._g_account, c_val)
                # Compute the contract value at each point of the grid
                c_val = value(gg, pp, interp_func)
                return calc_price(n-1, c_val)

        # Initial contract value
        gg, pp = np.meshgrid(self._g_account, self._p_account)
        ini_val = np.maximum(gg, pp)
        # Step 2. II Compute the contract value at each time step but t=0.
        val = calc_price(self._N-1, ini_val)

        # Contract value at inception (t=0)
        val_func = interpolate.RectBivariateSpline(self._p_account, self._g_account, val)
        _YY = self.premium*self._yy
        _U = np.repeat(self.premium, len(self._yy))
        final_integrand = val_func.ev(_YY, _U) * self._density
        self._price = self._m*trapz(final_integrand, dx=self._dx)
        return self._price

    def fair_fee(self, a=0, b=1, tol=1e-4, method="static"):
        """

        :param a:
        :param b:
        :param tol:
        :param method:
        :return:
        """

        def fun(x):
            self.set_fee(x)
            return self.price(method=method) - self.premium
        return optimize.brentq(fun, a=a, b=b, xtol=tol, maxiter=50)
