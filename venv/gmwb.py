
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
        self._k = 5  # internal factor to calculate the max personal account
        self._max_p_account = self._k * self._max_g_account  # max personal account
        # Variables for the pricing algorithm
        self._g_points = int(self._max_g_account / self.g_step) + 1
        self._p_points = int(self._max_p_account / self.p_step) + 1
        self._t_points = int(self.maturity / self.t_step)
        self._g_account = np.linspace(0, self._max_g_account, self._g_points)
        self._p_account = np.linspace(0, self._max_p_account, self._p_points)
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
        self._t_points = int(self.maturity / self.t_step)

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
        self._g_points = int(self._max_g_account / self.g_step) + 1
        self._g_account = np.linspace(0, self._max_g_account, self._g_points)

    def set_pstep(self, pstep=1):
        self.p_step = pstep
        self._p_points = int(self._max_p_account / self.p_step) + 1
        self._p_account = np.linspace(0, self._max_p_account, self._p_points)

    def price(self, method="static"):
        """

        :param method:
        :return:
        """
        @np.vectorize
        def value(a, w, ff):
            if a == 0 and w == 0:
                return 0.0
            max_guaranteed_withdraw = min(self.withdraw_amount, a)
            withdrawals = np.repeat(max_guaranteed_withdraw, 2)
            max_withdraw = max_guaranteed_withdraw
            if method == "dynamic":
                max_withdraw = max(max_guaranteed_withdraw, w)
                withdrawals_splits = int(max_withdraw / self._theta_step) + 1
                withdrawals = np.linspace(0, max_withdraw, num=withdrawals_splits)
            elif method == "mixed":
                withdrawals = np.array([max_guaranteed_withdraw, w])
                max_withdraw = max(max_guaranteed_withdraw, w)

            # Cashflows
            cash_flows = withdrawals - self.penalty * (withdrawals - max_guaranteed_withdraw).clip(0, max_withdraw)
            # Guaranteed account
            idx = withdrawals > max_guaranteed_withdraw
            guaranteed_account = (a - withdrawals)
            pro_rata = a * (1 - withdrawals / w) if w != 0 else np.zeros(withdrawals.shape)
            guaranteed_account[idx] = np.minimum(guaranteed_account[idx], pro_rata[idx]).clip(0, np.Inf)
            # Personal account
            personal_account = (w - withdrawals).clip(0, np.Inf)
            personal_account = personal_account.reshape((personal_account.shape[0], 1)) * self._yy
            guaranteed_account = np.tile(guaranteed_account.reshape((guaranteed_account.shape[0], 1)),
                                         personal_account.shape[1])
            # Calculate the integral
            accounts = np.hstack((personal_account, guaranteed_account))
            cols_paccount = personal_account.shape[1]
            integrands = np.apply_along_axis(lambda x: ff.ev(x[:cols_paccount], x[cols_paccount:]) * self._density, 1,
                                             accounts)
            integrals = cash_flows + self._m * np.apply_along_axis(lambda x: trapz(x, dx=self._dx), 1, integrands)
            return integrals.max()

        def calc_price(n, c_val):
            if n == 1:
                interp_func = interpolate.RectBivariateSpline(self._p_account, self._g_account, c_val)
                c_val = value(gg, pp, interp_func)
                return c_val
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
        val = calc_price(self._t_points - 1, ini_val)

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
