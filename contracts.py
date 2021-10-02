
import numpy as np
from scipy import interpolate, trapz, optimize
from numba import jit, int32, float64


class GMWB:
    """

    """

    def __init__(self, process, **kwargs):
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
        self.g_points = 21  # Number of points to discretize the guaranteed account
        self.p_points = 21  # Number of points to discretize of the personal account
        # Integration limits
        self.right_limit = 0.5
        self.left_limit = -1.5

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise KeyError(k)

        # Internal data members
        # Personal and guaranteed accounts
        self._max_g_account = self.premium  # max guaranteed account
        self._t_points = int(self.maturity / self.t_step)
        self._g_account = np.linspace(0, self._max_g_account, self.g_points)
        self._p_account = self._map_personal(np.linspace(0, 1, self.p_points))
        self._p_account[-1] = self._p_account[-2] * np.exp(self.spot)
        self.withdrawals_splits = 100  # numbers of points to discretize the control set in the dynamic approach.
        # step to discretize the control set in the dynamic approach.
        self.withdrawals_step = 0.10 * self.withdraw_amount
        # constants for the integration (step 2. II)
        self._d = np.real(-np.log(self.process.characteristic(-1j)))
        self._beta = np.exp((self.spot - self.div + self._d) * self.t_step) * (1 - self.fee * self.t_step)
        self._m = np.exp(-self.spot * self.t_step)
        self._xx = np.linspace(self.left_limit, self.right_limit, num=2 ** 8 + 1)
        self._dx = abs(self._xx[1] - self._xx[0])
        self._yy = self._beta * np.apply_along_axis(np.exp, 0, self._xx)
        self._density = np.apply_along_axis(np.vectorize(self.process.density), 0, self._xx)
        # Degree of the spline interpolation (1 - linear, 3 - cubic [default])
        self._x_dg = 3
        self._y_dg = 3
        # Price of the contract
        self._price = 0.0

    def set_gpoints(self, n_points=21):
        assert type(n_points) == int
        self.g_points = n_points
        self._g_account = np.linspace(0, self._max_g_account, self.g_points)

    def set_ppoints(self, n_points=21):
        assert type(n_points) == int
        self.p_points = n_points
        self._p_account = self._map_personal(np.linspace(0, 1, self.p_points))
        self._p_account[-1] = self._p_account[-2] * np.exp(self.spot)

    def set_penalty(self, penalty=0.05):
        assert type(penalty) == float
        self.penalty = penalty

    def set_maturity(self, maturity=20):
        assert type(maturity) == int
        self.maturity = maturity
        self._t_points = int(self.maturity / self.t_step)
        self.withdraw_amount = self.premium / self.maturity

    def set_integration_limits(self, right=0.5, left=-1.5):
        assert type(right) == float and type(left) == float
        self.right_limit = right
        self.left_limit = left
        self._xx = np.linspace(self.left_limit, self.right_limit, num=2 ** 8 + 1)
        self._dx = abs(self._xx[1] - self._xx[0])
        self._yy = self._beta * np.apply_along_axis(np.exp, 0, self._xx)
        self._density = np.apply_along_axis(np.vectorize(self.process.density), 0, self._xx)

    def set_fee(self, fee=0.0043):
        assert type(fee) == float
        self.fee = fee
        self._beta = np.exp((self.spot - self.div + self._d) * self.t_step) * (1 - self.fee * self.t_step)
        self._yy = self._beta * np.apply_along_axis(np.exp, 0, self._xx)

    def set_spot(self, spot=0.03):
        assert type(spot) == float
        self.spot = spot
        self._beta = np.exp((self.spot - self.div + self._d) * self.t_step) * (1 - self.fee * self.t_step)
        self._m = np.exp(-self.spot * self.t_step)
        self._yy = self._beta * np.apply_along_axis(np.exp, 0, self._xx)

    @staticmethod
    def _map_personal(t, a=100):
        # Given the personal account doesn't have an upper bound,
        # we simulate a "fake" unbounded interval [0, Inf] by mapping  [0, 1] by means of this function
        try:
            # This internal function takes either np.arrays or scalar values. In both cases
            # we check for the value 1 to prevent division by zero warnings.
            # Duck test. If it's not a np.array it raises TypeError
            res = np.array([a * x / (1 - x) if x != 1 else np.Inf for x in t])
        except TypeError:
            res = a * t / (1 - t) if t != 1 else np.Inf
        return res

    @staticmethod
    def _unmap_personal(p, a=100):
        # Un-maps [0, Inf] or sub-intervals of [0, Inf] back to [0, 1] or sub-intervals of [0, 1]
        return 1 - a / (p + a)

    @staticmethod
    @jit(float64[:](int32, float64[:], float64, float64[:, :], float64[:], float64), nopython=True, cache=True)
    def _calc_integrals(n, cf, m, interp_vals, density, dx):
        res = np.zeros(n)
        for row in np.arange(0, n):
            res[row] = cf[row] + \
                       m * trapz(interp_vals[row, :] * density, dx=dx)
        return res

    def price(self, method="static"):
        """

        :param method:
        :return:
        """

        @np.vectorize
        def value(a, w, ff):
            # Function to calculate the contract values at each point of the
            # guaranteed x personal account grid in a single point in time.
            # Since this is a numpy vectorized function (via decorator) we can apply it to
            # the whole numpy meshgrid.

            if a == 0 and w == 0:
                return 0.0
            # Define the control set based on the policyholder's  behavior.
            max_guaranteed_withdraw = min(self.withdraw_amount, a)
            withdrawals = np.repeat(max_guaranteed_withdraw, 2)
            if method == "dynamic":
                withdrawals = np.arange(0, a+self.withdrawals_step, self.withdrawals_step)

            elif method == "mixed":
                withdrawals = np.array([max_guaranteed_withdraw, w])

            # Calculate the Cashflows
            cash_flows = withdrawals - self.penalty * (withdrawals - max_guaranteed_withdraw).clip(0, np.Inf)
            # Calculate the Guaranteed account value
            idx = withdrawals > max_guaranteed_withdraw
            guaranteed_account = (a - withdrawals)
            pro_rata = a * (1 - withdrawals / w) if w != 0 else np.zeros(withdrawals.shape)
            guaranteed_account[idx] = np.minimum(guaranteed_account[idx], pro_rata[idx]).clip(0, np.Inf)
            # Calculate the Personal account values
            personal_account = (w - withdrawals).clip(0, np.Inf)
            personal_account = personal_account.reshape((personal_account.shape[0], 1)) * self._yy
            guaranteed_account = np.tile(guaranteed_account.reshape((guaranteed_account.shape[0], 1)),
                                         personal_account.shape[1])
            # Evaluate the spline on the above guaranteed and personal account values
            # The reshaping below is needed in order to pass vectors (x[i], y[i]) to the fast ev function
            # of RectBivariateSpline. This is faster than looping through the personal and guaranteed account arrays.
            n, m = personal_account.shape
            interpolated_values = ff.ev(personal_account.reshape(n * m), guaranteed_account.reshape(n * m))
            interpolated_values = interpolated_values.reshape((n, m))
            # Perform the integration
            # Since we loop through the interpolated_values array to calculate the integrals, let's use the jit compiled
            # function _calc_integrals to improve the performance to some extent.
            integrals = self._calc_integrals(n, cf=cash_flows, m=self._m, interp_vals=interpolated_values,
                                             density=self._density, dx=self._dx)
            return integrals.max()

        def calc_price(n, c_val):
            # Recursive function to calculate the guaranteed x personal account grid from T - 1 to 1
            if n == 1:
                interp_func = interpolate.RectBivariateSpline(self._p_account, self._g_account, c_val, kx=self._x_dg,
                                                              ky=self._y_dg)
                c_val = value(gg, pp, interp_func)
                return c_val
            else:
                # Interpolate the H x L triplets
                interp_func = interpolate.RectBivariateSpline(self._p_account, self._g_account, c_val, kx=self._x_dg,
                                                              ky=self._y_dg)
                # Compute the contract value at each point of the grid
                c_val = value(gg, pp, interp_func)
                return calc_price(n - 1, c_val)

        # *******
        #  MAIN
        # *******
        # Initial contract value
        gg, pp = np.meshgrid(self._g_account, self._p_account)
        ini_val = np.maximum(gg, pp)
        # Step 2. II Compute the contract value at each time step but t=0.
        # The calculation is done via recursion
        val = calc_price(self._t_points - 1, ini_val)

        # Contract value at inception (t=0)
        # Final interpolation
        val_func = interpolate.RectBivariateSpline(self._p_account, self._g_account, val, kx=self._x_dg, ky=self._y_dg)
        _YY = self.premium * self._yy
        _U = np.repeat(self.premium, len(self._yy))
        # Final integration
        final_integrand = val_func.ev(_YY, _U) * self._density
        self._price = self._m * trapz(final_integrand, dx=self._dx)
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
