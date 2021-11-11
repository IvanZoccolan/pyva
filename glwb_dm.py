import numpy as np
from scipy import optimize
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime

class GLWB_DM:

    def __init__(self, financial_process, mortality_process, **kwargs):
        self.financial_process = financial_process
        self.mortality_process = mortality_process
        self.maturity = 55
        self.fee = 0.0
        self.spot = 0.02
        self.premium = 100
        self.rollup = 0.0
        self.g_rate = 0.0  # Withdrawal rate
        self.penalty = 0.02
        # Steps to discretize time, withdrawal base account, personal account and intensity of mortality
        self.t_step = 1  # time step
        self.g_points = 21  # Number of points to discretize the withdrawal base account
        self.p_points = 21  # Number of points to discretize the personal account

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise KeyError(k)

        # Internal data members
        # Number of points to discretize the time
        self._t_points = int(self.maturity / self.t_step)
        # Withdrawal base
        self._max_g_account = self.premium * (1 + self.rollup) ** (self._t_points - 1)  # max guaranteed account
        self._g_account, self._g_step = np.linspace(0.0, self._max_g_account, self.g_points, retstep=True)
        # Personal account
        self.k = 10
        self._max_p_account = self.k * self._max_g_account  # max personal account
        self._p_account, self._p_step = np.linspace(0.0, self._max_p_account, self.p_points, retstep=True)
        # self._p_account = self._map_personal(np.linspace(0, 1, self.p_points))
        # self._p_account[-1] = self._p_account[-2] * np.exp(self.spot)
        # Constants for the integration
        self._d = np.real(-np.log(self.financial_process.characteristic(-1j)))
        self._beta = np.exp(self.spot + self._d) * (1 - self.fee)
        # Outer integral
        self.outer_right_limit = 0.5
        self.outer_left_limit = -1.5
        self._num_outer_pts = 2 ** 5 + 1
        self._xx, self._dx = np.linspace(self.outer_left_limit, self.outer_right_limit,
                                         num=self._num_outer_pts, endpoint=True, retstep=True)
        self._financial_density = np.apply_along_axis(np.vectorize(self.financial_process.density), 0, self._xx)
        self._points = {}
        for a in self._g_account:
            for w in self._p_account:
                for wd in [0, self.g_rate * a, w]:
                    self._points[(w, a, wd)] = self.calc_points(w, a, wd)
        self._interp_values = {}
        # Initial contract value
        self.c_val = np.zeros((self.p_points, self.g_points))
        self._price = 0.0

        # Methods

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

    def set_fee(self, fee=0.0043):
        assert type(fee) == float
        self.fee = fee
        self._beta = np.exp(self.spot + self._d) * (1 - self.fee)
        self._points = {}
        for a in self._g_account:
            for w in self._p_account:
                for wd in [0, self.g_rate * a, w]:
                    self._points[(w, a, wd)] = self.calc_points(w, a, wd)

    def set_spot(self, spot=0.02):
        self.spot = spot
        self._beta = np.exp(self.spot + self._d) * (1 - self.fee)
        self._points = {}
        for a in self._g_account:
            for w in self._p_account:
                for wd in [0, self.g_rate * a, w]:
                    self._points[(w, a, wd)] = self.calc_points(w, a, wd)

    def set_rollup(self, rollup=0.01):
        self.rollup = rollup
        self._max_g_account = self.premium * (1 + self.rollup) ** self.maturity  # max guaranteed account
        self._g_account, self._g_step = np.linspace(0, self._max_g_account, self.g_points, retstep=True)
        self._max_p_account = self.k * self._max_g_account  # max personal account
        self._p_account, self._p_step = np.linspace(0, self._max_p_account, self.p_points, retstep=True)
        self._points = {}
        for a in self._g_account:
            for w in self._p_account:
                for wd in [0, self.g_rate * a, w]:
                    self._points[(w, a, wd)] = self.calc_points(w, a, wd)
        self.c_val = np.empty((self.p_points, self.g_points))

    def set_g_rate(self, g_rate=0.01):
        self.g_rate = g_rate
        self._points = {}
        for a in self._g_account:
            for w in self._p_account:
                for wd in [0, self.g_rate * a, w]:
                    self._points[(w, a, wd)] = self.calc_points(w, a, wd)

    def calc_points(self, w, a, wd):
        # Given the personal account value w, guaranteed account value a and withdrawal amount wd
        # calculate the points where to evaluate the interpolated contract value at each time step.
        # For each (w, a, wd) triple  this function returns _num_inner_pts * _num_outer_pts points
        # which will be needed in the double integral to calculate the new contract values.
        beta = self._beta * max(w - wd, 0)
        if wd == 0:
            alpha = a * (1 + self.rollup)
        elif 0 < wd <= self.g_rate * a:
            alpha = a
        elif self.g_rate * a < wd <= w:
            alpha = a * ((w - wd) / (w - self.g_rate * a))
        mbeta = beta * np.exp(self._xx)
        malpha = np.repeat(alpha, self._num_outer_pts)
        points = (mbeta, malpha)
        return points

    def price(self, method="static"):
        """

        :param method:
        :return:
        """

        def calc_integral(shift, r, interp_vals, px, fin_density, deltax):
            res = shift + np.exp(-r) * px * simpson(interp_vals * fin_density, dx=deltax)
            return res

        def value(w, a, t):
            # Function to calculate the contract values at each point of the
            # base  x personal account in a single point in time.
            if method == "static":
                withdrawals = [self.g_rate * a]
            elif method == "dynamic":
                withdrawals = [0, self.g_rate * a, w]
            elif method == "mixed":
                withdrawals = [self.g_rate * a, w]
            integrals = []
            px = self.mortality_process.px(t+self.t_step) / self.mortality_process.px(t)
            qx = 1 - px
            for wd in withdrawals:
                shift = wd - self.penalty * max(wd - self.g_rate * a, 0) + \
                       qx * max(w - wd, 0) * (1 - self.fee)
                values = self._interp_values[(w, a, wd)]
                integrals.append(calc_integral(shift=shift, r=self.spot, interp_vals=values,
                                               px=px,
                                               fin_density=self._financial_density,
                                               deltax=self._dx)
                                 )
            return np.max(integrals)

        # Step 2. II Compute the contract value at each time step but t=0.
        for t in np.arange(self._t_points, 0, -self.t_step):
            t1 = datetime.utcnow()
            print(f'Step {t}\n')
            interp_func = RegularGridInterpolator((self._p_account, self._g_account), self.c_val, bounds_error=False, fill_value=None)
            for key in self._points.keys():
                self._interp_values[key] = interp_func(self._points[key])
            for k, w in enumerate(self._p_account):
                for j, a in enumerate(self._g_account):
                        self.c_val[k, j] = value(w, a, t)
            t2 = datetime.utcnow()
            d = t2 - t1
            print(f'Step  {t} done in {d} secs\n')

        # Contract value at inception (t=0)
        # Final interpolation
        t1 = datetime.utcnow()
        print(f'Step 0\n')
        interp_func = RegularGridInterpolator((self._p_account, self._g_account), self.c_val, bounds_error=False, fill_value=None)
        final_shift = self.mortality_process.qx(1) * self.premium * (1 - self.fee)
        final_beta = self.premium * self._beta
        x = final_beta * np.exp(self._xx)
        y = np.repeat(self.premium, self._num_outer_pts)
        final_values = interp_func((x, y))
        # Final integration
        self._price = final_shift + np.exp(-self.spot) * self.mortality_process.px(1) * \
                      simpson(final_values * self._financial_density, dx=self._dx)
        t2 = datetime.utcnow()
        d = t2 - t1
        print(f'Step 0 done in {d} secs\n')
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
