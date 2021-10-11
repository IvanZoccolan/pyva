import numpy as np
from scipy import optimize
from numba import jit, float64
from fast_interp import interp3d
from datetime import datetime
import sys

class GLWB:

    def __init__(self, financial_process, mortality_process, **kwargs):
        self.financial_process = financial_process
        self.mortality_process = mortality_process
        self.maturity = 55
        self.fee = 0.01
        self.spot = 0.02
        self.premium = 100
        self.rollup = 0.01
        self.g_rate = 0.04  # Withdrawal rate
        self.penalty = 0.02
        # Steps to discretize time, withdrawal base account, personal account and intensity of mortality
        self.t_step = 1  # time step
        self.g_points = 21  # Number of points to discretize the withdrawal base account
        self.p_points = 21  # Number of points to discretize the personal account
        self.num_mu_pts = 21  # Number of points to discretize the support of the intensity of mortality
        self.min_mu = self.mortality_process.mu0
        self.max_mu = 10  # maximum value for the intensity of mortality

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
        # Mortality space
        self._mu_space, self._mu_step = np.linspace(self.min_mu, self.max_mu, self.num_mu_pts, retstep=True)
        # Constants for the integration
        self._d = np.real(-np.log(self.financial_process.characteristic(-1j)))
        self._beta = np.exp(self.spot + self._d) * (1 - self.fee)
        self._qx = {x: self.mortality_process.qx(t=1, mu=x) for x in self._mu_space}
        # Outer integral
        self.outer_right_limit = 0.5
        self.outer_left_limit = -1.5
        self._num_outer_pts = 2 ** 5 + 1
        self._xx, self._dx = np.linspace(self.outer_left_limit, self.outer_right_limit,
                                         num=self._num_outer_pts, endpoint=True, retstep=True)
        self._financial_density = np.apply_along_axis(np.vectorize(self.financial_process.density), 0, self._xx)
        # Inner integral
        self._num_inner_pts = 2 ** 5 + 1
        # Define the inner integration limits specific to each value of the intensity of mortality
        # discretized space.
        _yy = []
        _dy = []
        for idx in np.arange(0, self.num_mu_pts):
            left, right = self.mortality_process.cond_support(mu=self._mu_space[idx], dt=1)
            yy, dy = np.linspace(left, right, num=self._num_inner_pts, endpoint=True, retstep=True)
            _yy.append(yy)
            _dy.append(dy)
        self._yy = np.array(_yy).reshape(self.num_mu_pts * self._num_inner_pts)
        self._dy = np.array(_dy).reshape(self.num_mu_pts)

        self._mortality_cpdf = np.empty((self.num_mu_pts, self._num_inner_pts))
        self.mortality_coeff = np.empty((self.num_mu_pts, self._num_inner_pts))
        self._total = self.num_mu_pts * self._num_inner_pts * self._num_outer_pts
        self._yyy = np.tile(self._yy, self._num_outer_pts).reshape((self._total, 1))
        for i in np.arange(0, self.num_mu_pts):
            for j in np.arange(0, self._num_inner_pts):
                self._mortality_cpdf[i, j] = self.mortality_process.cpdf(self._yy[i * self._num_inner_pts + j],
                                                                         mu=self._mu_space[i], dt=1)
                self.mortality_coeff[i, j] = np.exp(-0.5 * (self._mu_space[i] + self._yy[i * self._num_inner_pts + j]))
        self._points = {}
        for a in self._g_account:
            for w in self._p_account:
                for wd in [0, self.g_rate * a, w]:
                    self._points[(w, a, wd)] = self.calc_points(w, a, wd)
        self._interp_values = {}
        self.c_val = np.empty((self.p_points, self.g_points, self.num_mu_pts))
        self._price = 0.0

        # Methods

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
        self.c_val = np.empty((self.p_points, self.g_points, self.num_mu_pts))

    def set_g_rate(self, g_rate=0.01):
        self.g_rate = g_rate
        self._points = {}
        for a in self._g_account:
            for w in self._p_account:
                for wd in [0, self.g_rate * a, w]:
                    self._points[(w, a, wd)] = self.calc_points(w, a, wd)

    @staticmethod
    @jit(float64(float64, float64[:, :], float64[:], float64[:], float64[:], float64, float64), nopython=True,
         cache=True)
    def _calc_integral(shift, interp_vals, mort_coeff, mort_cpdf, fin_density, deltax, deltay):
        inner_integrals = np.zeros(interp_vals.shape[0])
        for row in np.arange(0, interp_vals.shape[0]):
            inner_integrals[row] = np.trapz(mort_coeff * interp_vals[row, :] * mort_cpdf, dx=deltay)
        res = shift + np.trapz(inner_integrals * fin_density, dx=deltax)
        return res

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
        mbeta = np.repeat(beta * np.exp(self._xx), self.num_mu_pts * self._num_inner_pts).reshape((self._total, 1))
        malpha = np.repeat(alpha, self._total).reshape((self._total, 1))
        points = (mbeta, malpha, self._yyy)
        return points

    def price(self, method="static"):
        """

        :param method:
        :return:
        """

        def value(w, a, mu):
            # Function to calculate the contract values at each point of the
            # base  x personal account x mu space grid in a single point in time.
            # Since this is a numpy vectorized function (via decorator) we can apply it to
            # the whole numpy meshgrid.
            if method == "static":
                withdrawals = [self.g_rate * a]
            elif method == "dynamic":
                withdrawals = [0, self.g_rate * a, w]
            elif method == "mixed":
                withdrawals = [self.g_rate * a, w]
            mu_idx, = (self._mu_space == mu).nonzero()
            mu_idx = mu_idx.item()
            bottom = mu_idx * self._num_inner_pts
            top = bottom + self._num_inner_pts
            integrals = []
            for wd in withdrawals:
                shift = wd - self.penalty * max(wd - self.g_rate * a, 0) + \
                        self._qx[mu] * max(w - wd, 0) * (1 - self.fee)
                values = self._interp_values[(w, a, wd)][:, bottom:top]
                integrals.append(self._calc_integral(shift=shift, interp_vals=values,
                                                     mort_coeff=self.mortality_coeff[mu_idx, :],
                                                     mort_cpdf=self._mortality_cpdf[mu_idx, :],
                                                     fin_density=self._financial_density,
                                                     deltax=self._dx,
                                                     deltay=self._dy[mu_idx])
                                 )
            res = np.max(integrals)
            return res

        def calc_price(n, c_val):
            # Recursive function to calculate the personal account x base x  mu space grid from T - 1 to 1
            if n == 1:
                t1 = datetime.utcnow()
                print(f'Step {n}\n')
                interp_func = interp3d(a=[0.0, 0.0, self.min_mu],
                                       b=[self._max_p_account, self._max_g_account, self.max_mu],
                                       h=[self._p_step, self._g_step, self._mu_step],
                                       f=c_val, k=1)
                for key in self._points.keys():
                    x, y, z = self._points[key]
                    values = interp_func(x, y, z).reshape((self._num_outer_pts, self.num_mu_pts * self._num_inner_pts))
                    if np.isnan(np.sum(values)):
                        print("Error NaNs produced\n")
                        sys.exit(1)
                    self._interp_values[key] = values
                for k, w in enumerate(self._p_account):
                    for j, a in enumerate(self._g_account):
                        for h, mu in enumerate(self._mu_space):
                            self.c_val[k, j, h] = value(w, a, mu)
                t2 = datetime.utcnow()
                d = t2 - t1
                print(f'Step  {n} done in {d} secs\n')
                return self.c_val
            else:
                t1 = datetime.utcnow()
                print(f'Step {n}\n')
                # Interpolate the H x L x K quadruplets
                interp_func = interp3d(a=[0.0, 0.0, self.min_mu],
                                       b=[self._max_p_account, self._max_g_account, self.max_mu],
                                       h=[self._p_step, self._g_step, self._mu_step],
                                       f=c_val, k=1)
                for key in self._points.keys():
                    x, y, z = self._points[key]
                    values = interp_func(x, y, z).reshape((self._num_outer_pts, self.num_mu_pts * self._num_inner_pts))
                    if np.isnan(np.sum(values)):
                        print("Error NaNs produced\n")
                        sys.exit(1)
                    self._interp_values[key] = values
                # Compute the contract value at each point of the grid
                for k, w in enumerate(self._p_account):
                    for j, a in enumerate(self._g_account):
                        for h, mu in enumerate(self._mu_space):
                            self.c_val[k, j, h] = value(w, a, mu)
                t2 = datetime.utcnow()
                d = t2 - t1
                print(f'Step  {n} done in {d} secs\n')
                return calc_price(n - 1, self.c_val)

        # Initial contract value
        ini_val = np.empty((self.p_points, self.g_points, self.num_mu_pts))
        for k, w in enumerate(self._p_account):
            for j, a in enumerate(self._g_account):
                for h, mu in enumerate(self._mu_space):
                    ini_val[k, j, h] = 0.0

        # Step 2. II Compute the contract value at each time step but t=0.
        # The calculation is done via recursion
        val = calc_price(self._t_points, ini_val)

        # Contract value at inception (t=0)
        # Final interpolation
        t1 = datetime.utcnow()
        print(f'Step 0\n')

        interp_func = interp3d(a=[0.0, 0.0, self.min_mu],
                               b=[self._max_p_account, self._max_g_account, self.max_mu],
                               h=[self._p_step, self._g_step, self._mu_step],
                               f=val, k=1)

        final_shift = self._qx[self.mortality_process.mu0] * self.premium * (1 - self.fee)
        final_beta = self.premium * self._beta
        x = np.repeat(final_beta * np.exp(self._xx), self.num_mu_pts * self._num_inner_pts).reshape((self._total, 1))
        y = np.repeat(self.premium, self._total).reshape((self._total, 1))
        z = self._yyy
        final_values = interp_func(x, y, z).reshape((self._num_outer_pts, self.num_mu_pts * self._num_inner_pts))
        # Final integration
        self._price = self._calc_integral(shift=final_shift, interp_vals=final_values[:, 0:self._num_inner_pts],
                                          mort_coeff=self.mortality_coeff[0, :],
                                          mort_cpdf=self._mortality_cpdf[0, :],
                                          fin_density=self._financial_density,
                                          deltax=self._dx,
                                          deltay=self._dy[0])
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
