import numpy as np
from scipy import optimize
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime


class GLWB:

    """

    VA contract with GLWB rider

    Class for a VA contract with Guarateed Life Withdrawal Benefit (GLWB) rider, so that the insured can withdraw
    from the account for all her lifetime regardless the  performance of the underlying fund.
    The fund and the mortality are described by Levy stochastic processes. The contract key characteristics (e.g: fee,
    premium etc), financial parameters (e.g: risk free rate), fund and mortality processes are specified during the
    object initialization. The contract can be priced under "static", "mixed" and "dynamic" policyholder behaviors.
    Under the "static" behavior the insured can withdraw only at a fixed annual withdrawal rate. Under the "mixed"
    behavior she can withdraw at the specified rate or surrender the contract. Finally under the "dynamic" behavior, the
    policyholder can withdraw any amount she wants or surrender.
    The pricing is done by means of  Dynamic Programming (see References) by the "price" method.

    Notes:
    ------
    In order to prevent numerical instability a central scenario for the intensity of mortality should be passed to
    the object via the set_mu_space before calling its price method (see Examples).

    Parameters
    ----------
    financial_process : object of class  LevyProcess describing the underlying fund

    mortality_process: object of class StochasticMortalityProcess describing the intensity of mortality process

    maturity: int, maturity of the contract. It should be max possible age 110 - actual age of the insured

    fee: float, contract fee

    spot: float, risk-free spot rate

    premium: float, unique premium at contract inception. The premium is fully invested into the fund

    rollup: float, guaranteed  rate for the rollup of the initial premium invested.

    g_rate: float, guaranteed annual withdrawal rate ("static", "mixed")

    penalty: float, penalty applied in case the amount withdrawn is greater than the guaranteed annual amount

    Methods
    -------

    set_maturity: set the maturity
    set_fee: set the fee
    set_spot: set the spot risk free rate
    set_g_rate: set the guaranteed withdrawal rate
    set_rollup: set the rollup rate
    set_mu_space: set the discretized grid of the intensity of mortality for the Dynamic Programming algorithm.
        This should be thought as the "central scenario" of the stochastic intensity of mortality (see Exemples)
    price: calculate the contract  value under the "static", "mixed" and "dynamic" approach.
    fair_fee: calculate the fair fee under the "static", "mixed" and "dynamic" approach.

    Examples
    --------

        import numpy as np
        from processes import CGMYProcess, CirMortality, Gompertz
        from glwb_contract import GLWB

        cgmy = CGMYProcess(c=0.02, g=5, m=15, y=1.2)
        gompertz = Gompertz(theta=0.09782020, mu0=0.01076324)  # IPS55M
        glwb = GLWB_DM(cgmy, cir)

        glwb.set_maturity(53)
        glwb.set_spot(0.02)
        glwb.set_rollup(0.08)
        glwb.set_g_rate(0.03)
        glwb.penalty = 0.05
        glwb.set_fee(0.05)

        # Set the central scenario for the intensity of mortality stochastic process.
        # Given the process is a CIR, the "central" scenario is the path of corresponding Gompertz deterministic
        # intensity of mortality.

        mu_space = np.exp(gompertz.theta * np.arange(0, 54)) * gompertz.mu0

        glwb.set_mu_space(mu_space)

        # Price the contract under "dynamic" policyholder behavior.

        glwb.price("dynamic")

        References
        ----------
        [1] Bacinello, Maggistro, Zoccolan "The valuation of GLWB variable annuities with stochastic mortality and
        dynamic withdrawals", TBD.

    """

    def __init__(self, financial_process, mortality_process, **kwargs):
        self.financial_process = financial_process
        self.mortality_process = mortality_process
        self.maturity = 55
        self.fee = 0.01
        self.spot = 0.02
        self.premium = 100
        self.rollup = 0.01
        self.g_rate = 0.02  # Withdrawal rate
        self.penalty = 0.02

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise KeyError(k)

        # Internal data members

        # Steps to discretize time, withdrawal base account, personal account and intensity of mortality
        self.t_step = 1  # time step
        self.g_points = 21  # Number of points to discretize the withdrawal base account
        self.p_points = 21  # Number of points to discretize the personal account
        self.num_mu_pts = 21  # Number of points to discretize the support of the intensity of mortality
        self.min_mu = 2.925E-03  # minimum value for the intensity of mortality
        self.max_mu = 3.087  # maximum value for the intensity of mortality

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
        self._mu_space = np.logspace(self.min_mu, self.max_mu, self.num_mu_pts)
        # Constants for the integration
        self._d = np.real(-np.log(self.financial_process.characteristic(-1j)))
        self._beta = np.exp(self.spot + self._d) * (1 - self.fee)
        self._qx = {x: self.mortality_process.qx(t=1, mu=x) for x in self._mu_space}
        self._qx[self.mortality_process.mu0] = self.mortality_process.qx(t=1, mu=self.mortality_process.mu0)
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

        # Initial contract value
        self.c_val = np.zeros((self.p_points, self.g_points, self.num_mu_pts))
        self._price = 0.0

        # Methods

    def set_maturity(self, maturity=55):
        self.maturity = maturity
        # Number of points to discretize the time
        self._t_points = int(self.maturity / self.t_step)

    def set_fee(self, fee=0.0043):
        assert type(fee) == float
        self.fee = fee
        self._beta = np.exp(self.spot + self._d) * (1 - self.fee)
        self._points = {}
        for a in self._g_account:
            for w in self._p_account:
                for wd in [0, self.g_rate * a, w]:
                    self._points[(w, a, wd)] = self.calc_points(w, a, wd)
        self.c_val = np.zeros((self.p_points, self.g_points, self.num_mu_pts))

    def set_spot(self, spot=0.02):
        self.spot = spot
        self._beta = np.exp(self.spot + self._d) * (1 - self.fee)
        self._points = {}
        for a in self._g_account:
            for w in self._p_account:
                for wd in [0, self.g_rate * a, w]:
                    self._points[(w, a, wd)] = self.calc_points(w, a, wd)
        self.c_val = np.zeros((self.p_points, self.g_points, self.num_mu_pts))

    def set_rollup(self, rollup=0.01):
        self.rollup = rollup
        self._max_g_account = self.premium * (1 + self.rollup) ** (self._t_points - 1)  # max guaranteed account
        self._g_account, self._g_step = np.linspace(0, self._max_g_account, self.g_points, retstep=True)
        self._max_p_account = self.k * self._max_g_account  # max personal account
        self._p_account, self._p_step = np.linspace(0, self._max_p_account, self.p_points, retstep=True)
        self._points = {}
        for a in self._g_account:
            for w in self._p_account:
                for wd in [0, self.g_rate * a, w]:
                    self._points[(w, a, wd)] = self.calc_points(w, a, wd)
        self.c_val = np.zeros((self.p_points, self.g_points, self.num_mu_pts))

    def set_g_rate(self, g_rate=0.01):
        self.g_rate = g_rate
        self._points = {}
        for a in self._g_account:
            for w in self._p_account:
                for wd in [0, self.g_rate * a, w]:
                    self._points[(w, a, wd)] = self.calc_points(w, a, wd)
        self.c_val = np.zeros((self.p_points, self.g_points, self.num_mu_pts))

    def set_mu_space(self, mu_space=np.logspace(2E-3, 3.0, 21)):
        """
        Set the discretized grid of the intensity of mortality for the Dynamic Programming algorithm.

        Parameters:
        ----------
        mu_space ndarray
        array with the discretized grid of the intensity of mortality.

        Notes:
        -----
        The choice of this grid is key in getting numerically stable results from the price method.
        Given the intensity of mortality process, the ideal choice should be a sample path
        corresponding to a "central scenario".
        The volatility of the process is taken into account by means of the transition distribution
        described in [1] (Eq. 12)

        """
        assert type(mu_space) == np.ndarray
        self._mu_space = mu_space
        self.num_mu_pts = mu_space.shape[0]
        # Constants for the integration
        self._qx = {x: self.mortality_process.qx(t=1, mu=x) for x in self._mu_space}
        self._qx[self.mortality_process.mu0] = self.mortality_process.qx(t=1, mu=self.mortality_process.mu0)

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
        # Initial contract value
        self.c_val = np.zeros((self.p_points, self.g_points, self.num_mu_pts))
        self._price = 0.0

    def calc_points(self, w, a, wd):
        # Given the personal account value w, guaranteed account value a and withdrawal amount wd
        # calculate the points where to evaluate the interpolated contract value at each time step.
        # For each (w, a, wd) triple  this function returns _num_inner_pts * _num_outer_pts points
        # which will be needed in the double integral to calculate the new contract values.
        beta = self._beta * max(w - wd, 0)
        alpha = 0
        if wd == 0:
            alpha = a * (1 + self.rollup)
        elif wd == self.g_rate * a:
            alpha = a
        elif wd == w and w > self.g_rate * a:
            alpha = 0
        mbeta = np.repeat(beta * np.exp(self._xx), self.num_mu_pts * self._num_inner_pts).reshape((self._total, 1))
        malpha = np.repeat(alpha, self._total).reshape((self._total, 1))
        points = (mbeta, malpha, self._yyy)
        return points

    def price(self, method="static"):
        """
        Price the contract under different policyholder behaviors.

        Parameters:
        -----------
        method string:  Approach used in pricing, it can be either "static" (default), "mixed"  or "dynamic".

        Returns:
        -------
        float:  value or price of the contract

        """

        assert method in ["static", "mixed", "dynamic"]

        def calc_integral(shift, r, interp_vals, mort_coeff, mort_cpdf, fin_density, deltax, deltay):
            zz = np.apply_along_axis(lambda row: mort_coeff * row * mort_cpdf, 1, interp_vals)
            zz = np.apply_along_axis(lambda col: fin_density * col, 0, zz)
            res = shift + np.exp(-r) * simpson(simpson(zz, dx=deltay), dx=deltax)
            return res

        def value(w, a, mu):
            # Function to calculate the contract values at each point of the
            # base  x personal account x mu space grid in a single point in time.
            if method == "static":
                withdrawals = [self.g_rate * a]
            elif method == "dynamic":
                withdrawals = [0, self.g_rate * a, max(w, self.g_rate * a)]
            elif method == "mixed":
                withdrawals = [self.g_rate * a, max(w, self.g_rate * a)]
            mu_idx, = (self._mu_space == mu).nonzero()
            mu_idx = mu_idx.item()
            bottom = mu_idx * self._num_inner_pts
            top = bottom + self._num_inner_pts
            integrals = []
            for wd in withdrawals:
                shift = wd - self.penalty * max(wd - self.g_rate * a, 0) + \
                        self._qx[mu] * max(w - wd, 0) * (1 - self.fee)
                values = self._interp_values[(w, a, wd)][:, bottom:top]
                integrals.append(calc_integral(shift=shift, r=self.spot, interp_vals=values,
                                                     mort_coeff=self.mortality_coeff[mu_idx, :],
                                                     mort_cpdf=self._mortality_cpdf[mu_idx, :],
                                                     fin_density=self._financial_density,
                                                     deltax=self._dx,
                                                     deltay=self._dy[mu_idx])
                                 )

            return np.max(integrals)

        # Step 2. II Compute the contract value at each time step but t=0.
        for t in np.arange(self._t_points, 0, -self.t_step):
            t1 = datetime.utcnow()
            print(f'Step {t}\n')
            if t == self._t_points:
                for k, w in enumerate(self._p_account):
                    for j, a in enumerate(self._g_account):
                        if method == "static":
                            withdrawals = [self.g_rate * a]
                        elif method == "dynamic":
                            withdrawals = [0, self.g_rate * a, w]
                        elif method == "mixed":
                            withdrawals = [self.g_rate * a, w]
                        val = []
                        for wd in withdrawals:
                            val.append(wd - self.penalty * max(wd - self.g_rate * a, 0) +
                                       max(w - wd, 0) * (1 - self.fee))
                        self.c_val[k, j, :] = np.max(val)
            else:
                interp_func = RegularGridInterpolator((self._p_account, self._g_account, self._mu_space), self.c_val, bounds_error=False, fill_value=None)
                for key in self._points.keys():
                    x, y, z = self._points[key]
                    values = interp_func((x, y, z)).reshape((self._num_outer_pts, self.num_mu_pts * self._num_inner_pts))
                    self._interp_values[key] = values
                for k, w in enumerate(self._p_account):
                    for j, a in enumerate(self._g_account):
                        for h, mu in enumerate(self._mu_space):
                            self.c_val[k, j, h] = value(w, a, mu)
            t2 = datetime.utcnow()
            d = t2 - t1
            print(f'Step  {t} done in {d} secs\n')

        # Contract value at inception (t=0)
        # Final interpolation
        t1 = datetime.utcnow()
        print(f'Step 0\n')
        interp_func = RegularGridInterpolator((self._p_account, self._g_account, self._mu_space), self.c_val,
                                              bounds_error=False, fill_value=None)
        final_shift = self._qx[self.mortality_process.mu0] * self.premium * (1 - self.fee)
        final_beta = self.premium * self._beta
        left, right = self.mortality_process.cond_support(mu=self.mortality_process.mu0, dt=1)
        yy, dy = np.linspace(left, right, num=self._num_inner_pts, endpoint=True, retstep=True)
        tot_pts = self._num_outer_pts*self._num_inner_pts
        x = np.repeat(final_beta * np.exp(self._xx), self._num_inner_pts).reshape((tot_pts, 1))
        y = np.repeat(self.premium, tot_pts).reshape((tot_pts, 1))
        z = np.tile(yy, self._num_outer_pts).reshape((tot_pts, 1))
        final_values = interp_func((x, y, z)).reshape((self._num_outer_pts, self._num_inner_pts))
        # Final integration
        mortality_cpdf = np.apply_along_axis(
            lambda y: self.mortality_process.cpdf(y, mu=self.mortality_process.mu0, dt=1), 0, yy)
        mortality_coeff = np.apply_along_axis(lambda y: np.exp(-0.5 * (self.mortality_process.mu0 + y)), 0, yy)
        self._price = calc_integral(shift=final_shift, r=self.spot, interp_vals=final_values,
                                    mort_coeff=mortality_coeff, mort_cpdf=mortality_cpdf,
                                    fin_density=self._financial_density, deltax=self._dx, deltay=dy)
        t2 = datetime.utcnow()
        d = t2 - t1
        print(f'Step 0 done in {d} secs\n')
        return self._price

    def fair_fee(self, a=0, b=1, tol=1e-4, method="static"):
        """
        Calculate the fair fee.

        Params:
        ------
        a, b float: with the interval extremes where to search for the fair fee.
        method string:  Approach used in pricing, it can be either "static" (default), "mixed"  or "dynamic".

        Returns:
        -------
        float with the fair fee.

        Notes:
        -----
        The fee is searched by the Brent's root finding method.
        """

        def fun(x):
            self.set_fee(x)
            return self.price(method=method) - self.premium

        return optimize.brentq(fun, a=a, b=b, xtol=tol, maxiter=50)

