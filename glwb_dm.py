from scipy import optimize
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class GLWB_DM:

    """
        VA contract with GLWB rider

        Class for a VA contract with Guarateed Life Withdrawal Benefit (GLWB) rider, so that the insured can withdraw
        from the account for all her lifetime regardless the  performance of the underlying fund.
        The fund is described by Levy stochastic processes while the mortality is deterministic.
        The contract key characteristics (e.g: fee, premium etc), financial parameters (e.g: risk free rate),
        fund and mortality processes are specified during the object initialization.
        The contract can be priced under "static", "mixed" and "dynamic" policyholder behaviors.
        Under the "static" behavior the insured can withdraw only at a fixed annual withdrawal rate. Under the "mixed"
        behavior she can withdraw at the specified rate or surrender the contract. Finally under the "dynamic" behavior,
        the policyholder can withdraw any amount she wants or surrender.
        The pricing is done by means of  Dynamic Programming (see References) by the "price" method.

        Parameters
        ----------
        financial_process : object of class  LevyProcess describing the underlying fund

        mortality_process: object of class DeterministicMortalityProcess describing the intensity of mortality process

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
        price: calculate the contract  value under the "static", "mixed" and "dynamic" approach.
        fair_fee: calculate the fair fee under the "static", "mixed" and "dynamic" approach.

        Examples
        --------

            import numpy as np
            from processes import CGMYProcess, Gompertz
            from glwb_dm import GLWB_DM

            cgmy = CGMYProcess(c=0.02, g=5, m=15, y=1.2)
            gompertz = Gompertz(theta=0.09782020, mu0=0.01076324)  # IPS55M
            glwb = GLWB_DM(cgmy, cir)

            glwb.set_maturity(53)
            glwb.set_spot(0.02)
            glwb.set_rollup(0.08)
            glwb.set_g_rate(0.03)
            glwb.penalty = 0.05
            glwb.set_fee(0.05)

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
        self.maturity = 53
        self.fee = 0.01
        self.spot = 0.02
        self.premium = 100
        self.rollup = 0.01
        self.g_rate = 0.02  # Withdrawal rate
        self.penalty = 0.02
        # Steps to discretize time, withdrawal base account, personal account and intensity of mortality
        self.t_step = 1  # time step
        self.g_points = 7  # Number of points to discretize the withdrawal base account
        self.p_points = 81  # Number of points to discretize the personal account

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise KeyError(k)

        # Internal data members
        # Number of points to discretize the time
        self._t_points = int(self.maturity / self.t_step)
        # Withdrawal base
        self._g_account = np.array([self.premium * (1 + self.rollup) ** j for j in range(0, self.g_points - 1)])
        self._g_account = np.insert(self._g_account, 0, 0)
        # Personal account
        self.k = 4
        self._max_p_account = self.k * self.premium # max personal account
        self._p_account, self._p_step = np.linspace(0.0, self._max_p_account, self.p_points, retstep=True)
        # Constants for the integration
        self._d = np.real(-np.log(self.financial_process.characteristic(-1j)))
        self._beta = np.exp(self.spot + self._d) * (1 - self.fee)
        # Outer integral
        self.outer_right_limit = 0.5
        self.outer_left_limit = -1.5
        self._num_outer_pts = 2 ** 6 + 1
        self._xx, self._dx = np.linspace(self.outer_left_limit, self.outer_right_limit,
                                         num=self._num_outer_pts, endpoint=True, retstep=True)
        self._financial_density = np.apply_along_axis(np.vectorize(self.financial_process.density), 0, self._xx)
        self._set_points()
        self._interp_values = {}
        # Initial contract value
        self.c_val = [np.zeros((self.p_points, self.g_points))] * (self.maturity+1)
        self._price = 0.0

        # Methods

    def _set_account_grid(self, method="static"):
        if method == "static" or method == "mixed":
            self._max_g_account = self.premium
            self.g_points = 2
            self._g_account = np.array([0, self.premium])
        elif method == "dynamic":
            self.g_points = 7
            self._g_account = np.array([self.premium * (1 + self.rollup) ** j for j in range(0, self.g_points-1)])
            self._g_account = np.insert(self._g_account, 0, 0)
        else:
            raise ValueError(f'method must be either static, mixed or dynamic')
        self._set_points()
        self.c_val = [np.zeros((self.p_points, self.g_points))] * (self.maturity+1)

    def set_maturity(self, maturity=55):
        self.maturity = maturity
        # Number of points to discretize the time
        self._t_points = int(self.maturity / self.t_step)

    def set_fee(self, fee=0.0043):
        assert type(fee) == float
        self.fee = fee
        self._beta = np.exp(self.spot + self._d) * (1 - self.fee)


    def set_spot(self, spot=0.02):
        self.spot = spot
        self._beta = np.exp(self.spot + self._d) * (1 - self.fee)

    def set_rollup(self, rollup=0.01):
        self.rollup = rollup

    def set_g_rate(self, g_rate=0.01):
        self.g_rate = g_rate

    def calc_points(self, w, a, wd):
        # Given the personal account value w, guaranteed account value a and withdrawal amount wd
        # calculate the points where to evaluate the interpolated contract value at each time step.
        # For each (w, a, wd) triple  this function returns _num_inner_pts * _num_outer_pts points
        # which will be needed in the double integral to calculate the new contract values.
        beta = self._beta * max(w - wd, 0)
        # if wd == 0:
        #     alpha = a * (1 + self.rollup)
        # elif 0 < wd <= self.g_rate * a:
        #     alpha = a
        # elif self.g_rate * a < wd <= w:
        #     alpha = a * ((w - wd) / (w - self.g_rate * a))
        alpha = 0
        if abs(wd) <= 1E-10:
            alpha = a * (1 + self.rollup)
        elif abs(wd - self.g_rate * a) <= 1E-10:
            alpha = a
        elif abs(wd - w) <= 1E-10 and w > self.g_rate * a:
            alpha = 0
        mbeta = beta * np.exp(self._xx)
        malpha = np.repeat(alpha, self._num_outer_pts)
        points = (mbeta, malpha)
        return points

    def _set_points(self):
        self._points = {}
        for a in self._g_account:
            for w in self._p_account:
                for wd in [0, self.g_rate * a, w]:
                    # print(f'w: {w}, a: {a}, wd: {wd}\n')
                    self._points[(w, a, wd)] = self.calc_points(w, a, wd)

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

        if method not in ["static", "mixed", "dynamic"]:
            raise ValueError(f'method must be either static, mixed or dynamic')

        self._set_account_grid(method=method)

        def calc_integral(shift, r, interp_vals, px, fin_density, deltax):
            res = shift + np.exp(-r) * px * np.trapz(interp_vals * fin_density, dx=deltax)
            return res

        def value(w, a, t):
            # Function to calculate the contract values at each point of the
            # base  x personal account in a single point in time.
            if method == "static":
                withdrawals = [self.g_rate * a]
            elif method == "dynamic":
                withdrawals = [0, self.g_rate * a, max(w, self.g_rate * a)]
            elif method == "mixed":
                withdrawals = [self.g_rate * a, max(w, self.g_rate * a)]
            integrals = []
            px = self.mortality_process.px(t + self.t_step) / self.mortality_process.px(t) if t != self.maturity else 0.0
            # px = self.mortality_process.px(t + self.t_step)
            qx = 1 - px
            for wd in withdrawals:
                shift = wd - self.penalty * max(wd - self.g_rate * a, 0) + \
                       qx * max(w - wd, 0) * (1 - self.fee)
                values = self._interp_values[(w, a, wd)]
                integrals.append(calc_integral(shift=shift, r=self.spot, interp_vals=values,
                                               px=px,
                                               fin_density=self._financial_density,deltax=self._dx))
            return np.max(integrals)

        # Step 2. II Compute the contract value at each time step but t=0.

        self.c_val = [np.zeros((self.p_points, self.g_points))] * (self.maturity+1)

        for t in np.arange(self._t_points, 0, -self.t_step):
            # t1 = datetime.utcnow()
            # print(f'Step {t}\n')
            interp_func = RegularGridInterpolator((self._p_account, self._g_account), self.c_val[t], bounds_error=False, fill_value=None)
            # print(self.c_val[t])
            for key in self._points.keys():
                self._interp_values[key] = interp_func(self._points[key])
            for k, w in enumerate(self._p_account):
                for j, a in enumerate(self._g_account):
                        self.c_val[t-1][k, j] = value(w, a, t)
            # t2 = datetime.utcnow()
            # d = t2 - t1
            # print(f'Step  {t} done in {d} secs\n')

        # Contract value at inception (t=0)
        # Final interpolation
        # t1 = datetime.utcnow()
        # print(f'Step 0\n')
        interp_func = RegularGridInterpolator((self._p_account, self._g_account), self.c_val[t], bounds_error=False, fill_value=None)
        final_shift = self.mortality_process.qx(1) * self.premium * (1 - self.fee)
        final_beta = self.premium * self._beta
        x = final_beta * np.exp(self._xx)
        y = np.repeat(self.premium, self._num_outer_pts)
        final_values = interp_func((x, y))
        # Final integration
        self._price = final_shift + np.exp(-self.spot) * self.mortality_process.px(1) * \
                       np.trapz(final_values * self._financial_density, dx=self._dx)
        # t2 = datetime.utcnow()
        # d = t2 - t1
        # print(f'Step 0 done in {d} secs\n')
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