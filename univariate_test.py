try:
    # note that type annotation have no performance gain
    from typing import NewType, Any, List, Dict, Union, Optional
    from abc import ABC, abstractmethod
    import numpy as np
    from math import sqrt, pow, log
    import scipy as sp
    import scipy.special as ss
    import matplotlib.pyplot as plt
except Exception as e:
    print("some modules are missing {}".format(e))

T = [int, float, np.int16, np.int32, np.int64,
     np.float16, np.float32, np.float64]
number = NewType('number', T)


class Base(ABC):
    def __init__(self, data: List[number]) -> None:
        self.data = data

    def plot(self,
             x: number,
             y: number,
             xlim: number = None,
             ylim: number = None,
             xlabel: str = None,
             ylabel: str = None) -> None:
        if ylim is not None:
            plt.ylim(0, ylim)
        if xlim is not None:
            plt.xlim(-xlim, xlim)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.plot(x, y, "black", alpha=0.5)
        plt.show()

    def logpdf(self, pdf: Union[number, List[number]]) -> Union[number, str]:
        return np.log(pdf)

    def logcdf(self, cdf: Union[number, List[number]]) -> Union[number, str]:
        return np.log(cdf)

    def pvalue(self) -> Union[number, str]:
        return "unsupported"

    def confidence_interval(self) -> Union[number, str]:
        return "currently unsupported"

    def rvs(self) -> Union[number, str]:
        # (adaptive) rejection sampling implementation
        """
        returns random variate samples default (unsupported)
        """
        return "currently unsupported"

    def mean(self) -> Union[number, str]:
        """
        returns mean default (unsupported)
        """
        return "unsupported"

    def median(self) -> Union[number, str]:
        """
        returns median default (unsupported)
        """
        return "unsupported"

    def mode(self) -> Union[number, str]:
        """
        returns mode default (unsupported)
        """
        return "unsupported"

    def var(self) -> Union[number, str]:
        """
        returns variance default (unsupported)
        """
        return "unsupported"

    def std(self) -> Union[number, str]:
        """
        returns the std default (undefined)
        """
        return "unsupported"

    def skewness(self) -> Union[number, str]:
        """
        returns skewness default (unsupported)
        """
        return "unsupported"

    def kurtosis(self) -> Union[number, str]:
        """
        returns kurtosis default (unsupported)
        """
        return "unsupported"

    def entropy(self) -> Union[number, str]:
        """
        returns entropy default (unsupported)
        """
        return "unsupported"

    def stdnorm_pdf(self, x: number) -> Union[number, str]:
        return np.exp(-pow(x, 2) / 2) / sqrt(2 * np.pi)

    def stdnorm_cdf(self, x: number) -> Union[number, str]:
        return sp.integrate.quad(self.stdnorm_pdf, -np.inf, x)[0]

    def stdnorm_cdf_inv(self, x: number, p: number, mean=0, std=1) -> Union[number, str]:
        """
        qunatile function of the normal cdf. Note thatn p can only have values between (0,1).
        defaults to standard normal but can be expressed more generally.
        """
        return mean + std * sqrt(2) * ss.erfinv(2 * p - 1)


class Uniform:
    """
    This class contains methods concerning the Continuous Uniform Distribution.

    Args:

        a(int): lower limit of the distribution
        b(int): upper limit of the distribution

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.

    Referene:
    - Weisstein, Eric W. "Uniform Distribution." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/UniformDistribution.html
    """

    def __init__(self, a: number, b: number) -> None:
        self.a = a
        self.b = b

    def pdf(self, plot=False, xlim=None, ylim=None, xlabel=None, ylabel=None) -> number:
        """
        Args:

            plot (bool): returns plot if true.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.

        Returns:
            either plot of the distribution or probability density evaluation at a to b.
        """
        a = self.a
        b = self.b
        threshold = b - a

        def _generator(a, b, x): return 1 / (b - a) if a <= x and x <= b else 0
        if plot:
            x = np.linspace(a, b, threshold)
            y = np.array([_generator(a, b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(a, b, np.abs(b - a))

    def cdf(self, plot=False, xlim=None, ylim=None, xlabel=None, ylabel=None) -> number:
        """
        Args:

            plot (bool): returns plot if true.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.

        Returns:
            either plot of the distribution or probability density evaluation at a to b.
        """
        a = self.a
        b = self.b
        threshold = b - a

        def _generator(a, b, x):
            if x < a:
                return 0
            if (a <= x and x <= b):
                return (x - a) / (b - a)
            if x > b:
                return 1

        if plot:
            x = np.linspace(a, b, threshold)
            y = np.array([_generator(a, b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(a, b, threshold)  # what does it really say?

    def mean(self) -> Union[number, str]:
        """
        Returns: Mean of the Uniform distribution.
        """
        return 1 / 2 * (self.a + self.b)

    def median(self) -> Union[number, str]:
        """
        Returns: Median of the Uniform distribution.
        """
        return 1 / 2 * (self.a + self.b)

    def mode(self) -> Union[number, str]:
        """
        Returns: Mode of the Uniform distribution.

        Note that the mode is any value in (a,b)
        """
        return (self.a, self.b)

    def var(self) -> Union[number, str]:
        """
        Returns: Variance of the Uniform distribution.
        """
        return (1 / 12) * (self.b - self.a)**2

    def std(self) -> Union[number, str]:
        """
        Returns: Standard deviation of the Uniform distribution.
        """
        return sqrt(self.var())

    def skewness(self) -> Union[number, str]:
        """
        Returns: Skewness of the Uniform distribution.
        """
        return 0

    def kurtosis(self) -> Union[number, str]:
        """
        Returns: Kurtosis of the Uniform distribution.
        """
        return -6 / 5

    def entropy(self) -> Union[number, str]:
        """
        Returns: entropy of uniform Distirbution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return np.log(self.b-self-a)

    def summary(self) -> None:
        """
        Returns: Summary statistic regarding the Uniform distribution.
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class F(Base):
    """
    This class contains methods concerning the F-distribution.

    Args:

        x(float | [0,infty)): random variable
        df1(int | x>0): first degrees of freedom
        df2(int | x>0): second degrees of freedom

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.

    References:
    - Mood, Alexander; Franklin A. Graybill; Duane C. Boes (1974).
    Introduction to the Theory of Statistics (Third ed.). McGraw-Hill. pp. 246–249. ISBN 0-07-042864-6.

    - Weisstein, Eric W. "F-Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/F-Distribution.html
    - NIST SemaTech (n.d.). F-Distribution. Retrived from https://www.itl.nist.gov/div898/handbook/eda/section3/eda3665.htm
    """

    def __init__(self, x: number, df1: int, df2: int):
        if isinstance(df1, int) == False or df1 < 0:
            raise Exception(
                'degrees of freedom(df) should be a whole number. Entered value for df1: {}'.format(df1))
        if isinstance(df2, int) == False or df2 < 0:
            raise Exception(
                'degrees of freedom(df) should be a whole number. Entered value for df2: {}'.format(df2))
        if x < 0:
            raise Exception(
                'random variable should be greater than 0. Entered value for x:{}'.format(x))

        self.x = x
        self.df1 = df1
        self.df2

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> number:
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either probability density evaluation for some point or plot of F-distribution.
        """
        df1 = self.df1
        df2 = self.df2
        randvar = self.x

        def _generator(x, df1, df2): return (1 / ss.beta(
            df1 / 2, df2 / 2)) * np.power(df1 / df2, df1 / 2) * np.power(
                x, df1 / 2 - 1) * np.power(1 +
                                           (df1 / df2) * x, -((df1 + df2) / 2))

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, df1, df2) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(randvar, df1, df2)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either cumulative distribution evaluation for some point or plot of F-distribution.
        """
        k = self.df2/(self.df2+self.df1*self.x)
        def _generator(x, df1, df2): return 1-ss.betainc(df1/2, df2/2, x)

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, self.df1, self.df2) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(k, self.df1, self.df2)

    def p_val(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the F-distribution evaluated at some random variable.
        """
        if x_lower < 0:
            x_lower = 0
        if x_upper is None:
            x_upper = self.x

        def _cdf_def(x, df1, df2): return 1 - \
            ss.betainc(df1/2, df2/2, df2/(df2+df1*x))

        return _cdf_def(x_upper, self.df1, self.df2) - _cdf_def(x_lower, self.df1, self.df2)

    def confidence_interval(self) -> Optional[number]:
        pass

    def mean(self) -> Union[number, str]:
        """
        Returns: Mean of the F-distribution.
        """
        if self.df3 > 2:
            return self.df2 / (self.df2 - 2)
        return "undefined"

    def mode(self) -> Union[number, str]:
        """
        Returns: Mode of the F-distribution.
        """
        df1 = self.df1
        df2 = self.df2
        if df1 > 2:
            return (df2 * (df1 - 2)) / (df1 * (df2 + 2))
        return "undefined"

    def var(self) -> Union[number, str]:
        """
        Returns: Variance of the F-distribution.
        """
        df1 = self.df1
        df2 = self.df2
        if df2 > 4:
            return (2 * (df2**2) * (df1 + df2 - 2)) / (df1 * ((df2 - 2)**2) *
                                                       (df2 - 4))
        return "undefined"

    def std(self) -> Union[number, str]:
        """
        Returns: Standard deviation of the F-distribution.
        """
        if self.var() == "undefined":
            return "undefined"
        return sqrt(self.var())

    def skewness(self) -> Union[number, str]:
        """
        Returns: Skewness of the F-distribution.
        """
        df1 = self.df1
        df2 = self.df2
        if df2 > 6:
            return ((2 * df1 + df2 - 2) * np.sqrt(8 * (df2 - 4))) / (
                (df2 - 6) * np.sqrt(df1 * (df1 + df2 - 2)))
        return "undefined"

    def entropy(self) -> Union[number, str]:
        """
        Returns: differential entropy of F-distribution.

        Reference: Lazo, A.V.; Rathie, P. (1978). "On the entropy of continuous probability distributions". IEEE Transactions on Information Theory
        """
        df1 = self.df1
        df2 = self.df2
        return np.log(ss.gamma(df1/2))+np.log(ss.gamma(df2/2))-np.log(ss.gamma((df1+df2)/2))+(1-df1/2)*ss.digamma(1+df1/2)-(1-df2/2)*ss.digamma(1+df2/2)+(df1+df2)/2*ss.digamma((df1+df2)/2)+np.log(df1/df2)

    def summary(self) -> None:
        """
        Returns:  summary statistic regarding the F-distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Chisq(Base):
    """
    This class contains methods concerning the Chi-square distribution.

    Args:

        x(float): random variable.
        df(int): degrees of freedom.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.

    References:
    - Weisstein, Eric W. "Chi-Squared Distribution." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/Chi-SquaredDistribution.html
    - Wikipedia contributors. (2020, December 13). Chi-square distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 04:37, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Chi-square_distribution&oldid=994056539
    """

    def __init__(self, df, x):
        if isinstance(df, int) == False:
            raise Exception(
                'degrees of freedom(df) should be a whole number. Entered value for df: {}'.format(df))
        self.x = x
        self.df = df

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either probability density evaluation for some point or plot of Chi square-distribution.

        """
        df = self.df
        randvar = self.x

        def _generator(x, df): return (1 / (np.power(2, (df / 2) - 1) * ss.gamma(
            df / 2))) * np.power(x, df - 1) * np.exp(-x**2 / 2)
        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return _generator(randvar, df)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either cumulative distribution evaluation for some point or plot of Chi square-distribution.
        """
        def _generator(x, df): return ss.gammainc(df / 2, x / 2)
        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, self.df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.randvar, self.df)

    def p_val(self, x_lower=-np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. If not defined defaults to random variable x. Optional.
            args(list of float): pvalues of each elements from the list

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Chi square distribution evaluated at some random variable.
        """
        def _cdf_def(x, df): return ss.gammainc(df / 2, x / 2)
        if x_upper != None:
            if x_lower > x_upper:
                raise Exception('x_lower should be less than x_upper.')
            return _cdf_def(x_upper, self.df) - _cdf_def(x_lower, self.df)
        return _cdf_def(self.randvar, self.df)

    def mean(self) -> Union[number, str]:
        """
        Returns: Mean of the Chi-square distribution.
        """
        return self.df

    def median(self) -> Union[number, str]:
        """
        Returns: Median of the Chi-square distribution.
        """
        return self.k * (1 - 2 / (9 * self.k))**3

    def var(self) -> Union[number, str]:
        """
        Returns: Variance of the Chi-square distribution.
        """
        return 2 * self.df

    def std(self) -> Union[number, str]:
        """
        Returns: Standard deviation of the Chi-square distribution.
        """
        return sqrt(self.var())

    def skewness(self) -> Union[number, str]:
        """
        Returns: Skewness of the Chi-square distribution.
        """
        return np.sqrt(8 / self.df)

    def kurtosis(self) -> Union[number, str]:
        """
        Returns: Kurtosis of the Chi-square distribution.
        """
        return 12 / self.df

    def entropy(self) -> Union[number, str]:
        """
        Returns: differential entropy of Chi-square distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        df = self.df
        return df/2+np.log(2*ss.gamma(df/2))+(1-df/2)*ss.digamma(df/2)

    def summary(self) -> None:
        """
        Returns: Summary statistic regarding the Chi-square distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Chi(Base):
    """
    This class contains methods concerning the Chi distribution.

    Args:

        x(float): random variable.
        df(int | x>0): degrees of freedom.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.

    References:
    - Weisstein, Eric W. "Chi Distribution." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/ChiDistribution.html
    - Wikipedia contributors. (2020, October 16). Chi distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 10:35, January 2, 2021, from https://en.wikipedia.org/w/index.php?title=Chi_distribution&oldid=983750392
    """

    def __init__(self, df, x):
        if isinstance(df, int) == False:
            raise Exception(
                'degrees of freedom(df) should be a whole number. Entered value for df: {}'.format(df))
        self.x = x
        self.df = df

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> number:
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either probability density evaluation for some point or plot of Chi-distribution.

        """
        df = self.df
        randvar = self.x

        def _generator(x, df): return (1 / (np.power(2, (df / 2) - 1) * ss.gamma(
            df / 2))) * np.power(x, df - 1) * np.exp(-x**2 / 2)
        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return _generator(randvar, df)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either cumulative distribution evaluation for some point or plot of Chi-distribution.
        """
        def _generator(x, df): return ss.gammainc(df/2, x**2/2)
        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, self.df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.randvar, self.df)

    def p_val(self, x_lower=-np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. If not defined defaults to random variable x. Optional.
            args(list of float): pvalues of each elements from the list

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Chi distribution evaluated at some random variable.
        """
        def _cdf_def(x, df): return ss.gammainc(df/2, x**2/2)
        if x_upper != None:
            if x_lower > x_upper:
                raise Exception('x_lower should be less than x_upper.')
            return _cdf_def(x_upper, self.df) - _cdf_def(x_lower, self.df)
        return _cdf_def(self.randvar, self.df)

    def mean(self) -> Union[number, str]:
        """
        Returns: Mean of the Chi distribution.
        """
        return np.sqrt(2)*ss.gamma((self.df+1)/2)/ss.gamma(self.df/2)

    def median(self) -> Union[number, str]:
        """
        Returns: Median of the Chi distribution.
        """
        return np.power(self.df*(1-(2/(1*self.df))), 3/2)

    def mode(self) -> Union[number, str]:
        """
        Returns: Mode of the Chi distribution.
        """
        if self.df >= 1:
            return np.sqrt(self.df-1)
        return "undefined"

    def var(self) -> Union[number, str]:
        """
        Returns: Variance of the Chi distribution.
        """
        return pow(self.df-self.mean(), 2)

    def std(self) -> Union[number, str]:
        """
        Returns: Standard deviation of the Chi distribution.
        """
        return self.df-self.mean()

    def skewness(self) -> Union[number, str]:
        """
        Returns: Skewness of the Chi distribution.
        """
        std = np.sqrt(self.var())
        return (self.mean()-2*self.mean()*std**2)/std**3

    def kurtosis(self) -> Union[number, str]:
        """
        Returns: Kurtosis of the Chi distribution.
        """
        sk = self.skewness()
        var = self.var()
        mean = self.mean()
        return 2*(1-mean*np.sqrt(var)*sk-var)/var

    def entropy(self) -> Union[number, str]:
        """
        Returns: differential entropy of Chi distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        df = self.df
        return np.log(ss.gamma(df/2)/sqrt(2))-(df-1)/2*ss.digamma(df/2)+df/2

    def summary(self) -> None:
        """
        Returns: Summary statistic regarding the Chi distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# check plotting function


class Explonential(Base):
    """
    This class contans methods for evaluating Exponential Distirbution.

    Args:

        - lambda_(float | x>0): rate parameter.
        - x(float | x>0): random variable.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.

    References:
    - Weisstein, Eric W. "Exponential Distribution." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/ExponentialDistribution.html
    - Wikipedia contributors. (2020, December 17). Exponential distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 04:38, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Exponential_distribution&oldid=994779060
    """

    def __init__(self, lambda_, x):
        if lambda_ < 0:
            raise Exception(
                'lambda parameter should be greater than 0. Entered value for lambda_:{}'.format(lambda_))
        if x < 0:
            raise Exception(
                'random variable should be greater than 0. Entered value for x:{}'.format(x))
        self.lambda_ = lambda_
        self.x = x

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either probability density evaluation for some point or plot of exponential-distribution.
        """
        lambda_ = self.lambda_
        x = self.x

        def _generator(lambda_, x):
            if x >= 0:
                return lambda_ * np.exp(-(lambda_ * x))
            return 0

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(lambda_, x_i) for x_i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(lambda_, x)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either cumulative distribution evaluation for some point or plot of  exponential distribution.
        """
        lambda_ = self.lambda_
        x = self.x

        def _generator(x, lambda_):
            if x > 0:
                return 1 - np.exp(-lambda_ * x)
            return 0

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(x_i, lambda_) for x_i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(x, lambda_)

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[number]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Exponential distribution evaluated at some random variable.
        """
        lambda_ = self.lambda_
        x = self.x
        if x_lower < 0:
            raise Exception(
                'x_lower cannot be lower than 0. Entered value: {}'.format(x_lower))
        if x_upper is None:
            x_upper = x

        def _cdf_def(x, lambda_):
            if x > 0:
                return 1 - np.exp(-lambda_ * x)
            return 0
        return _cdf_def(x_upper, lambda_) - _cdf_def(x_lower, lambda_)

    def mean(self) -> Union[number, str]:
        """
        Returns: Mean of the Exponential distribution
        """
        return 1 / self.lambda_

    def median(self) -> Union[number, str]:
        """
        Returns: Median of the Exponential distribution
        """
        return np.log(2) / self.lambda_

    def mode(self) -> Optional[number]:
        """
        Returns: Mode of the Exponential distribution
        """
        return 0

    def var(self) -> Union[number, str]:
        """
        Returns: Variance of the Exponential distribution
        """
        return 1 / pow(self.lambda_, 2)

    def std(self) -> Union[number, str]:
        """
        Returns: Standard deviation of the Exponential distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Exponential distribution
        """
        return 2

    def kurtosis(self):
        """
        Returns: Kurtosis of the Exponential distribution
        """
        return 6

    def entorpy(self):
        """
        Returns: differential entropy of the Exponential distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1-np.log(self.lambda_)

    def summary(self) -> None:
        """
        Returns: summary statistic regarding the Exponential distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# check. add pvalue method.


class Gamma(Base):
    """
    This class contains methods concerning a variant of Gamma distribution.

    Args:

        a(float | [0, infty)): shape
        b(float | [0, infty)): scale
        x(float | [0, infty)): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.

    References:
    - Matlab(2020). Gamma Distribution.
    Retrieved from: https://www.mathworks.com/help/stats/gamma-distribution.html?searchHighlight=gamma%20distribution&s_tid=srchtitle
    """

    def __init__(self, a, b, x):
        if a < 0:
            raise Exception(
                'shape should be greater than 0. Entered value for a:{}'.format(a))
        if b < 0:
            raise Exception(
                'scale should be greater than 0. Entered value for b:{}'.format(b))
        if x < 0:
            raise Exception(
                'random variable should be greater than 0. Entered value for x:{}'.format(x))
        self.a = a
        self.b = b
        self.x = x

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either probability density evaluation for some point or plot of Gamma-distribution.
        """
        def _generator(a, b, x): return (1 / (b**a * ss.gamma(a))) * np.power(
            x, a - 1) * np.exp(-x / b)
        if plot:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([_generator(self.a, self.b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.x)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either cumulative distribution evaluation for some point or plot of Gamma-distribution.
        """
        # there is no apparent explanation for reversing gammainc's parameter, but it works quite perfectly in my prototype
        def _generator(a, b, x): return 1 - ss.gammainc(a, x / b)

        if plot:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([_generator(self.a, self.b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.x)

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[number]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Gamma distribution evaluated at some random variable.
        """
        if x_lower < 0:
            raise Exception(
                'x_lower cannot be lower than 0. Entered value: {}'.format(x_lower))
        if x_upper is None:
            x_upper = self.x

        def _cdf_def(a, b, x): return 1 - ss.gammainc(a, x / b)

        return _cdf_def(self.a, self.b, x_upper, self.lambda_) - _cdf_def(self.a, self.b, x_lower, self.lambda_)

    def mean(self) -> Union[number, str]:
        """
        Returns: Mean of the Gamma distribution
        """
        return self.a * self.b

    def median(self) -> Union[number, str]:
        """
        Returns: Median of the Gamma distribution.
        """
        return "No simple closed form."

    def mode(self) -> Optional[number]:
        """
        Returns: Mode of the Gamma distribution
        """
        return (self.a - 1) * self.b

    def var(self) -> Union[number, str]:
        """
        Returns: Variance of the Gamma distribution
        """
        return self.a * pow(self.b, 2)

    def std(self) -> Union[number, str]:
        """
        Returns: Standard deviation of the Gamma distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Gamma distribution
        """
        return 2 / np.sqrt(self.a)

    def kurtosis(self):
        """
        Returns: Kurtosis of the Gamma distribution
        """
        return 6 / self.a

    def entropy(self) -> Union[number, str]:
        """
        Returns: differential entropy of the Gamma distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        k = self.a
        theta = self.b
        return k + np.log(theta)+np.log(ss.gamma(k))-(1-k)*ss.digamma(k)

    def summary(self) -> None:
        """
        Returns: summary statistic regarding the Gamma distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# semi-infinite


class Pareto(Base):
    """
    This class contains methods concerning the Pareto Distribution Type 1.

    Args:

        scale(float | x>0): scale parameter.
        shape(float | x>0): shape parameter.
        x(float | [shape, infty]): random variable.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.

    References:
    - Barry C. Arnold (1983). Pareto Distributions. International Co-operative Publishing House. ISBN 978-0-89974-012-6.
    - Wikipedia contributors. (2020, December 1). Pareto distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 05:00, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Pareto_distribution&oldid=991727349
    """

    def __init__(self, shape, scale, x=1):
        if scale < 0:
            raise Exception(
                'scale should be greater than 0. Entered value for scale:{}'.format(scale))
        if shape < 0:
            raise Exception(
                'shape should be greater than 0. Entered value for shape:{}'.format(shape))
        if x > shape:
            raise Exception(
                'random variable x should be greater than or equal to shape. Entered value for x:{}'.format(x))

        self.shape = shape
        self.scale = scale
        self.x = x

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either probability density evaluation for some point or plot of Pareto distribution.
        """
        x_m = self.scale
        alpha = self.shape

        def _generator(x, x_m, alpha):
            if x >= x_m:
                return (alpha * pow(x_m, alpha)) / np.power(x, alpha + 1)
            return 0

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, x_m, alpha) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.x, x_m, alpha)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either cumulative distribution evaluation for some point or plot of Pareto distribution.
        """
        x_m = self.scale
        alpha = self.shape

        def _generator(x, x_m, alpha):
            if x >= x_m:
                return 1 - np.power(x_m / x, alpha)
            return 0

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, x_m, alpha) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.x, x_m, alpha)

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[number]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Pareto distribution evaluated at some random variable.
        """
        if x_lower < 0:
            x_lower = 0
        if x_upper is None:
            x_upper = self.x

        def _cdf_def(x, x_m, alpha):
            if x >= x_m:
                return 1 - np.power(x_m / x, alpha)
            return 0
        return _cdf_def(x_upper, self.scale, self.alpha)+_cdf_def(x_lower, self.scale, self.alpha)

    def mean(self) -> Union[number, str]:
        """
        Returns: Mean of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale

        if a <= 1:
            return np.inf
        return (a * x_m) / (a - 1)

    def median(self) -> Union[number, str]:
        """
        Returns: Median of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale
        return x_m * pow(2, 1 / a)

    def mode(self) -> Optional[number]:
        """
        Returns: Mode of the Pareto distribution.
        """
        return self.scale

    def var(self) -> Union[number, str]:
        """
        Returns: Variance of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale
        if a <= 2:
            return np.inf
        return (pow(x_m, 2) * a) / (pow(a - 1, 2) * (a - 2))

    def std(self) -> Union[number, str]:
        """
        Returns: Variance of the Pareto distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale
        if a > 3:
            scale = (2 * (1 + a)) / (a - 3)
            return scale * sqrt((a - 2) / a)
        return "undefined"

    def kurtosis(self):
        """
        Returns: Kurtosis of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale
        if a > 4:
            return (6 * (a**3 + a**2 - 6 * a - 2)) / (a * (a - 3) * (a - 4))
        return "undefined"

    def entropy(self) -> Union[number, str]:
        """
        Returns: differential entropy of the Pareto distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        a = self.shape
        x_m = self.scale
        return np.log(x_m/a)+1+(1/a)

    def summary(self) -> None:
        """
        Returns: summary statistic regarding the Pareto distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


# class Weibull(Base):
#     """
#     This class contains methods concerning Weibull Distirbution. Also known as Fréchet distribution.
#     Args:

#         shape(float | [0, infty)): mean parameter
#         scale(float | [0, infty)): standard deviation
#         randvar(float | [0, infty)): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

#     Methods:

#         - pdf for probability density function.
#         - cdf for cumulative distribution function.
#         - pvalue for p-values.
#         - mean for evaluating the mean of the distribution.
#         - median for evaluating the median of the distribution.
#         - mode for evaluating the mode of the distribution.
#         - var for evaluating the variance of the distribution.
#         - std for evaluating the standard deviation of the distribution.
#         - skewness for evaluating the skewness of the distribution.
#         - kurtosis for evaluating the kurtosis of the distribution.
#         - entropy for differential entropy of the distribution.
#         - summary for printing the summary statistics of the distribution.

#     Reference:
#     - Wikipedia contributors. (2020, December 13). Weibull distribution. In Wikipedia, The Free Encyclopedia.
#     Retrieved 11:32, December 28, 2020, from https://en.wikipedia.org/w/index.php?title=Weibull_distribution&oldid=993879185
#     """

#     def __init__(self, shape, scale, randvar=0.5):
#         if shape < 0 or scale < 0 or randvar < 0:
#             raise Exception('all parameters should be a positive number. Entered values: shape: {0}, scale{1}, randvar{2}'.format(
#                 shape, scale, randvar))
#         self.scale = scale
#         self.shape = shape
#         self.randvar = randvar

#     def pdf(self,
#             plot=False,
#             interval=1,
#             threshold=1000,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         """
#         Args:

#             interval(int): defaults to none. Only necessary for defining plot.
#             threshold(int): defaults to 1000. Defines the sample points in plot.
#             plot(bool): if true, returns plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
#             xlabel(string): sets label in x axis. Only relevant when plot is true.
#             ylabel(string): sets label in y axis. Only relevant when plot is true.


#         Returns:
#             either probability density evaluation for some point or plot of Weibull distribution.
#         """
#         def _generator(_lamnda, k, x):
#             if x < 0:
#                 return 0
#             if x >= 0:
#                 return (k/lambda_)*(x/lambda_)**(k-1)*np.exp(-(x/lambda_)**k)
#         if plot:
#             x = np.linspace(-interval, interval, int(threshold))
#             y = np.array([_generator(self.scale, self.shape, i) for i in x])
#             return super().plot(x, y, xlim, ylim, xlabel, ylabel)
#         return _generator(self.scale, self.shape, self.randvar)

#     def cdf(self,
#             plot=False,
#             interval=1,
#             threshold=1000,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         """
#         Args:

#             interval(int): defaults to none. Only necessary for defining plot.
#             threshold(int): defaults to 1000. Defines the sample points in plot.
#             plot(bool): if true, returns plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
#             xlabel(string): sets label in x axis. Only relevant when plot is true.
#             ylabel(string): sets label in y axis. Only relevant when plot is true.


#         Returns:
#             either cumulative distribution evaluation for some point or plot of Weibull distribution.
#         """
#         def _generator(_lamnda, k, x):
#             if x < 0:
#                 return 0
#             if x >= 0:
#                 return 1-np.exp(-pow(x/lambda_, k))

#         if plot:
#             x = np.linspace(-interval, interval, int(threshold))
#             y = np.array([_generator(self.scale, self.shape, i) for i in x])
#             return super().plot(x, y, xlim, ylim, xlabel, ylabel)
#         return _generator(self.scale, self.shape, self.randvar)

#     def pvalue(self, x_lower=0, x_upper=None) -> Optional[number]:
#         """
#         Args:

#             x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
#             x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

#             Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
#             Otherwise, the default random variable is x.

#         Returns:
#             p-value of the Weilbull distribution evaluated at some random variable.
#         """
#         if x_lower < 0:
#             raise Exception(
#                 'x_lower should be a positive number. X_lower:{}'.format(x_lower))
#         if x_upper == None:
#             x_upper = self.randvar
#         if x_lower > x_upper:
#             raise Exception(
#                 'lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))

#         def _cdf_def(_lamnda, k, x):
#             if x < 0:
#                 return 0
#             if x >= 0:
#                 return 1-np.exp(-pow(x/lambda_, k))

#         return _cdf_def(self.location, self.shape, x_upper)-_cdf_def(self.location, self.shape, x_lower)

#     def mean(self) -> Union[number, str]:
#         """
#         Returns: Mean of the Weibull distribution.
#         """
#         return self.scale*ss.gamma(1+(1/self.shape)

#     def median(self) -> Union[number, str]:
#         """
#         Returns: Median of the Weibull distribution.
#         """
#         return self.scale*np.power(np.log(2), 1/self.shape)

#     def mode(self) -> Optional[number]:
#         """
#         Returns: Mode of the Weibull distribution.
#         """
#         k=self.shape
#         if k > 1:
#             return self.scale*np.power((k-1)/k, 1/k)
#         return 0

#     def var(self) -> Union[number, str]:
#         """
#         Returns: Variance of the Weibull distribution.
#         """
#         lambda_=self.scale
#         k=self.shape
#         return lambda_**2*(((ss.gamma(1+2/k) - ss.gamma(1+1/k)))**2)

#     def std(self) -> Union[number, str]:
#         """
#         Returns: Standard deviation of the Weilbull distribution
#         """
#         return sqrt(self.var())

#     def entropy(self) -> Union[number, str]:
#         """
#         Returns: differential entropy of the Weilbull distribution.

#         Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
#         link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
#         """
#         lambda_=self.shape
#         k=self.scale
#         return (k+1)*np.euler_gamma/k+np.log(lambda_/k)+1

#     def summary(self) -> None:
#         """
#         Returns: summary statistics of the Weilbull distribution.
#         """
#         mean=self.mean()
#         median=self.median()
#         mode=self.mode()
#         var=self.var()
#         std=self.std()
#         skewness=self.skewness()
#         kurtosis=self.kurtosis()
#         cstr=" summary statistics "
#         print(cstr.center(40, "="))
#         return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

if __name__ == '__main__':
    ch = Chisq(6, 14.4)
    ch.cdf(plot=True, ylim=3, xlim=4)
