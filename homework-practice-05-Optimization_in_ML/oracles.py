import numpy as np
import scipy
import scipy.sparse
from scipy.special import expit

class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
       func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
    matvec_Ax : function
        Computes matrix-vector product Ax, where x is a vector of size n.
    matvec_ATx : function of x
        Computes matrix-vector product A^Tx, where x is a vector of size m.
    matmat_ATsA : function
        Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # f(x) = 1/m * sum(log(1 + exp(-b_i * <a_i, x>))) + regcoef/2 * ||x||^2
        Ax = self.matvec_Ax(x)
        margins = -self.b * Ax
        
        # Use logaddexp for numerical stability
        logistic_loss = np.mean(np.logaddexp(0, margins))
        l2_reg = 0.5 * self.regcoef * np.dot(x, x)
        
        return logistic_loss + l2_reg

    def grad(self, x):
        # grad = -1/m * A^T @ (b * sigmoid(-b * Ax)) + lambda * x
        Ax = self.matvec_Ax(x)
        margins = -self.b * Ax
        sigmoid_vals = expit(margins)
        
        m = self.b.shape[0]
        # gradient of loss part
        grad_loss = - (1.0 / m) * self.matvec_ATx(self.b * sigmoid_vals)
        # gradient of regularization part
        grad_reg = self.regcoef * x
        
        return grad_loss + grad_reg

    def hess(self, x):
        # hess = 1/m * A^T @ diag(s) @ A + regcoef * I
        # where s = sigmoid(-bAx) * (1 - sigmoid(-bAx))
        Ax = self.matvec_Ax(x)
        margins = -self.b * Ax
        p = expit(margins)
        s = p * (1 - p)
        
        m = self.b.shape[0]
        hess_loss = (1.0 / m) * self.matmat_ATsA(s)
        
        # Handle sparse vs dense logic for I
        if scipy.sparse.issparse(hess_loss):
            hess_reg = scipy.sparse.eye(x.shape[0], format='csr') * self.regcoef
            res = hess_loss + hess_reg
            # *** FIX: Convert to dense for np.allclose compatibility in tests ***
            return res.toarray()
        else:
            hess_reg = np.eye(x.shape[0]) * self.regcoef
            return hess_loss + hess_reg


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle with caching for optimal performance.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        
        self._x_current = None
        self._Ax_current = None
        
        self._d_current = None
        self._Ad_current = None
        
        self._x_last_dir = None
        self._Ax_last_dir = None

    def _update_x(self, x):
        if self._x_current is not None and np.array_equal(x, self._x_current):
            return
        if self._x_last_dir is not None and np.array_equal(x, self._x_last_dir):
            self._x_current = self._x_last_dir
            self._Ax_current = self._Ax_last_dir
            return
        self._x_current = np.copy(x)
        self._Ax_current = self.matvec_Ax(x)

    def _update_d(self, d):
        if self._d_current is not None and np.array_equal(d, self._d_current):
            return
        self._d_current = np.copy(d)
        self._Ad_current = self.matvec_Ax(d)

    def func(self, x):
        self._update_x(x)
        margins = -self.b * self._Ax_current
        logistic_loss = np.mean(np.logaddexp(0, margins))
        l2_reg = 0.5 * self.regcoef * np.dot(x, x)
        return logistic_loss + l2_reg

    def grad(self, x):
        self._update_x(x)
        margins = -self.b * self._Ax_current
        sigmoid_vals = expit(margins)
        m = self.b.shape[0]
        grad_loss = - (1.0 / m) * self.matvec_ATx(self.b * sigmoid_vals)
        grad_reg = self.regcoef * x
        return grad_loss + grad_reg

    def hess(self, x):
        self._update_x(x)
        margins = -self.b * self._Ax_current
        p = expit(margins)
        s = p * (1 - p)
        
        m = self.b.shape[0]
        hess_loss = (1.0 / m) * self.matmat_ATsA(s)
        
        if scipy.sparse.issparse(hess_loss):
            hess_reg = scipy.sparse.eye(x.shape[0], format='csr') * self.regcoef
            res = hess_loss + hess_reg
            # *** FIX: Convert to dense here as well ***
            return res.toarray()
        else:
            hess_reg = np.eye(x.shape[0]) * self.regcoef
            return hess_loss + hess_reg

    def func_directional(self, x, d, alpha):
        self._update_x(x)
        self._update_d(d)
        
        Ax_alpha_d = self._Ax_current + alpha * self._Ad_current
        x_alpha_d = x + alpha * d
        
        self._x_last_dir = x_alpha_d
        self._Ax_last_dir = Ax_alpha_d
        
        margins = -self.b * Ax_alpha_d
        logistic_loss = np.mean(np.logaddexp(0, margins))
        l2_reg = 0.5 * self.regcoef * np.dot(x_alpha_d, x_alpha_d)
        return logistic_loss + l2_reg

    def grad_directional(self, x, d, alpha):
        self._update_x(x)
        self._update_d(d)
        
        Ax_alpha_d = self._Ax_current + alpha * self._Ad_current
        x_alpha_d = x + alpha * d
        
        self._x_last_dir = x_alpha_d
        self._Ax_last_dir = Ax_alpha_d
        
        margins = -self.b * Ax_alpha_d
        sigmoid_vals = expit(margins)
        m = self.b.shape[0]
        
        grad_loss_dot_d = - (1.0 / m) * np.dot(self.b * sigmoid_vals, self._Ad_current)
        grad_reg_dot_d = self.regcoef * np.dot(x_alpha_d, d)
        
        return grad_loss_dot_d + grad_reg_dot_d


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
    """
    if scipy.sparse.issparse(A):
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)
        def matmat_ATsA(s):
            return A.T.dot(scipy.sparse.diags(s).dot(A))
    else:
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)
        def matmat_ATsA(s):
            return (A.T * s).dot(A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)

    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Computes numerical gradient using finite differences.
    """
    n = len(x)
    grad = np.zeros(n)
    f_x = func(x)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        f_x_plus = func(x + e_i)
        grad[i] = (f_x_plus - f_x) / eps
        
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Computes numerical Hessian using finite differences.
    """
    n = len(x)
    hess = np.zeros((n, n))
    f_x = func(x)
    
    # Precompute f(x + e_i) to reduce oracle calls
    f_x_plus_i = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        f_x_plus_i[i] = func(x + e_i)
        
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        for j in range(i, n):
            e_j = np.zeros(n)
            e_j[j] = eps
            
            f_x_plus_ij = func(x + e_i + e_j)
            
            val = (f_x_plus_ij - f_x_plus_i[i] - f_x_plus_i[j] + f_x) / (eps ** 2)
            hess[i, j] = val
            hess[j, i] = val
            
    return hess