import numpy as np
import scipy
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
        #  func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.
        
        Ax = self.matvec_Ax(x) # A@x
        z = -self.b * Ax #z_i = -b_i * a_i^T x
        loss = np.mean(np.logaddexp(0, z)) #1/m * sum(log(1 + exp(z_i)))
        reg = 0.5 * self.regcoef * np.dot(x, x) # regcoef/2 * ||x||^2
        
        return loss + reg 

    def grad(self, x):
        Ax = self.matvec_Ax(x) # A@x
        z = -self.b * Ax #z_i = -b_i * a_i^T x
        s = expit(z) #s_i = 1/(1 + exp(-z_i))
        v = -self.b *s #v_i = -b_i * s_i
        grad = self.matvec_ATx(v) / len(self.b) + self.regcoef * x # 1/m * A^T @ v + regcoef * x
        
        return grad

    def hess(self, x):
        Ax = self.matvec_Ax(x) # A@x
        z = -self.b * Ax #z_i = -b_i * a_i^T x
        s = expit(z) * (1 - expit(z))
        H = self.matmat_ATsA(s) / len(self.b) # 1/m * A^T * Diag(s * (1 - s)) * A
        
        return H + self.regcoef * np.eye(len(x)) # + regcoef * I


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """

    if scipy.sparse.issparse(A):
        # Для разреженных матриц
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)
        
        def matmat_ATsA(s):
            # A^T @ diag(s) @ A для разреженной матрицы
            return A.T.dot(scipy.sparse.diags(s).dot(A))
    else:
        # Для плотных матриц
        matvec_Ax = lambda x: A @ x
        matvec_ATx = lambda x: A.T @ x
        
        def matmat_ATsA(s):
           
            return A.T @ np.diag(s) @ A
        
    # def matvec_Ax(x):
    #     return A.dot(x)

    # def matvec_ATx(x):
    #     return A.T.dot(x)

    # def matmat_ATsA(s):
    #     # Вычисляет A^T * diag(s) * A
    #     s =  np.asarray(s).reshape(-1, 1)  # Преобразуем s в столбец вектор
    #     if scipy.sparse.issparse(A):
    #         As = A.multiply(s)  # Элементное умножение каждой строки A на соответствующий s_i
    #     else:
    #         As = A * s  # Элементное умножение каждой строки A на соответствующий s_i
    #     return A.T.dot(As)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)



def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float) # преобразуем вход x в numpy-вектор
    grad = np.zeros_like(x)  # vector to store resulting gradient
    f0 = func(x)  # м значение функции в точке x
    
    for i in range(len(x)):
        x_shifted = x.copy()
        x_shifted[i]+= eps
        f1 = func(x_shifted)
        grad[i] = (f1 - f0) / eps

    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    H = np.zeros((n,n), dtype=float)
    f0 = func(x)
    for i in range(n):
        ei = np.zeros_like(x)
        ei[i] = eps
        f_i = func(x + ei)
        for j in range(i, n):
            ej = np.zeros_like(x)
            ej[j] = eps
            if i == j:
                f_ij = func(x + 2*ei)
                H[i,j] = (f_ij - 2*f_i + f0) / (eps**2)
            else:
                    f_ij = func(x + ei + ej)
                    f_j = func(x + ej)
                    H[i, j] = (f_ij - f_i - f_j + f0) / (eps ** 2)
                    H[j, i] = H[i, j] 
    return H
