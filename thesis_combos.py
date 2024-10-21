import numpy as np

from typing import Union, Callable
from scipy.stats import norm
from scipy.special import betainc
from sklearn.metrics import log_loss
from scipy.optimize import minimize


class ComboMethods:

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray):
        
        for arg in [y_true, y_score]:
            name  = f'{arg}'.split('=')[0]
            if not isinstance(arg, np.ndarray):
                raise TypeError(f"Argument '{name}' must be of type np.ndarray, 
                                but got {type(arg).__name__} instead")
        
        if y_true.shape[0] != y_score.shape[0]:
            raise IndexError("Lengths of 'y_true' and 'y_score' do not match. Check the inputs!")
        
        self.y_true = y_true
        self.y_score = y_score

        try:
            self.n = y_score.shape[1]
        except IndexError:
            raise ValueError(
                "'y_score' expected to be 2D array, got 1D without second axis instead"
            )
        
        if self.n == 1:
            raise ValueError("'y_score' expected to be 2D array, got 1D instead")
        
        print(
            '''
            Hey there! You've just created an instance of an amazing ComboMethods class! 
            It allows you to combine different predictive distrubutions to obtain better 
            aggregated forecasts.
            '''
        )
        

    def glp_funcs(self) -> dict[str, Callable]:
        '''
        Description:
            A bunch (actually dict) of functions for combining predictive distributions 
            using generalized linear pool

        Args:
            None

        Returns:
            dict: 4 functions for different specifications of GLP
        '''

        glp_funcs = {
            'simple': lambda y_score, params: np.dot(y_score, params),
            'harmonic': lambda y_score, params: 1 / np.dot(1/y_score, params),
            'logarithmic': lambda y_score, params: np.exp(np.dot(np.log(y_score), params)),
            'N(0,1)': lambda y_score, params: norm.ppf(np.dot(norm.cdf(y_score), params))
        }

        return glp_funcs
    

    def blp_func(self, y_score: np.ndarray, params: np.ndarray) -> np.ndarray:
        '''
        Description:
            Combines predictive distributions using beta transformed linear pool

        Args:
            y_score (np.ndarray): 2D array of predictive distributions
            params (np.ndarray): 1D array of parameteres, first two are those of 
                beta distributions and remaining pooling weights 

        Returns:
            np.ndarray: 1D array of beta transformed linear pool of predictive distributions
        '''

        lp = np.dot(y_score, params[2:])
        blp = betainc(params[0], params[1], lp)

        return blp
    

    def bmc_func(self, y_score: np.ndarray, params: np.ndarray, m: int) -> np.ndarray:
        '''
        Description:
            Combines predictive distributions using beta mixture combination
        
        Args:
            y_score (np.ndarray): 2D array of predictive distributions
            params (np.ndarray): 1D array of parameteres, m groups of (2 + self.n) BLP's 
                parameteres and m mixture weights

        Returns:
            np.ndarray: 1D array of beta mixture combination of predictive distributions
        '''

        blps = []

        for i in range(m):
            lp = np.dot(y_score, params[2+i*(2+self.n):(i+1)*(2+self.n)])
            blp = betainc(params[i*(2+self.n)], params[1+i*(2+self.n)], lp)
            blps.append(blp)

        bmc = np.dot(params[m*(2+self.n):], np.array(blps))

        return bmc


    def glp_fit(self, name: str) -> dict[str, Union[Callable, np.ndarray]]:
        '''
        Description:
            Fits the generalized linear pool

        Args:
            name (str): which linear pool is desired to be implemented - simple, harmonic, 
                logarithmic or N(0,1)

        Returns:
            np.ndarray: 1D array of generalized linear pool of predictive distributions
        '''

        allowed_names = ('simple', 'harmonic', 'logarithmic', 'N(0,1)')

        if name not in allowed_names:
            raise ValueError(f'Input correct pool name! The allowed names: {allowed_names}')
        
        # replace 0s with a very small value in order to avoid division by zero error (and log(0))
        # which may raise in harmonic and 
        eps = 1e-7
        y_score_safe = self.y_score.copy()
        y_score_safe = np.where(y_score_safe == 0, eps, y_score_safe)

        def neg_llh(params, y_true, y_score):
            '''Negative log-likelihood function for the specified generalized linear pool'''
            
            return log_loss(y_true, self.glp_funcs()[name](y_score, params))
        
        glp_initial = np.ones(self.n) / self.n
        glp_cons = [
            {'type': 'eq', 
             'fun': lambda x:  1 - sum(x)}, # weights sum up to one 
            {'type': 'ineq', 
             'fun': lambda x: 1 - self.glp_funcs()[name](y_score_safe, x)} # combined proba <= 1
        ] 
        glp_bounds = [(0, 1)] * self.n

        res_glp = minimize(neg_llh, glp_initial, method='SLSQP', bounds=glp_bounds, 
                           constraints=glp_cons, args=(self.y_true, y_score_safe))
        glp = self.glp_funcs()[name](y_score_safe, res_glp.x)

        return {
            'scores': glp,
            'params': res_glp.x
        }
    
    
    def blp_fit(self) -> dict[str, Union[Callable, np.ndarray]]:
        '''
        Description:
            Fits the beta transformed linear pool

        Args:
            self.y_true (np.ndarray): 1D array of true target values
            self.y_score (np.ndarray): 2D array of self.n predictive distributions

        Returns:
            dict: 1D array of beta-transformed linear pool of predictive distributions and 
                the corresponding estimated parameteres
        '''

        def neg_blp_llh(params, y_true, y_score):
            '''Negative log-likelihood function for the beta-transformed linear pool'''
    
            return log_loss(y_true, self.blp_func(y_score, params))
        
        blp_initial = [1, 1.1] + [1/self.n] * self.n
        blp_bounds = [(0, None)] * 2 + [(0, 1)] * self.n
        blp_cons = [
            {'type': 'eq', 'fun': lambda x:  1 - sum(x[2:])},
            {'type': 'ineq', 'fun': lambda x: 1 - self.blp_func(y_score=self.y_score, params=x)}
        ]

        res_blp = minimize(neg_blp_llh, blp_initial, method='SLSQP', bounds=blp_bounds, 
                           constraints=blp_cons, args=(self.y_true, self.y_score))
        
        blp = self.blp_func(self.y_score, res_blp.x)

        return {
            'scores': blp,
            'params': res_blp.x
        }

    
    def bmc_fit(self, M: int) -> dict[str, Union[Callable, np.ndarray]]:
        '''
        Description:
            Fits the beta mixture combination

        Args:
            self.y_true (np.ndarray): 1D array of true target values
            self.y_score (np.ndarray): 2D array of self.n predictive distributions
            m (int): order of mixture, i.e. number of different BLPs in it

        Returns:
            dict: 1D array of beta mixture combination of predictive distributions and 
                the corresponding estimated parameteres
        '''
        
        allowed_m = (2, 3, 4)

        if M not in allowed_m:
            raise ValueError(f"Input correct 'm' value! The allowed ones are: {allowed_m}")
        
        def neg_bmc_llh(params, y_true, y_score):
            '''Negative log-likelihood function for the beta mixture combination'''

            return log_loss(y_true, self.bmc_func(y_score=y_score, params=params, m=M))
        
        bmc_initial = ([1, 1.1] + [1/self.n] * self.n) * M + [1/M] * M
        bmc_bounds = ([(0, None)] * 2 + [(0, 1)] * self.n) * M + [(0, 1)] * M
        bmc_cons = []

        for i in range(M):
            cons = {'type': 'eq', 'fun': lambda x:  1 - sum(x[2+i*(2+self.n):(i+1)*(2+self.n)])}
            bmc_cons.append(cons)
        else:
            theta_cons = {'type': 'eq', 'fun': lambda x:  1 - sum(x[M*(2+self.n):])}
            total_proba_cons = {
                'type': 'ineq', 
                'fun': lambda x: 1 - self.bmc_func(y_score=self.y_score, params=x, m=M)
            }
            bmc_cons.append(theta_cons)
            bmc_cons.append(total_proba_cons)
        
        res_bmc = minimize(neg_bmc_llh, bmc_initial, method='SLSQP', bounds=bmc_bounds, 
                           constraints=bmc_cons, args=(self.y_true, self.y_score))
        
        bmc = self.bmc_func(y_score=self.y_score, params=res_bmc.x, m=M)

        return {
            'scores': bmc,
            'params': res_bmc.x
        }