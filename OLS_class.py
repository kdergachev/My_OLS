# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy as sp


class MyOLS:
    
    """class to run linear regression, store some of its parameters and apply 
    tests
    uses numpy as np, math, scipy as sp
    
    
    Stores:
        __init__ -> .Y, .X, .Xt
        OLS_est -> .betas, .Yhat, .resids, .RSS, .R2, .ses (variance matrix), 
        .method
        GLS_est -> .wX, .betas, .Yhat, .resids, .chi, .method
        misc -> .status
    
    
    methods defined yet are:
        OLS_est, White_HCSE, GLS_est, predict, t_test, F_test, Chow_test
    """
    
    # This was intended to be a simple ols + Chow test class initially, but 
    # over time it grew into this monstrosity
    # TODO: add summary
    # !!!!! GLS seemed to give wrong results !!!!!!! <- check/fix
    
    def __init__(self, dset, yrow = 0, add_intercept=True):
        
        """dset - numpy array of type 
        [[y_1, y_2, y_3...y_n], 
         [x_11, x_12, x_13...x_1n],
                    ...
         [x_k1, x_k2, x_k3...x_kn]]. 
        position of dependent variable can be specified by yrow=__, 0 by 
        default and vector for constant term is added on the left
        if add_intercept=True. 
        No non numeric (nan, inf...) are supported.
        """
        
        
        # possibly remove Xt or X as one is redundant
        # get x and y matrices preprocessed for obtaining betas
        self.Y = dset[yrow].T
        self.X = np.delete(dset, yrow, 0)
        if add_intercept:
            self.X = np.vstack((np.ones(len(self.X[0])), self.X))
        self.X = self.X.T
        
        # some weird manipulations but in the end X, Y, X^t are acquired
        
        # self.Xt = self.X.T      #########
        self.status = True
        
    def OLS_est(self):
        
        """OLS regresson method. Inputs should've been supplied at __init__. 
        returns None so chaining is not possible all the results are stored 
        in the object"""
        
        Xt = self.X.T
        try:
            mult = np.linalg.inv(Xt.dot(self.X))         #(X^t * X)^-1
        except:
            print("Perfect multicollinearity problem. Returning None")
            self.status = False
            return None
            
        
        # linalg does not throw eror when determinant is 0 ->-> and calculates 
        # the determinant as a huge (~10^15) value
        if (abs(mult) > 10000000000).all():             
            print("Warning, inverse likely doesnt exist. Possibly a problem" 
                  "of perfect multicollinearity")      
            self.status = False                         
                   
                                                    
        #((X^t * X)^-1)*(X^t * Y) (= betas)
        self.betas = mult.dot((Xt.dot(self.Y)))  
        self.Yhat = self.betas.dot(Xt)          #betas*X = Y predicted
        self.resids = self.Yhat - self.Y
        self.RSS = ((self.resids)**2).sum()
        # get R^2
        Ybar = np.mean(self.Y)
        TSS = ((self.Y - Ybar)**2).sum()
        self.R2 = 1 - self.RSS/TSS
        varce = self.RSS/(len(self.Y) - len(Xt))
        # standrd error matrix (one where u_i are considered orthagonal)
        self.ses = mult * varce     
        self.method = "OLS"
        return None 
    
    
    def White_HCSE(self, replace=True):
        
        """Calculate White heteroscedasticity consistent standard errors for 
        regression that was already estimated. 
        replace parameter decides whether the errors are returned or replace 
        standard errors saved in object"""
        
        
        Xt = self.X.T
        res = self.resids ** 2
        res = res * (len(self.Y)/(len(self.Y) - len(Xt)))
        res = res.diagonal()    # diag residual covariance matrix (corrected)
        mult = Xt.dot(self.X)
        mult = np.linalg.inv(mult)   # (XtX)^{-1}
        res = Xt.dot(res).dot(self.X)  # XtSX
        ses = mult.dot(res).dot(mult) # (XtX)^{-1}XtSX(XtX)^{-1}
        if replace:
            self.ses = ses
            return None
        else:
            return ses
    
    
    # !!Not done yet!!
    def GLS_est(self, covs):
        
        """GLS regression method. needs to be supplied with covariance matrix 
        of n*n size. identity for OLS, diagonal for WLS. Additionaly saves 
        .wX which should at least be useful for WLS. Returns None as with OLS 
        method"""
        # missing se for GLS betas. Can't be bothered to add them now
        Xt = self.X.T
        omegat = np.linalg.inv(covs)
        self.wX = Xt.dot(omegat)
        mult1 = self.wX.dot(self.X)
        mult1 = np.linalg.inv(mult1)
        mult2 = self.wX.dot(self.Y)
        self.betas = mult1.dot(mult2)    # (Xt*omega^-1 * x)(Xt*omega^-1 * Y)
        
        self.Yhat = self.X.dot(self.betas)         # betas*X = Y predicted
        self.resids = self.Y - self.Yhat     
        chi = self.resids.T.dot(omegat).dot(self.resids) 
        # no idea why the chi2/df is too high for COVID data
        self.chi = chi/(len(self.Y) - len(Xt))
        self.method = "GLS"
        return None
    
    
    
    
    def predict(self, dta, const=True):
        
        """Input matrix like in __init__ without y : 
        [[x_11, x_21...x_n1],
        [x_12, x_22...x_n2],
                ...
        [x_1k, x_2k,...x_nk]].
        constant vector is added for beta[0] if const==True.
        returns array of predicted ys"""
        
        if const:
            dta = np.vstack((np.ones(len(dta[0])), dta))
        dta = dta.T
        return dta.dot(self.betas)  # easy enough: Xbeta = Y
    
    
    def t_test(self, hyp = None):
        
        """If OLS is run it will return an array of t statistics for each 
        coefficient and degree of freedom for t distribution.
        Can be supplied with hypothesised values for betas, but it has to be 
        an array of length equal to amount of parameters"""
        
        # TODO: make input a bit better
        
        Xt = self.X.T
        if hyp is None:
            hyp = np.zeros(len(self.betas))
        
        bmean = self.betas - hyp              # (beta - \mu)
        denom = 1/np.sqrt(self.ses.diagonal())    # 1/sqrt(\sigma)
        denom = np.einsum('i,i->i', bmean, denom)  # (beta - \mu)/sqrt(\sigma)
        # print(denom, len(self.Y) - len(self.Xt))
        denom = sp.stats.t.cdf(denom, len(self.Y) - len(Xt))
        denom = np.minimum(denom, 1 - denom) * 2    # make the p-value 2 sided
        return denom
    
    
    
    def F_test(self, test_pos = None):
        
        """Inputs are implying test of type 
        (beta at position test_pos[i][0]) == test_pos[i][1] for all i
        if test_pos is not supplied overall regression test is conducted
        retrns [Fstat, df1, df2]"""
        
        
        Xt = self.X.T
        if test_pos is None:    # overall regression test
            Fstat = (self.R2/(len(Xt) - 1))/((1 - self.R2)/(len(self.Y) 
                                                             - len(Xt)))
    
            Fstat = sp.stats.f.cdf(Fstat, len(Xt) - 1, len(self.Y) - len(Xt))
            return Fstat
        else:
            a = np.zeros(len(self.betas))
            pos = [i[0] for i in test_pos]
            a[[int(i) for i in pos]] = [j[1] for j in test_pos]
            Y = self.Y - self.X.dot(a)    # correct Y by hypothsysed betas
            Y = np.vstack((Y, np.delete(Xt, pos, 0)))  # remove betas in hyp
            # Now it's the data for regression
            RSSR = MyOLS(Y, 0, False)
            RSSR.OLS_est()
            RSSR = RSSR.RSS
            #print(self.RSS, RSSR)
            Fstat = ((RSSR - self.RSS)/len(pos))/(self.RSS/(len(self.Y) 
                                                            - len(Xt)))
            # print(Fstat, len(pos), len(self.Y) - len(self.Xt))
            return 1 - sp.stats.f.cdf(Fstat, len(pos), len(self.Y) 
                                      - len(Xt))
        
    
    # In progress. Seems to work
    def Ramsey_RESET(self, order=1):
        
        
        Xt = self.X.T
        # Not ready to do that with GLS
        if (not self.status) | (not self.method == "OLS"): 
            print("The object is not fit for the test")
            return None
        Yhats = self.Yhat**2    # first order of test <=> second order of Yhat
        Yhats = Yhats.reshape(1, len(Yhats))
        # add other orders
        for i in range(1, order):
            Yhats = np.vstack((Yhats, self.Yhat**(i+1)))
        testpos = len(Yhats)
        Xs = np.vstack((Xt, Yhats))
        Xs = np.vstack((self.Y.T, Xs))
        # Xs is a matrix of [Ys, Xs, Yhat^i]
        testpos = range(len(Xs) - testpos - 1, len(Xs) - 1)
        # reun regression on that and F_test it
        temp = MyOLS(Xs, 0, False)
        temp.OLS_est()
        #print(temp.betas)
        #print(temp.RSS)
        testpos = np.vstack((testpos, np.zeros(len(testpos))))
        testpos = testpos.T
        temp = temp.F_test(testpos)
        return temp
    
        
    def Chow_test(self, intervals, percents = True, save_betas=False):
        
        """Chow test for stability. Method of MyOLS so total data is supplied 
        through initialization. Intervals supplied are either in percents of 
        total data or in actual positions in the dataset. 
         Returns F-statistic 
        of the test and degrees of freedom. p-value can be obtained 
        by using a library with CDF of F distribution
        Does not work for program evaluation"""
        
        # TODO change to accept any subsets (input a list of positions to take 
        # from a whole dataset into each subset)
        
        # No idea what happens when the subsamples overlap.
        
        
        # back to initial data format as MyOLS is used to find RSSR and I 
        # don't feel like storing the whole dataset in the class
        dataused = np.column_stack((self.Y, self.X))
        
        # turn whatever intervals supplied into [(lower, upper), (lower, upper)...] 
        # in positions rather than percents
        if percents:
            intervals = [(math.floor(len(self.X) * min(j)/100), 
                          math.ceil(len(self.X) 
                                    * max(j)/100)) for j in intervals]
        else:
            intervals = [(min(j), max(j)) for j in intervals]
        
        
        Xt = self.X.T
        RSSRsum = 0
        Nsum = 0
        betas = []
        # get RSSR on subsample with MyOLS and add to RSSRsum, also get sum 
        # of sizes of subsamples
        for l, u in intervals:
            datatemp = dataused[l:u]
            regr = MyOLS(datatemp.T, 0, False)
            regr.OLS_est()
            print(regr.status)
            #######
            if not regr.status:
                print("Something in one of the subsets went wrong")
                return 
            print("a")
            RSSR = regr.RSS
            if save_betas:
                betas.append(regr.betas)
            RSSRsum += RSSR
            Nsum += u - l
        print("b")
        df1 = (len(intervals) - 1)*len(Xt)            # (m-1*)k
        df2 = Nsum - len(intervals)*len(Xt)           # sum(n) - mk
        Fstat = ((self.RSS - RSSRsum)/df1)/(RSSRsum/df2)
        Fstat = sp.stats.f.cdf(Fstat, df1, df2)
        return (Fstat, betas)






if __name__ == "__main__":
    
    import pandas as pd
    import os
    import time
    import sklearn as sk
    import sklearn.linear_model
    import matplotlib.pyplot as plt
    import sys
    import statsmodels.api as sm
    np.set_printoptions(suppress=True)
    
    
    a = np.array([[  0.72134752,   4.48142012,  2.97201341,   2.2124622,    2.66884837,
    3.67736674,  21.49612347,  11.49274997,   1.95761519,   1.62552559,
    7.88944019,   8.89062881,   6.36190659,  11.49274997,   2.87103296,
   42.99806195,   3.23619524,   7.07155295,  34.62259313,  16.10593737,
    6.46711944,  19.88469824,   7.04174121,  20.6388196,   11.59281253,
   13.57494287,  15.81826159,  11.16561453,  10.95791344,  14.06550441,
   10.6922073,   14.49425105,  13.22097021,  34.05310843,  15.30407903,
   33.69752706,  16.04025962,  13.4938249,   11.91047991,  18.25932524,
   12.5558637,   22.01092001,   9.39229194,  14.1180979,   16.96355626,
    8.84834267,  13.73173675,  17.92072327,  10.99880744,  14.72592043,
    9.26070012,  19.6862433,   20.33215888,  17.06721065,  15.31024368,
   11.42224452,  15.94038657,  17.47711618,  19.36641964],
 [  1.,           1.,           1.,           1.,           1.,
    1.,           1.,           1.,           1.,           1.,
    1.,           1.,           1.,           1.,           1.,
    1.,           1.,           1.,           1.,           1.,
    1.,           1.,           1.,           1.,           1.,
    1.,           1.,           1.,           1.,           1.,
    1.,           1.,           1.,           1.,           1.,
    1.,           1.,           1.,           1.,           1.,
    1.,           1.,           1.,           1.,           1.,
    1.,           1.,           1.,           1.,           1.,
    1.,           1.,           1.,           1.,           1.,
    1.,           1.,           1.,           1.,        ],
 [ 14.,          16.,          17.,          20.,          21.,
   22.,          23.,          26.,          28.,          30.,
   31.,          32.,          33.,          35.,          36.,
   37.,          38.,          39.,          40.,          41.,
   42.,          43.,          44.,          45.,          46.,
   47.,          48.,          49.,          50.,          51.,
   52.,          53.,          54.,          55.,          56.,
   57.,          58.,          59.,          60.,          61.,
   62.,          63.,          64.,          65.,          66.,
   67.,          68.,          69.,          70.,          71.,
   72.,          73.,          74.,          75.,          76.,
   77.,          78.,          79.,          80.,        ],
 [ 50.53151667,  55.65580236,  58.31073033,  66.64665473,  69.54900969,
   72.5132214,   75.53928986,  84.98863572,  91.59748337,  98.45375801,
  101.97468045, 105.55745964, 109.20209558, 116.6769377,  120.50714388,
  124.39920681, 128.35312649, 132.36890291, 136.44653609, 140.58602601,
  144.78737268, 149.0505761,  153.37563626, 157.76255317, 162.21132684,
  166.72195725, 171.2944444,  175.92878831, 180.62498896, 185.38304636,
  190.20296051, 195.08473141, 200.02835906, 205.03384345, 210.10118459,
  215.23038248, 220.42143712, 225.6743485,  230.98911663 ,236.36574152,
  241.80422314, 247.30456152, 252.86675665, 258.49080852 ,264.17671714,
  269.92448251, 275.73410463, 281.60558349, 287.53891911, 293.53411147,
  299.59116058, 305.71006643, 311.89082904, 318.13344839, 324.43792449,
  330.80425734, 337.23244694, 343.72249328, 350.27439637]])
    # __init__ test 
    b = MyOLS(a[:], 0, False)
    print(b.Y)
    print(b.X)
    print(b.status)
    # seems to work as expected
    
    # OLS_est test
    b = MyOLS(a[:], 0, False)
    b.OLS_est()
    c = sm.OLS(a[0], a[1:].T)
    c = c.fit()
    print("residuals")    # if RSS is correct Yhat and resids are correct
    print(b.RSS)
    print(b.ses)
    print(b.betas)
    print(c.summary())
    print(b.R2)
    # works well
    # partial t and F tests
    print(b.t_test())
    print(b.F_test())
    print(b.F_test([(1, 0), (2, 0)]))
    print(b.F_test([(1, 1.1287)]))
    print(b.t_test([0, 1.1287, 0]))
    # seems correct, but a bit off, at 4th, or greater decimal place is different
    
    # TODO: add remaining tests for chosen data. 
    # Note: look into GLS. F-stats seemed to give low p-values all the time.
    # check it for some bad specification to be sure
    # HCSE
    
    # RESET
    
    # Chow



