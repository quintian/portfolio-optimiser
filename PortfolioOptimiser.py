# This file takes historical data from web for your portfolio and returns
# the exected returns and volatility with different weights of each asset.
# The most important, it computes the optimal weights for a given return 
# and the highested Return Risk Ratio. 

'''CITATION: 
I learned how to use Pandas from pandas.pydata.org, 
Numpy from numpy.org, 
matplotlib from https://matplotlib.org 
SciPy from (https://www.scipy.org)
Pillow from ( https://pypi.org/project/Pillow/) 
 '''

from pandas_datareader import data as web
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

# riskFreeReturn is a global variable:
def riskFreeReturn():
    downloadTB=pd.read_csv("TB4WK.csv")# annual basis points return
    dataTB=downloadTB[0:len(downloadTB)-1]
    returnArrayTB=np.array(dataTB['TB4WK'])
    # riskFreeReturn=np.mean(dataTB)/100
    riskFreeReturn=np.mean(returnArrayTB)/100 # basis points convert to percentage
    
    return riskFreeReturn

    # Another way to compute risk free return with continous compounding:
    # dataTB['Monthly Return TB']=dataTB['TB4WK']/12 #monthly return
    # TB_returns_m=dataTB['Monthly Return TB']
    # TB_returns_a=np.exp(np.sum(np.log(TB_returns_m/100+1))/3)-1    
    # return TB_returns_a

riskFreeReturn=riskFreeReturn()

'''
CITATION: reading data from Yahoo.com and Numpy systax for volatility inspired by
https://medium.com/python-data/assessing-the-riskiness-of-a-portfolio-with-python-6444c727c474
'''

class Portfolio(object):
    def __init__(self, assets):
        self.assets=assets       
        

    def getDownload(self, assets):    
        df = pd.DataFrame()  

        for stock in assets:
            df[stock] = web.DataReader(stock, data_source='yahoo',
                                    start='2017-1-1' , end='2020-1-1')['Adj Close']
        d_returns=df.pct_change()                            
                                  
        return d_returns                               

    def getReturns(self, assets):
        d_returns=self.getDownload(assets)    

        lnDailyReturn=np.log((d_returns[1:]+1))
        sumLnReturn=np.sum(lnDailyReturn)
        annualReturn=np.exp(sumLnReturn/3)-1

        m_returns=np.exp(np.sum(np.log((d_returns[1:]+1))/36))-1
        a_returns=np.exp(np.sum(np.log((d_returns[1:]+1))/3))-1
    
        print('annual return\n', a_returns)

        return a_returns
    

    def getCovMatrix(self, assets): #get annual covariance matrix
        
        d_returns=self.getDownload(assets) #daily return
       
        cov_matrix_d = d_returns.cov()
        
        cov_matrix_a = cov_matrix_d * 251
        
        # dfReturns=np.array(d_returns[1:])
        return d_returns, cov_matrix_a


    def getExpectedReturnAnnual(self, assets):
        d_returns, cov_matrix_a=self.getCovMatrix(self.assets)
        expectedReturnMatrix = np.exp(np.sum(np.log((d_returns[1:]+1))/3))-1
        # exp(sum of three years' return/3 )-1
        return expectedReturnMatrix

    def getPortVolatilityReturn(self, weights): 
        # self.weights=weights
        d_returns, cov_matrix_a=self.getCovMatrix(self.assets)
                
        weights=np.array(weights)       
        
        # calculate the variance and risk of the portfolo
        port_variance = np.dot(weights.T, np.dot(cov_matrix_a, weights))       
    
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_a, weights)))    
    
        expectedReturnMatrix = np.exp(np.sum(np.log((d_returns[1:]+1))/3))-1    

        port_expected_return = np.dot(expectedReturnMatrix, weights.T)    

        print(f'Expected Return of Portfolio is {port_expected_return},\
            Portfolio Volatility is {port_volatility}')
            
        return port_expected_return, port_volatility

    
    def equalWeightsRiskReturn(self):
        n=len(self.assets)
        print('line 113', self.assets)
        weight=1/n
        weights=[]
        for i in range(n):
            weights.append(weight)
        
        print('line 119', weights)
        ER, volatility=self.getPortVolatilityReturn(weights)
        return ER, volatility 

    # assets A, B have weights of x, 1-x
    # Variance of portfolio A&B: x**2*VarV+(1-x)**2VarB+2*x*(x-x)CovAB
    # 1st derivative of Variance: 2(VarA+VarB-2CovAB)x^2+2(CovAB-VarB)
    # while slope=0, variance/standard deviation of this portfolio reaches minimum
    # solve it: x=(VarB-CovAB)/(VarA+VarB-2CovAB)

    """ CITATION:The psudo code of Newton's Method for sloving n dimensions 
     function and the mathimatical solutions
     of optimal weights are inspired by 8th chapter of textbook "A Primer for 
     the Mathematics of Financial Engineering" by Dan Stefanica """
     
    #     # f(weights)=variance/volatility of portfolio
    #           =weighted sum of variance of each assets +
    #            covariance between each pair of assets
    #     Constraints: 1. the sum of weights is 1; 
    #                  2. with a given expected return
    
    #     take the second derivative of f(weights) to solve for min and
    #     and get Hessian matrix:
    #     array1=getHesianArray()
    #     array2=ny.array(w1,w2,w3,w4,w5, lamda1, lamda2)
    #     array3=np.array(0,0,0,0,0,1,expected_return_p)

    #     # solve_linear_system(xOld) is:
    #     # np.multiply(array1,array2)=array3
    #     solve array2=array3/array
# Newton's method for n-dimension function:
    #     x0=the initial guess of weights for mininal portfolio volatility
    #     h=step
    #     tol_Fx=tolerance for f(x)
    #     tol_x=tolerance for x
    #     xNew=(x0+h)
    #     # xOld=f(x0)

    #     while abs(f(xNew))>tol_Fx or abs(xNew-xOld)>tol_x:
    #         xOld=xNew
    #         xNew=xOld-ssolve_linear_ystem(xOld)
    #     return xNew

    def getHessianArray(self):
        n=len(self.assets)
        d_returns, cov_matrix_a=self.getCovMatrix(self.assets) 
          
        M=np.multiply(cov_matrix_a,2)
        A=np.array(M)
        # print(type(A))
        # print('A:', A)

        row1ToAdd=np.array([1]*n)
        # print('row1', row1ToAdd)        
        assets=self.assets
        expectedReturns=self.getExpectedReturnAnnual(assets)
        row2ToAdd=np.array(expectedReturns)      

        # print('row2:', row2ToAdd) 
        result=np.vstack((A, row1ToAdd, row2ToAdd))
        # print('result:', result)    
        
        col1ToAdd=np.array(([1]*n+[0,0]))
        col1ToAdd.shape=(col1ToAdd.size//-1,1 )
        # print('col1:', col1ToAdd)
        
        col2ToAdd=np.array(np.append(row2ToAdd, [0,0]))
        col2ToAdd.shape=(col2ToAdd.size//-1,1 )
        # print('col2:', col2ToAdd)    
        
        result=np.hstack((result, col1ToAdd, col2ToAdd)) 
        # print(np.shape(result))
        return result

    # array1*array2=array3 is the final solve_linear_system
    # Since we got the Hessian Array as array1, array 3 is knows as well,
    # we can solve for array 2, which are [weights]+[lamda 1, lamda 2]

    def minVolatilityWeights(self, expected_return_p):
        n=len(self.assets)
        array1=self.getHessianArray()
        # array2=np.array(w1,w2,w3,w4,w5, lamda1, lamda2)
        array3=np.array([0]*n+[1,expected_return_p])
        array2=np.dot(np.linalg.inv(array1), array3)
        # print('array2:  ', array2)
        return array2[0:n]
    # print(minVolatilityWeights(0))

    def riskReturnChart(self):
        result=[]
        volatility=[]
        ER=[-0.01, 0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.7]
        # ER is the portfolio expected return.
        for ER in (-0.01, 0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.7):
            
            weights=self.minVolatilityWeights(ER)
            port_expected_return, port_volatility=\
                self.getPortVolatilityReturn(weights)
            result.append((ER, port_volatility))
            volatility.append(port_volatility)         
      
        return volatility

    def riskReturnRatio(self, ER):    
        weights=self.minVolatilityWeights(ER)
        # print('line 194', weights)
        port_expected_return, port_volatility=\
            self.getPortVolatilityReturn(weights)
        riskReturnRatio=(port_expected_return-riskFreeReturn)/port_volatility
        print("Return Risk Ratio: ", riskReturnRatio)
        return riskReturnRatio

    #  With 3 helper functions-riskReturnRatio(ER) and getPortRiskReturn(weights)
    #  and minVolatilityWeights(ER), the below function-optimalSolver()
    #  does the recursion with preset steps of h and i, and initial guess of 
    #  ER, to reach the optimal solution of Ratio and Weights
    #  with prescision of 0.00001. 
    def optimalSolver(self):   
        ER=0.40 # initial guess  
        h=0.02  # 1st step of recursion
        i=0.01  # 2nd step of recursion       

        if self.riskReturnRatio(ER+h)>self.riskReturnRatio(ER):
            ER+=h
            oldRatio_h=self.riskReturnRatio(ER)

            newRatio_h=self.riskReturnRatio(ER+h)

            while newRatio_h-oldRatio_h>0.00001:
                ER+=h
                oldRatio_h=newRatio_h
                newRatio_h=self.riskReturnRatio(ER+h)            

            print('line214: ', ER, newRatio_h)
            oldRatio_i=self.riskReturnRatio(ER)
            newRatio_i=self.riskReturnRatio(ER+i)
            while newRatio_i-oldRatio_i>0.00001:
                ER+=i
                oldRatio_i=newRatio_i
                newRatio_i=self.riskReturnRatio(ER+i)
           
            optimalER=ER
            optimalRatio=oldRatio_i
        else:
            ER-=h
            oldRatio_h=self.riskReturnRatio(ER)
            newRatio_h=self.riskReturnRatio(ER-h)
            while newRatio_h-oldRatio_h>0.0001:
                ER-=h
                oldRatio_h=newRatio_h
                newRatio_h=self.riskReturnRatio(ER-h)
            
            oldRatio_i=self.riskReturnRatio(ER)
            newRatio_i=self.riskReturnRatio(ER-i)
            while newRatio_i-oldRatio_i>0.00001:
                ER-=i
                oldRatio_i=newRatio_i
                newRatio_i=self.riskReturnRatio(ER-i)
             
            optimalER=ER
            optimalRatio=oldRatio_i
        # another way of recusion:
        '''
        if self.riskReturnRatio(ER+h)>self.riskReturnRatio(ER):
            ER+=h       

            while self.riskReturnRatio(ER+h)>self.riskReturnRatio(ER):
                # print('ER:', ER+h, 'Risk Return:', riskReturnRatio(ER+h))
                ER+=h
            print(ER)              
        
            # oldRatio_i=riskReturnRatio(ER)
            # newRatio_i=riskReturnRatio(ER+i)
            while self.riskReturnRatio(ER+i)>self.riskReturnRatio(ER):
                # print('ER:', ER+i, 'Risk Return:', riskReturnRatio(ER+i))
                ER+=i
            print(ER)
        else:
            ER-=h 
            while self.riskReturnRatio(ER-h)>self.riskReturnRatio(ER):
                ER-=h
            print(ER)
            while self.riskReturnRatio(ER-i)>self.riskReturnRatio(ER):
                ER-=h
            print(ER)
 
        optimalER=ER

        optimalRatio=self.riskReturnRatio(optimalER)'''

        optimalWeights=self.minVolatilityWeights(optimalER)        
       
        return optimalER, optimalRatio, optimalWeights
    
    """
    CITATION: writeFile() is from courseweb
     https://www.cs.cmu.edu/~112/notes/notes-strings.html
     """
    def writeFile(self, path, contents):
        with open(path, "w+") as f:
            f.write(contents)

    def plotGraph(self):
        volatility=self.riskReturnChart()
        ER=[-0.01, 0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.7]
        ER=np.array(ER)
        volatility=np.array(volatility)

        # plt.plot(volatility,ER)
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.title(f'''Portfolio Risk Return Optimization
        for portfolio of {self.assets}''')
        '''CITATION: smooth lines inspired by one post from
        https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
        '''
        # 300 represents number of points to make between ER.min and ER.max
        ynew = np.linspace(ER.min(), ER.max(), 300) 

        spl = make_interp_spline(ER, volatility, k=3)  # type: BSpline
        volatility_smooth = spl(ynew)

        plt.plot(volatility_smooth, ynew)

        optimalER, optimalRatio, optimalWeights=self.optimalSolver()
        optimalVolatility=(optimalER-riskFreeReturn)/optimalRatio
        plt.plot([0, (0.7-riskFreeReturn)/optimalRatio],[riskFreeReturn, 0.7],\
             'k-', linestyle='-', linewidth=2 )
        plt.plot(optimalVolatility, optimalER, 'ro', markersize=10)

        # plt.show()
        
# # evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)

# # red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()

# Testcases:
# assets =  ['NVDA', 'JPM', 'WMT', 'GE', 'SPY']
# # # assets = ['baba', 'spy', 'goog', 'amzn']
# port=Portfolio(assets)
# # # # # print("Equal Weights Port Return and Volatility: \n", \
# # # # #     port.equalWeightsRiskReturn())
# # # # # print('Hessian Array:\n', port.getHessianArray())
# # # # # # print("Volatility and Return Chart: \n", port.riskReturnChart())
# # # # print('Optimal ER, Ratio, weights: \n', port.optimalSolver())

# port.plotGraph()




            
    