# This file read asset's data from Yahoo,  does regression analysis 
# on asset's return and markets return, and generate the predicted returns.

'''CITATION: I learned how to use Pandas and Numpy from pandas.pydata.org 
and numpy.org 
Matplotlib from
https://matplotlib.org/
'''
import pandas as pd
import numpy as np
import datetime
import pandas_datareader as dr
import matplotlib.pyplot as plt

'''CITATION: The regression code is from the link:
https://stackoverflow.com/questions/38676323/is-it-possible-to-get-monthly-historical-stock-prices-in-python
'''
def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1)

'''treasury bill monthly rate'''
def monthlyTBReturns():
     downloadTB=pd.read_csv("TB4WK.csv")# annual return
     dataTB=downloadTB[0:len(downloadTB)-1]
     
     ''' use np to calculate monthly return:'''
     dataTB['Monthly Return TB']=dataTB['TB4WK']/1200 #monthly return
    
     return dataTB

# riskFreeReturn is a global variable:
dataTB=monthlyTBReturns() 
TB_returns=dataTB['Monthly Return TB'] 
a_return_TB=np.average(dataTB['TB4WK']/100)

def monthlySPReturns():
     df = pd.DataFrame()     
     df=dr.get_data_yahoo('^GSPC', interval='m',
                                start='2017-1-1' , end='2020-1-1')['Adj Close']

     # calculate monthly Return 
     m_returns=df.pct_change()
     # exclude the first row since percentage change starts from the second row
     return np.array(m_returns[1:])    
    

class Asset(object):
    def __init__(self, ticker):
        self.ticker=ticker

    def getData(self):
        df = pd.DataFrame()          
        df=dr.get_data_yahoo(self.ticker,
                            start='2017-1-1' , end='2020-1-1')['Adj Close']
                
        return df 
        
    def getMonthlyData(self):
        df = pd.DataFrame()          
        df=dr.get_data_yahoo(self.ticker, interval='m',
                            start='2017-1-1' , end='2020-1-1')['Adj Close']
                 
        return df        

    def getDailyReturns(self):
        df=self.getData()
        d_returns=df.pct_change()
        
        return np.array(d_returns[1:])

    def averageMonthlyReturn(self):
        d_returns=self.getDailyReturns()
        m_return=np.exp(np.sum(np.log(d_returns+1)/36))-1
        return m_return

    def averageAnnualReturn(self):
        d_returns=self.getDailyReturns()
        a_return=np.exp(np.sum(np.log(d_returns+1)/3))-1
        return a_return

    def getVolatility(self):
        d_returns=self.getDailyReturns()
        a_volatility=np.std(d_returns)*np.sqrt(251)
        return a_volatility

    def monthlyReturnArray(self):
        df=self.getMonthlyData()
        m_returns=df.pct_change()
        
        return np.array(m_returns[1:])

    def excessReturnArray(self):
        m_returns=np.array(self.monthlyReturnArray())
        # print('m_________',np.size(m_returns))        
        # print('tb________', np.size(TB_returns))

        excess_returns=np.subtract(m_returns,TB_returns)
        m_returns_SP=monthlySPReturns()
        # print('sp________', np.size(m_returns_SP))
        excessReturns_SP=np.subtract(m_returns_SP,TB_returns)
        return excessReturns_SP, excess_returns

    def logRegression(self):
        excessReturns_SP, excess_returns=self.excessReturnArray()
        x,y=np.log(excessReturns_SP+1),np.log(excess_returns+1)
        alpha, beta=estimate_coef(x, y)

        plt.plot(x, y, 'o')
        plt.plot(x, beta*x + alpha)
        plt.xlabel('SP Excess Returns')
        plt.ylabel('Asset Excess Returns')
        plt.title(f'Regression on Excess Returns of {self.ticker} over Market')
        # plt.show()

        return alpha, beta

    def predictedMonthlyReturns(self):
        alpha, beta=self.logRegression()          
        excessReturn_SP, excess_returns=self.excessReturnArray()
        x,y=np.log(excessReturn_SP+1),np.log(excess_returns+1)

        predicted_log_excess=alpha+beta*x
        predicted_excess_m=np.exp(alpha+beta*x)-1  
        
        # risk_free_returns is TB_returns
        predicted_returns_m=predicted_excess_m+TB_returns
        
        return predicted_returns_m  

    def predictedAnnualReturn(self):
        predicted_returns_m=self.predictedMonthlyReturns()
        predicted_return_a=np.exp(np.sum(np.log(predicted_returns_m+1))/3)-1
        return predicted_return_a     
    
    def monthlyTable(self):        
        returns_m=(self.monthlyReturnArray())
        average_returns_m=self.averageMonthlyReturn()
        excessReturns_SP, excess_returns=self.excessReturnArray()
        predicted_returns_m=self.predictedMonthlyReturns()

        returns_table=pd.DataFrame({"returns_m": returns_m, "average_returns_m": \
            average_returns_m, "excessReturns_SP": excessReturns_SP, \
                "excess_returns": excess_returns, \
                "predicted_returns_m":predicted_returns_m})
        
        return returns_table
        
    def summaryTable(self):
        a_return=self.averageAnnualReturn()
        a_predicted_return=self.predictedAnnualReturn()
        a_volatility=self.getVolatility()
        alpha, beta=self.logRegression()

        summary1=dict({"Expected Anual Return": a_return, 
                         "Predicted Annual Return": a_predicted_return, 
                         "Annual Volatility": a_volatility, 
                         "Alpha": alpha, "Beta": beta  })

        a_return_pct=str(round(a_return*100, 2)) + '%'
        a_predicted_return_pct=str(round(a_predicted_return*100, 2)) + '%'
        a_volatility_pct=str(round(a_volatility*100, 2)) + '%'        
        alpha_pct=str(round(alpha*100, 2)) + '%'
        beta_pct=str(round(beta*100, 2)) + '%'

        summary=f"""Asset Ticker: {self.ticker}
Expected Anual Return: {a_return_pct},
Predicted Annual Return: {a_predicted_return_pct}, 
Annual Volatility: {a_volatility_pct}, 
Alpha: {alpha_pct}, 
Beta: {beta_pct}"""
        return summary

# Test cases:
#      
asset1=Asset('amzn')

print('asset1 predicted returns monthly:',asset1.predictedMonthlyReturns())
# print('monthlyReturnArray:\n', asset1.monthlyReturnArray())
# print('average annual\n', asset1.averageAnnualReturn())
# print('predicted annual return\n', asset1.predictedAnnualReturn())
# print('volatility:\n', asset1.getVolatility())
# print('returns_table \n', asset1.monthlyTable())
# print(asset1.summaryTable())






