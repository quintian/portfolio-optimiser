'''
Citation:
Pil installed from
https://pypi.org/project/Pillow/
cmu_112_graphics obtained from 
https://www.cs.cmu.edu/~112/notes/cmu_112_graphics.py 
The organization of UI is inspired by:
https://www.cs.cmu.edu/~112/notes/notes-animations-part2.html
'''
from cmu_112_graphics import*
import PIL.Image
import string
import os
from AssetAnalyser import*
from PortfolioOptimiser import*
import pandas as pd
import numpy as np

# Risk free return is a global variable:
def riskFreeReturn():
    downloadTB=pd.read_csv("TB4WK.csv")# annual basis points return on treasury
    dataTB=downloadTB[0:len(downloadTB)-1]
    returnArrayTB=np.array(dataTB['TB4WK'])
    riskFreeReturn=np.mean(dataTB)/100 # basis points convert to percentage
   
    return riskFreeReturn['TB4WK']
riskFreeReturn=riskFreeReturn()    

class MyModalApp(ModalApp):
    def appStarted(app):
        app.mainMode = MainMode()
        app.explainMode = ExplainMode()
        app.helpMode = HelpMode()
        app.historyMode = HistoryMode()
        app.setActiveMode(app.mainMode)

class HelpMode(Mode):
    def redrawAll(mode, canvas):
        helpMessage=f'''
        Note: this model takes the last three years of price data from 
        Yahoo Finance for all the computation. If it's a recent IPO, you 
        need to change the parameters in the code to see some results, 
        but not recommended for this portfolio strategy without historical 
        data backup. 
       
        Press 'a' for assets look up.
        Press 'p' for portfolio risk and returns.
        Press 'f' to save portfolio optimal reports.
        Press 'g' for graph that plotted on the results of your assets.       
        Press 'r' to restart the app. 
        (Note: the info you entered would be gone.)
        Press 'e' to display Explanation.
        Press 'h' for help.
        Press 's' for saved reports from past.
        (You can click mouse to go through different reports.)
        Press 'q' to quit.

        Please press any key to go back to main mode, then click 'a' or 'p'.
        '''
        canvas.create_text(mode.width/2,  mode.height/2,\
        text=helpMessage, font='Arial 16 bold')        
    
    def keyPressed(mode, event):
        if (event.key == 'h'):
            mode.app.setActiveMode(mode.app.helpMode)
     
        if (event.key == 'e'):
            mode.app.setActiveMode(mode.app.explainMode)

        if (event.key == 's'):
            mode.app.setActiveMode(mode.app.historyMode)

        else: mode.app.setActiveMode(mode.app.mainMode)

# HistoryMode shows some saved reports:
class HistoryMode(Mode):
    def appStarted(mode):
        mode.counter=0
        
        mode.image1 = mode.loadImage('port Optimal report.png')      
      
        mode.image2 = mode.loadImage('port5 optimization.png')
        mode.image3 = mode.loadImage('AMZN regression.png')
        mode.image4 = mode.loadImage('port report (amzn, goog, gild).png')
        mode.image5 = mode.loadImage('port report (spy, wmt, biib).png')
        mode.image6 = mode.loadImage('baba jpm optimal.png')
              
    def mousePressed(mode, event):
        mode.counter+=1  

    def redrawAll(mode, canvas):
        if (mode.counter%6 == 0):
            canvas.create_image(400, 400, image=ImageTk.PhotoImage(mode.image1))

        if (mode.counter%6 == 1):
            canvas.create_image(400, 400, image=ImageTk.PhotoImage(mode.image2))

        if (mode.counter%6 == 2):
            canvas.create_image(400, 400, image=ImageTk.PhotoImage(mode.image3))

        if (mode.counter%6 == 3):
            canvas.create_image(400, 400, image=ImageTk.PhotoImage(mode.image4))

        if (mode.counter%6 == 4):
            canvas.create_image(400, 400, image=ImageTk.PhotoImage(mode.image5))

        if (mode.counter%6 == 5):
            canvas.create_image(400, 400, image=ImageTk.PhotoImage(mode.image6))
    
    def keyPressed(mode, event):
        if (event.key == 'h'):
            mode.app.setActiveMode(mode.app.helpMode)
     
        if (event.key == 'e'):
            mode.app.setActiveMode(mode.app.explainMode)

        if (event.key == 's'):
            mode.app.setActiveMode(mode.app.historyMode)

        else: mode.app.setActiveMode(mode.app.mainMode)

# the below shows the explanation of methodology and theories behind this app:
class ExplainMode(Mode):
    def appStarted(mode):
        mode.counter=0
        
        mode.image1 = mode.loadImage('Regression.png')
       
        mode.image1 = mode.scaleImage(mode.image1, 2/3)
        mode.image2 = mode.loadImage('explainPic.jpg')
        
        mode.image2 = mode.scaleImage(mode.image2, 2/3)
      
    def mousePressed(mode, event):
        mode.counter+=1       
            

    def redrawAll(mode, canvas):
        if (mode.counter%2 == 0):
            canvas.create_image(400, 200, image=ImageTk.PhotoImage(mode.image1))
            
            explainMessage=f'''
            The regression is taken between the natural logs of assets' excess 
        returns and market(i.e. S&P 500) excess returns, then gengerated the 
        predicted expected returns of assets from the regression results. 

            The blue dots are the actual distribution of excess return sets 
        between market and assets from the past. The red line is the predicted 
        excess returns based on market excess returns. 
        
            Alpha is the intercept of the red line with Y axis, i.e. the 
        assets' excess return while the market excess return is 0.

            Beta is the slope of the red line, i.e. the ratio of the assets' 
        excess returns over the the market excess returns.            
        
            Please click mouse to see another page of explanation.

            Please press any key to go back main mode.
            '''
            canvas.create_text(mode.width/2,  mode.height*3/4,\
            text=explainMessage, font='Arial 12 bold')        
    
        if (mode.counter%2 == 1):

            canvas.create_image(400, 200, image=ImageTk.PhotoImage(mode.image2))
            
            explainMessage=f'''
            The distribution of a portfolio’s risk and return sets is like the green
        parabola in the graph. Capital allocation line is the yellow line with risk
        free return here. The tangent point of these two lines has the 
        highest slope, i.e. the best Return Risk Ratio,  thus is the optimal point
        for this portfolio. 

            This application will apply Efficient Fronterior theory and mathimatical 
        compuation to help you find the optimal return, optimal return risk
        ratio and optimal weights of each asset for your investment. 

            Please click mouse to see another page of explanation.

            Please press any key to go back main mode.
            '''
            canvas.create_text(mode.width/2,  mode.height*3/4,\
            text=explainMessage, font='Arial 12 bold')        
    
    def keyPressed(mode, event):
        if (event.key == 'h'):
            mode.app.setActiveMode(mode.app.helpMode)
     
        if (event.key == 'e'):
            mode.app.setActiveMode(mode.app.explainMode)

        if (event.key == 's'):
            mode.app.setActiveMode(mode.app.historyMode)

        else: mode.app.setActiveMode(mode.app.mainMode)

class MainMode(Mode):
    def appStarted(mode):
        mode.round=0        
        mode.A=True        
        mode.asset=None
        mode.assets=None
        mode.weights=None
        mode.optimal=None
        mode.yourResult=None
        mode.summaryA=None
        mode.graph=None
              

    def keyPressed(mode, event):
        mode.round+=1
        
        if (event.key == 'h'):
            mode.app.setActiveMode(mode.app.helpMode)
     
        if (event.key == 'e'):
            mode.app.setActiveMode(mode.app.explainMode)

        if (event.key == 's'):
            mode.app.setActiveMode(mode.app.historyMode)     

        if (event.key=='r'):
            mode.appStarted()

        if (event.key == 'q'):
            sys.exit()          
        # go to assets' functions:
        if (event.key == 'a'):
            mode.A=True
            mode.asset=(mode.getUserInput('''Please type the ticker of any asset 
                        you are interested to see. '''))
            if mode.asset==None or mode.asset=="":
                mode.app.setActiveMode(mode.app.helpMode)
            else:
                assetObject = Asset(mode.asset)
                
                mode.summaryA = assetObject.summaryTable()                              
        # plot the graph of asset with regression test:    
        if event.key=='g':           
           
            plt.show()            

        # go to portfolio functions:
        if (event.key=='p'):
            mode.A=False
            mode.assets=(mode.getUserInput('''Please input the assets for your 
        portfolio, seperated by ",". 
    You can choose default portfolio by clicking 'OK' or 'Enter'. '''))
            mode.weights=(mode.getUserInput('''Please input the weights for your 
        assets, seperated by ",".
    Or you can choose equal weights of your assets without inputs by 
            clicking 'Enter' or "OK".'''))
        
            if mode.assets==None:
                mode.app.setActiveMode(mode.app.helpMode)
            else:
                if mode.assets=="":
                    mode.assets=['NVDA', 'JPM', 'WMT', 'GE', 'SPY']
                    port=Portfolio(mode.assets)
                    port_expected_return, port_volatility=\
                        port.equalWeightsRiskReturn()

                else:       
                
                    A=[]            
                    for each in mode.assets.split(','):
                        each=each.strip()
                        A.append(each)
                    mode.assets=A
                    assets=A
                    print(mode.assets) 
                    assets=mode.assets
                    port=Portfolio(assets) # Portfolio Object in Optimiser.py                   
                
                if (mode.weights=="" or mode.weights==None):
                    
                    port_expected_return, port_volatility=\
                    port.equalWeightsRiskReturn()
                    
                elif (mode.weights!="" and mode.weights!=None):
                    weights=[]
                    for w in mode.weights.split(','):
                        w=w.strip()
                        weights.append(float(w))
                    mode.weights=weights
                    print('mode:', mode.weights)
                        
                    port_expected_return, port_volatility=\
                        port.getPortVolatilityReturn(mode.weights) 
                
                yourReturn=port_expected_return, port_volatility
            
                riskReturnRatio=(port_expected_return-riskFreeReturn)/port_volatility

                port_expected_return_pct=str(round(port_expected_return*100, 2)) + '%'
                port_volatility_pct=str(round(port_volatility*100, 2)) + '%'
                riskReturnRatio_pct=str(round(riskReturnRatio, 2))                
          
                
                mode.yourResult=f'''With the weights in your inputs,
        the Expected Return of Portfolio is {port_expected_return_pct}.
        the Portfolio Volatility is {port_volatility_pct}.
        The Return Risk Ratio is {riskReturnRatio_pct}.'''

                optimalER, optimalRatio, optimalWeights=port.optimalSolver()

                optimalER_pct=str(round(optimalER * 100, 2)) + '%'
                optimalRatio_pct=str(round(optimalRatio, 2))
                optimalWeights_pct=[]
                for weight in optimalWeights:
                    weight_pct=str(round(weight * 100, 2)) + '%'
                    optimalWeights_pct.append(weight_pct)    

                mode.optimal=f'''With the Optimal Solver of this application,
        The Optimal Expected Return for your portfolio is {optimalER_pct}.
        The Optimal Return Risk Ratio for your portfolio is {optimalRatio_pct}.
        The corresponding Optimal Weights suggested to take are 
        {optimalWeights_pct} 
        for your assets: {mode.assets}.
        ''' 
        # save the optimal report with the file name you input:    
        if (event.key=='f') and (mode.optimal!=None):
            fileName=(mode.getUserInput('''Please type the file name you want to save.'''))        
            port=Portfolio(mode.assets)
            port.writeFile(f"{fileName}.txt", mode.optimal)
        # plot the graph of portfolio:
        if (event.key=='g') and (mode.assets!=None):
           
            port=Portfolio(mode.assets) 
            port.plotGraph()  
            plt.show()                 
   
    def redrawAll(mode, canvas):
        if mode.round==0:           

            canvas.create_text(mode.width/2,  mode.height/2,\

                    text='''Hello! Welcome to Investment Portfolio Optimiser!

        Please press:
        'a' for assets, 
        'p' for portfolio, 
        'e' for explanation,
        's' for saved reports,
        'h' for help.''', \
            font='Arial 18 bold')                         

        if mode.A==True and (mode.asset!=None):            
            
            canvas.create_text(mode.width/2,  mode.height/2,\
                           text=f'''{mode.summaryA}

            Please click 'g' for graph plotted on your assets.''', \
                font='Arial 16 bold')

        if (mode.A==False) and (mode.assets!=None) and (mode.weights!=None):
                   
            canvas.create_text(mode.width/2,  mode.height/5,\

                    text=mode.yourResult, font='Arial 16 bold' )               
      
            canvas.create_text(mode.width/2,  mode.height/2,\

        text=mode.optimal, font='Arial 16 bold', fill="green") 
        
            canvas.create_text(mode.width/2,  mode.height*4/5,\

        text='''You can click 'f' to save Optimal report as a text file 
in your computer, and click 'g' to see the graph plotted
on the results of your portfolio.''', font='Times 14 bold')                            



app = MyModalApp(width=800, height=800)

