# portfolio-optimiser

Overview

This Python application helps with the management and optimization of investment portfolios. It retrieves the historical data of assets from the last 3 years from Yahoo Finance and predicts the excess returns of risky assets through regression analysis between the excess returns of individual assets and the S&P 500 index.

Features

Asset Integration: Users can add any assets to their portfolio.
Performance Metrics: Generates expected returns, volatility, and return-risk ratio at the portfolio level based on different asset weight combinations.
Optimization: Produces optimal weights to achieve minimal risk and maximum return-risk ratio (portfolio excess return/volatility). This applies finance theories from the Efficient Frontier and Capital Allocation Line.

How to Run

Run UserInterface.py to open the application window.
Follow the on-screen guidance to navigate through each function.
Ensure all Python files, images, and Excel CSV files are in the same folder as UserInterface.py.

Shortcut Commands

Press 'e': Go to the explanation screen.
Press 's': View saved reports.
Press 'h': Open the help screen.
Press 'a': Access assets.
Press 'p': Manage portfolio.
Press 'g': View graphs.
Press 'f': Save the optimal portfolio report to the local drive.


Libraries/modules needed:
Pandas (https: //pandas.pydata.org/)
Numpy (https://numpy.org/  )
Matplotlib (https://matplotlib.org/)
SciPy (https://www.scipy.org)
Pillow ( https://pypi.org/project/Pillow/)

Youtube Demo: https://youtu.be/AqSoykQc1LA?si=ExYJ_L1GbLjThPfv
