import problexity as px
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('../results/CLL_MCL_ADA_BOOST/features/generation_20.csv', header=None)
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]


    complexity_calculator = px.ComplexityCalculator()
    complexity_calculator.fit(X,y)

    # print(str(complexity_calculator.complexity()))
    fig = plt.figure(figsize=(7,7))
    complexity_calculator.plot(fig,(1,1,1))

    plt.show()