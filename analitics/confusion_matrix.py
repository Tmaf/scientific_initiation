import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
RANDOM_STATE = 123456


def generate_confusion_matrix(df, classifier,title, classes):
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    classifier.fit(x_train,y_train)

    y_pred = classifier.predict(x_test)

    ax = plt.subplot()
    cm=confusion_matrix(y_test,y_pred)

    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="Blues")  # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predição')
    ax.set_ylabel('Classe')
    ax.set_title(f'Matrix de confusão\n{title}')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    plt.plot()
    plt.show()

if __name__ == "__main__":
    df_20 = pd.read_csv('../src/results/CLL_MCL_ADA_BOOST/features/generation_20.csv', header=None, sep=',')
    df_01 = pd.read_csv('../src/results/CLL_MCL_ADA_BOOST/features/generation_1.csv', header=None, sep=',')
    # classifier = RandomForestClassifier(random_state=RANDOM_STATE)
    # classifier = RandomForestClassifier(random_state=RANDOM_STATE)
    classifier = AdaBoostClassifier(random_state=RANDOM_STATE)

    generate_confusion_matrix(df_01, classifier, title="LLC x LCM Ada Boost: Gen. 1", classes =["CLL", "MCL"])
    generate_confusion_matrix(df_20, classifier, title="LLC x LCM Ada Boost: Gen. 20", classes =["CLL", "MCL"])
