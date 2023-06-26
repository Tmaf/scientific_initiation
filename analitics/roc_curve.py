import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
RANDOM_STATE = 123456

def generate_roc(df,classifier, title="",saveTitle=""):
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]
    n_samples, n_features = X.shape
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X.iloc[train, :], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X.iloc[test, :],
            y[test],
            name=f"ROC fold {fold + 1}",
            alpha=0.3,
            lw=1,
            ax=ax,
        )

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    ax.plot([0, 1], [0, 1], "k--", label="Escolha aleatória (AUC = 0.5)") # chance level

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"AUC Média(AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 Desvio Padrão",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="Falso Positivo",
        ylabel="Positivo Verdadeiro",
        title=f"Curva ROC média\n {title}",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    # plt.show()
    plt.savefig(f"{saveTitle}.png")



if __name__ == "__main__":
    rd = RandomForestClassifier(random_state=RANDOM_STATE)

        # title, classifierTitle,classifier, file1, file2
    experiments = [
        ("LLC x LCM", "Ada Boost", AdaBoostClassifier(random_state=RANDOM_STATE),
         '../results/CLL_MCL_ADA_BOOST/features/generation_1.csv',
         '../results/CLL_MCL_ADA_BOOST/features/generation_20.csv'),
        ("LLC x LCM", "Random Forest", RandomForestClassifier(random_state=RANDOM_STATE),
         '../results/CLL_MCL_RANDOM_FOREST/features/generation_1.csv',
         '../results/CLL_MCL_RANDOM_FOREST/features/generation_20.csv'),
        ("LLC x LCM", "SVM", RandomForestClassifier(random_state=RANDOM_STATE),
         '../results/CLL_MCL_LINEAR_SVM/features/generation_1.csv',
         '../results/CLL_MCL_LINEAR_SVM/features/generation_20.csv'),

        ("LLC x LF", "Ada Boost", AdaBoostClassifier(random_state=RANDOM_STATE),
         '../results/FL_CLL_ADA_BOOST/features/FL_CLL_ADA_BOOST_generation_0.csv',
         '../results/FL_CLL_ADA_BOOST/features/FL_CLL_ADA_BOOST_generation_19.csv'),
        ("LLC x LF", "Random Forest", RandomForestClassifier(random_state=RANDOM_STATE),
         '../results/FL_CLL_RANDOM_FOREST/features/FL_CLL_RANDOM_FOREST_generation_0.csv',
         '../results/FL_CLL_RANDOM_FOREST/features/FL_CLL_RANDOM_FOREST_generation_19.csv'),
        ("LLC x LF", "SVM", RandomForestClassifier(random_state=RANDOM_STATE),
         '../results/FL_CLL_LINEAR_SVM/features/generation_1.csv',
         '../results/FL_CLL_LINEAR_SVM/features/generation_20.csv'),

        ("LF x LCM", "Ada Boost", AdaBoostClassifier(random_state=RANDOM_STATE),
         '../results/FL_MCL_ADA_BOOST/features/generation_1.csv',
         '../results/FL_MCL_ADA_BOOST/features/generation_20.csv'),
        ("LF x LCM", "Random Forest", RandomForestClassifier(random_state=RANDOM_STATE),
         '../results/FL_MCL_RANDOM_FOREST/features/generation_1.csv',
         '../results/FL_MCL_RANDOM_FOREST/features/generation_20.csv'),
        ("LF x LCM", "SVM",RandomForestClassifier(random_state=RANDOM_STATE),
         '../results/FL_MCL_LINEAR_SVM/features/generation_1.csv',
         '../results/FL_MCL_LINEAR_SVM/features/generation_20.csv')
    ]

    for (title,classifierTitle,classifier, gen1, gen20) in experiments:
        df_01 = pd.read_csv(gen1, header=None, sep=',')
        generate_roc(df_01, classifier=classifier, title=f"{title} usando {classifierTitle}: Ger. 01",
                     saveTitle=f"{title}_{classifierTitle}_1")

        df_20 = pd.read_csv(gen20, header=None, sep=',')
        generate_roc(df_20, classifier=classifier, title=f"{title} usando {classifierTitle}: Ger. 20",
                     saveTitle=f"{title}_{classifierTitle}_20")

