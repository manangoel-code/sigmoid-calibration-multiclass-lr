# Kept unchanged for provenance: the original combined benchmark + plot script.
# Running it overwrites results.csv, reliability_bins.csv, summary.csv and the archived
# reliability.png in this folder. Use experiment.py and make_graph.py instead.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, log_loss

OUT = Path(__file__).resolve().parent
rows, bins = [], []
for dataset, loader in [('Breast cancer', load_breast_cancer), ('Digits', load_digits)]:
    X, y = loader(return_X_y=True)
    for seed in range(10):
        # Keep the test set separate from both fitting and calibration.
        Xdev, Xt, ydev, yt = train_test_split(X, y, test_size=.3, stratify=y, random_state=seed)
        Xfit, Xcal, yfit, ycal = train_test_split(Xdev, ydev, test_size=2/7, stratify=ydev, random_state=seed)
        models = {'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                  'Random forest': RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
                  'Naive Bayes': GaussianNB()}
        for name, model in models.items():
            model.fit(Xfit, yfit)
            calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid').fit(Xcal, ycal)
            for variant, fitted in [('Original', model), ('Sigmoid', calibrated)]:
                p = fitted.predict_proba(Xt)
                pred = fitted.classes_[p.argmax(axis=1)]
                conf, correct = p.max(axis=1), pred == yt
                onehot = (yt[:, None] == fitted.classes_[None, :]).astype(float)
                ece = 0.
                for b in range(10):
                    mask = (conf >= b/10) & (conf < (b+1)/10 if b < 9 else conf <= 1)
                    if mask.any():
                        c, a = conf[mask].mean(), correct[mask].mean()
                        ece += mask.mean() * abs(c-a)
                        bins.append([dataset, seed, name, variant, b, int(mask.sum()), c, a])
                rows.append([dataset, seed, name, variant, len(yfit), len(ycal), len(yt), accuracy_score(yt,pred), log_loss(yt,p), np.mean(np.sum((p-onehot)**2,axis=1)), ece, conf.mean()])
results = pd.DataFrame(rows, columns=['dataset','seed','model','variant','n_fit','n_calibration','n_test','accuracy','log_loss','brier','ece','mean_confidence'])
results.to_csv(OUT/'results.csv',index=False)
pd.DataFrame(bins,columns=['dataset','seed','model','variant','bin','count','confidence','accuracy']).to_csv(OUT/'reliability_bins.csv',index=False)
summary=results.groupby(['dataset','model','variant'])[['accuracy','log_loss','brier','ece']].agg(['mean','std'])
summary.to_csv(OUT/'summary.csv')
fig, axes = plt.subplots(1,2,figsize=(12,5))
for ax, dataset in zip(axes,['Breast cancer','Digits']):
    ax.plot([.5,1],[.5,1],'--',color='gray',label='Perfect calibration')
    for name in ['Logistic regression','Random forest','Naive Bayes']:
        # Plot the first split only: pooled overlapping test splits are not independent.
        for variant, style in [('Original','-'),('Sigmoid',':')]:
            b=pd.DataFrame(bins,columns=['dataset','seed','model','variant','bin','count','confidence','accuracy'])
            b=b[(b.dataset==dataset)&(b.seed==0)&(b.model==name)&(b.variant==variant)]
            ax.plot(b.confidence,b.accuracy,style,marker='o',label=f'{name}: {variant}',markersize=3)
    ax.set(title=dataset+' (seed 0)',xlabel='Mean predicted confidence',ylabel='Fraction correct',xlim=(.45,1.02),ylim=(0,1.03))
axes[1].legend(fontsize=7,loc='lower right')
fig.tight_layout();fig.savefig(OUT/'reliability.png',dpi=180)
(OUT/'environment.json').write_text(json.dumps({'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},indent=2))
print(summary.to_string())
