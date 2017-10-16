import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc(y_test, y_score, ax=None, title=None, **kwargs):
    """Plots the ROC curve for a binary classifier.
    
    Inputs:
    y_test - The true values of the observations.
    y_score - The corresponding scores.
    ax - Matplotlib axes object (Default: None)
    title - Plotting title. It will add AUC after this. (Default: None)
    kwargs - Matplotlib keyword arguments
    """
    auc_val = roc_auc_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    
    def _get_plot_title(title):
        plot_title = 'AUC: {:.3f}'.format(auc_val)
        if title is not None:
            plot_title = '{}\n{}'.format(title, plot_title)
        return plot_title
    
    plot_title = _get_plot_title(title)
        
    if ax is None:
        plt.plot(fpr, tpr, **kwargs)
        plt.title(plot_title)
        plt.xlabel('FPR')
        plt.ylabel('TPR')

        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        ax.plot(fpr, tpr, **kwargs)
        ax.set_title(plot_title)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    
def plot_feature_importances(clf, feat_names, top_n=None, **kwargs):
    """Plots the top feature importances.
    
    Inputs:
    clf - A DecisionTreeClassifier, DecisionTreeRegressor,
          RandomForestClassifier, or RandomForestRegressor object
    feat_names - A list of the feature names
    top_n - The number of top features to plot. If None, plot all
            features (Default: None)
    kwargs - Matplotlib keyword arguments
    
    Returns a DataFrame of the feature importances.
    """
    
    if not (isinstance(clf, DecisionTreeClassifier)
            or isinstance(clf, DecisionTreeRegressor)
            or isinstance(clf, RandomForestClassifier)
            or isinstance(clf, RandomForestRegressor)):
        raise TypeError('clf should be one of (RandomForestClassifier, RandomForestRegressor, RandomForestClassifier, RandomForestRegressor)')
        
    feat_imp_df = pd.DataFrame()
    feat_imp_df['feat_name'] = feat_names
    feat_imp_df['feat_importance'] = clf.feature_importances_
    
    feat_imp_df = feat_imp_df\
        .set_index('feat_name')\
        .sort_values('feat_importance')
        
    if top_n is not None:
        plot_df = feat_imp_df.tail(top_n)
    else:
        plot_df = feat_imp_df
        
    plot_df.plot(kind='barh', legend=None)
    
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    
    return feat_imp_df.iloc[::-1]
    

def plot_regression_coefficients(clf, feat_names, top_n=None, **kwargs):
    """Plots the most extreme regression coefficients.
    
    feat_names - A list of the feature names
    top_n - The number of most extreme features to plot. If None, plot 
            all features (Default: None)
    kwargs - Matplotlib keyword arguments
    
    Returns a DataFrame of the regression coefficients.
    """
    
    if not (isinstance(clf, ElasticNet)
            or isinstance(clf, LinearRegression)
            or isinstance(clf, LogisticRegression)):
        raise TypeError('clf should be one of (ElasticNet, LinearRegression, LogisticRegression)')
        
    def _create_coef_df(clf, feat_names):
        reg_coef_df = pd.DataFrame()
        reg_coef_df['feat_name'] = feat_names
        if isinstance(clf, LogisticRegression):
            reg_coef_df['coef'] = clf.coef_[0]
        else:
            reg_coef_df['coef'] = clf.coef_

        return reg_coef_df\
            .set_index('feat_name')\
            .sort_values('coef')
            
    reg_coef_df = _create_coef_df(clf, feat_names)
        
    if top_n is not None:
        # Most negative coefficients
        reg_coef_df_head = reg_coef_df.head(top_n)
        # Most positive coefficients
        reg_coef_df_tail = reg_coef_df.tail(top_n)
    else:
        # Most negative coefficients
        reg_coef_df_head = reg_coef_df
        # Most positive coefficients
        reg_coef_df_tail = reg_coef_df
    
    # Get yticks
    plot_feat_names = reg_coef_df_head.index.tolist()\
                      + [' ']\
                      + reg_coef_df_tail.index.tolist()
    
    # Plot bar chart
    plt.barh(np.arange(1, top_n+1), reg_coef_df_tail.coef, color=green)
    plt.plot([0, 0], [-top_n - 0.5, top_n + 0.5], '--', color=black)
    plt.barh(np.arange(-top_n, 0), reg_coef_df_head.coef, color=red)
    
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature Name')
    plt.yticks(np.arange(-top_n, top_n + 1), plot_feat_names)
    
    return reg_coef_df.iloc[::-1]
