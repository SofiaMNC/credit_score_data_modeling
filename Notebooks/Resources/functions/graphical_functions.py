'''
    This module defines a set of graphical
    functions for project 7.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from PIL import Image

import helper_functions as hf

#------------------------------------------

def plot_repartition(data, title, long, larg):
    ''' 
        Plots a pie chart of the repartition of the 
        values in a given series.
        
        Parameters
        ----------------
        data   : pandas series

        title  : string
                 The title to give the plot

        long   : int 
                 The length of the figure for the plot
        
        larg   : int
                 The width of the figure for the plot
                 
        Returns
        ---------------
        -
    '''
    
    TITLE_SIZE = 35
    TITLE_PAD = 60

    # Initialize the figure
    f, ax = plt.subplots(figsize=(long, larg))


    # Set figure title
    plt.title(title, fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
       
    # Put everything in bold
    plt.rcParams["font.weight"] = "bold"


    # Create pie chart for topics
    a = data.value_counts(normalize=True).plot(kind='pie', 
                                               autopct=lambda x:'{:1.2f}'.format(x) + '%', 
                                               fontsize =30)
    # Remove y axis label
    ax.set_ylabel('')
    
    # Make pie chart round, not elliptic
    plt.axis('equal') 
    
    # Display the figure
    plt.show()

#------------------------------------------

def plot_percentage_missing_values_for(data, long, larg):
    '''
        Plots the proportions of filled / missing values for each unique value
        in column as a horizontal bar chart.

        Parameters
        ----------------
        data : pandas dataframe with:
                - a column column
                - a column "Percent Filled"
                - a column "Percent Missing"
                - a column "Total"

       long : int
            The length of the figure for the plot

        larg : int
               The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    data_to_plot = hf.get_missing_values_percent_per(data)\
                     .sort_values("Percent Filled").reset_index()

    # Constants for the plot
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50

    sns.set(style="whitegrid")

    #sns.set_palette(sns.dark_palette("purple", reverse=True))

    # Initialize the matplotlib figure
    _, axis = plt.subplots(figsize=(long, larg))

    plt.title("PROPORTIONS DE VALEURS RENSEIGNÉES / NON-RENSEIGNÉES PAR COLONNE",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    handle_plot_1 = sns.barplot(x="Total", y="index",
                                data=data_to_plot,
                                label="non renseignées",
                                color="thistle", alpha=0.3)

    handle_plot_1.set_xticklabels(handle_plot_1.get_xticks(),
                                  size=TICK_SIZE)
    _, ylabels = plt.yticks()
    handle_plot_1.set_yticklabels(ylabels, size=TICK_SIZE)


    # Plot the Percent Filled values
    handle_plot_2 = sns.barplot(x="Percent Filled",
                                y="index",
                                data=data_to_plot,
                                label="renseignées",
                                color="darkviolet")

    handle_plot_2.set_xticklabels(handle_plot_2.get_xticks(),
                                  size=TICK_SIZE)
    handle_plot_2.set_yticklabels(ylabels, size=TICK_SIZE)


    # Add a legend and informative axis label
    axis.legend(bbox_to_anchor=(1.04, 0), loc="lower left",
                borderaxespad=0, ncol=1, frameon=True, fontsize=LEGEND_SIZE)

    axis.set(ylabel="Colonnes", xlabel="Pourcentage de valeurs (%)")

    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label, fontsize=LABEL_SIZE,
                    labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label, fontsize=LABEL_SIZE,
                    labelpad=LABEL_PAD, fontweight="bold")

    axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,
                                                               pos: '{:2d}'.format(int(x)) + '%'))
    axis.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    # Display the figure
    plt.show()

#------------------------------------------

def plot_barplot(data, y_feature, title, long, larg):
    '''
        Plots a barplot of y_feature = f(x_feature) in data

        Parameters
        ----------------
        data      : pandas dataframe with:
                    - a qualitative column named x_feature
                    - a quantitative column named y_feature

        y_feature : string
                    The name of a feature in data

        title     : string
                    The title to give the plot

        long      : int
                    The length of the figure for the plot

         larg     : int
                    The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    # Constants for the plot
    TITLE_SIZE = 50
    TITLE_PAD = 80

    sns.set(style="whitegrid")

    sns.set_palette(sns.dark_palette("purple", reverse=True))

    plt.figure(figsize = (long, larg))

    # Graph the age bins and the average of the target as a bar plot
    plt.bar(data.index.astype(str), 100 * data[y_feature])

    # Plot labeling
    plt.xticks(rotation = 75); 

    plt.xlabel('Années (tranches)')

    plt.ylabel('Défaut de remboursement (%)')

    plt.title(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, 
              pad=TITLE_PAD)


    # Display the figure
    plt.show()
                   
#------------------------------------------

def plotBoxPlots(data, long, larg, nb_rows, nb_cols):
    '''
        Displays a boxplot for each column of data.
        
        Parameters
        ----------------
        data : dataframe containing exclusively quantitative data
                                 
        long : int
               The length of the figure for the plot
        
        larg : int
               The width of the figure for the plot
               
        nb_rows : int
                  The number of rows in the subplot
        
        nb_cols : int
                  The number of cols in the subplot
                                  
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 35
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 25
    LABEL_PAD = 10

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("VALEURS QUANTITATIVES - DISTRIBUTION", fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)


    row = 0
    column = 0

    for ind_quant in data.columns.tolist():
        ax = axes[row, column]

        sns.despine(left=True)

        b = sns.boxplot(x=data[ind_quant], ax=ax, color="darkviolet")

        plt.setp(axes, yticks=[])

        plt.tight_layout()

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

        lx = ax.get_xlabel()
        ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
        if ind_quant == "salt_100g":
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        ly = ax.get_ylabel()
        ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        ax.tick_params(axis='both', which='major', pad=TICK_PAD)

        ax.xaxis.grid(True)
        ax.set(ylabel="")

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

#------------------------------------------

def plotDistplotsWithRug(data, long, larg, nb_rows, nb_cols):
    '''
        Plots the distribution of all columns in the given
        dataframe (must be quantitative columns only) coupled
        with a rug plot of the distribution
        
        Parameters
        ----------------
        data : dataframe containing exclusively quantitative data
                                 
        long : int
               The length of the figure for the plot
        
        larg : int
               The width of the figure for the plot
               
        nb_rows : int
                  The number of rows in the subplot
        
        nb_cols : int
                  The number of cols in the subplot
                                 
        Returns
        ---------------
        -
    '''
        
    TITLE_SIZE = 30
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    LABEL_SIZE = 20
    LABEL_PAD = 30

    sns.set_palette(sns.dark_palette("purple", reverse="True"))

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("DISTRIBUTION DES VALEURS QUANTITATIVES", fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = 0
    column = 0

    for ind_quant in data.columns.tolist():

        sns.despine(left=True)

        ax = axes[row, column]

        b = sns.distplot(data[ind_quant], ax=ax, rug=True)

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
        b.set_xlabel(ind_quant,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        if ind_quant in ["saturated-fat_100g", "salt_100g"]:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        plt.setp(axes, yticks=[])

        plt.tight_layout()

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

#------------------------------------------

def plot_qualitative_dist(data, nb_rows, nb_cols, long, larg, title):
    '''
        Displays a bar chart showing the frequency of the modalities
        for each column of data.

        Parameters
        ----------------
        data    : dataframe
                  Working data containing exclusively qualitative data

        nb_rows : int
                  The number of rows in the subplot
        
        nb_cols : int
                  The number of cols in the subplot

        long    : int
                  The length of the figure for the plot

        larg    : int
                  The width of the figure for the plot

        title   : string
                  The title to give the plot

        Returns
        ---------------
        -
    '''

    # Contants for the plot
    TITLE_SIZE = 130
    TITLE_PAD = 1.05
    TICK_SIZE = 50
    LABEL_SIZE = 100
    LABEL_PAD = 30

    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    fig.suptitle("DISTRIBUTION DES VALEURS QUALITATIVES"+" "+title,
                 fontweight="bold", fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = column = 0
    
    for ind_qual in data.columns.tolist():
        
        data_to_plot = data.sort_values(by=ind_qual).copy()
        
        if(nb_rows == 1 and nb_cols == 1):
            axis = axes
        elif(nb_rows == 1 or nb_cols == 1):
            if nb_rows == 1:
                axis = axes[column]
            else:
                axis = axes[row]
        else:
            axis = axes[row, column]

        plot_handle = sns.countplot(y=ind_qual,
                                    data=data_to_plot,
                                    color="darkviolet",
                                    ax=axis,
                                    order=data_to_plot[ind_qual].value_counts().index)

        plt.tight_layout()

        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=0.5, hspace=0.2)

        plot_handle.set_xticklabels(plot_handle.get_xticks(), size=TICK_SIZE)
    
        yticks = [item.get_text().upper() for item in axis.get_yticklabels()]
        plot_handle.set_yticklabels(yticks, size=TICK_SIZE, weight="bold")
        
        x_label = axis.get_xlabel()
        axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        y_label = axis.get_ylabel()
        axis.set_ylabel(y_label.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        axis.xaxis.grid(True)

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

#------------------------------------------

def plot_correlation_heatmap(data, long, larg):
    '''
        Plots a heatmap of the Spearman correlation coefficients
        between the quantitative columns in data
        
        ----------------
        - data : a dataframe containing the data

        - long : int
                 The length of the figure for the plot
        
        - larg : int
                 The width of the figure for the plot
        
        Returns
        ---------------
        _
    '''
    
    TITLE_SIZE = 40
    TITLE_PAD = 1
    TICK_SIZE = 20
    LABEL_SIZE = 45
    LABEL_PAD = 30
    
    f, ax = plt.subplots(figsize=(long, larg))
                
    f.suptitle("MATRICE DE CORRÉLATION - SPEARMAN", fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.heatmap(data, mask=np.zeros_like(data, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax,
                    annot=data, annot_kws={"fontsize":20}, fmt=".2f")

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    b.set_xticklabels(xlabels, size=TICK_SIZE, weight="bold")
    b.set_xlabel(data.columns.name,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")
    b.set_ylabel(data.index.name,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()

#------------------------------------------

def plotKDE(data, column, groupby_col, long, larg, title):
    '''
        Plots a KDE plot of column in data, grouped by
        groupby_col.
        
        ----------------
        - data : a dataframe containing the data

        - column : the correlation method ("pearson" or "spearman")

        - groupby_col : int
                 The length of the figure for the plot

        - long : int
                 The length of the figure for the plot

        - larg : int
                 The width of the figure for the plot

        - title : string
                  The title to give the plot
        
        Returns
        ---------------
        _
    '''

    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 30
    LABEL_SIZE = 25
    LEGEND_SIZE = 15

    sns.set(style="whitegrid")

    sns.despine(left=True)

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)

    plt.setp(ax, yticks=[])

    for groupby_criterion, data_df in data.groupby([groupby_col]):
        sns.kdeplot(data=data_df[column],
                    label= groupby_criterion, shade=True)
        
    ax.xaxis.grid(False)

    plt.legend()

#------------------------------------------

def plot_feature_importances2(df, threshold = 0.9):
    '''
        Plots 15 most important features and the cumulative importance of features.
        Prints the number of features needed to reach threshold cumulative importance.
    
        Parameters
        --------
        df : dataframe
            Dataframe of feature importances. Columns must be feature and importance
        threshold : float, default = 0.9
            Threshold for prining information about cumulative importances
        
        Return
        --------
        df : dataframe
            Dataframe ordered by feature importances with a normalized column (sums to 1)
            and a cumulative importance column
    '''
    
    plt.rcParams['font.size'] = 18
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    # Cumulative importance plot
    plt.figure(figsize = (8, 6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features'); plt.ylabel('Cumulative Importance'); 
    plt.title('Cumulative Feature Importance');
    plt.show();
    
    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))
    
    return df