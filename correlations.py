# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:47:17 2023

@author: siets009
"""

import os
import pandas as pd
import numpy as np

import statsmodels.formula.api as sm
from scipy.stats import zscore, spearmanr, pearsonr

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LogNorm, Normalize
from matplotlib import gridspec

sns.set()

from simple_BERTopic import OUTPUT_FOLDER, DATA_FOLDER



#%% Load data -- same as heatmap, but now with food and GDP data added too
nr_topics = 52
input_df =  os.path.join(OUTPUT_FOLDER, f'{nr_topics}_df.csv')
topic_excel = os.path.join(OUTPUT_FOLDER, f'{nr_topics}_topics_named.xlsx')
countries_table = os.path.join(DATA_FOLDER, 'CountryMappingTable.csv')
food_index_table = os.path.join(DATA_FOLDER, 'foodInsecurity.xlsx')
gdp_table = os.path.join(DATA_FOLDER, 'WorldBankGDPwPP_Capita.csv')

df_in = pd.read_csv(input_df, index_col = 0, encoding = 'utf-8')
# We have named the topics in an excel based on  the get_and_save_representative_pages_per_topic output
topics = pd.read_excel (topic_excel)
countries =  pd.read_csv(countries_table, encoding = 'utf-8')
food  = pd.read_excel(food_index_table)
gdp = pd.read_csv(gdp_table, encoding = 'utf-8')

df = df_in.merge(topics[['topic_name', 'topic_category']], left_on = 'topic_nr', right_index = True)

# Before we add the country names, ensure they use the same country names
# (this is why we should use ISO codes, but oh well...)
no_match = set(df['country'].unique()) - set(countries['Country'].unique())
#Based on this, rename
df.loc[df['country'] == "Naura", 'country'] = "Nauru"
df.loc[df['country'] == "Turkiye", 'country'] = "Türkiye"

assert len(set(df['country'].unique()) - set(countries['Country'].unique())) == 0

# Add the country info
df =df.merge(countries, left_on = 'country', right_on = 'Country', how='left')

# Country names should be matching up already for food
assert len(set(df['country'].unique()) - set(food['country'].unique())) == 0
food.rename(columns = {'Prevalence of Moderate or Severe Food Insecurity in the Total Population 2020-22 (%)':
                       'food_score'}, inplace=True)
df =df.merge(food, left_on = 'country', right_on = 'country', how='left')
# And we have ISO for GDP -- ony take the 2021 nrs to be consistent
df =df.merge(gdp[['Country Code', '2021']], left_on = 'Alpha-3',
             right_on = 'Country Code', how='left'
             ).rename(columns = {'2021': '2021_gdp'})

# Finally, let's add a region column which is basically UN continental
# Except we split up America's in North en South
df['Region'] = df['UN continental']
df.loc[df['World Bank Region'] == 'Latin America & Caribbean', 'Region'] = 'South America'
df.loc[df['World Bank Region'] == 'North America', 'Region'] = 'North America'



#%% Plotting function for scatter plots

# Repurposing some of my code from https://doi.org/10.1007/s10113-023-02113-3
# Input a df with a 'Region' column and values (here: gdp and food insecurity index)
# Specify which columns with scores (i.e. topic prevalence) should be included

def plot_by_region(fig, ax, score, valueVars, dfs, regression = False, order=1,
                   xlabels = False, invert= False,
                   region_col = 'Region',
                   regionMarkers = { 'Oceania': 'o',
                                    'Europe': 'X',
                                    'North America': 's',
                                    'Asia': 'P',
                                    'Africa': 'D',
                                    'South America': '*'},
                   customPalette = False,
                   xlog = False, linthreshx = 10, linscalex=2.5,
                   legend ="default",
                   title ='Attention to topic categories',
                   ):
    

    #plot scores coloured by type of score, with marker by region
    n=0


    
    #to be able to plot from multiple dfs, you can input them as a list
    #NB: the order needs to line up with the score rows!
    if type(dfs) == list:
        df = dfs[n]
    else: df = dfs
    
    #Plot scatter -- first melt data
    meltDf = pd.melt(df, id_vars=[score, region_col], 
                                 value_vars= valueVars)
   

    #Formatting options depending on size/legend input
    if len(meltDf) >= 800: 
        markersize=25
        alpha = .75
    else: 
        markersize =35
        alpha = 0.85
    if legend == "default" or legend == False: l = False
    else: l = True
    
    #Plot the regions, either with pre-defined colours or with the palette
    if customPalette == False:
        customPalette = sns.color_palette('colorblind')

    for m in regionMarkers:
        sns.scatterplot(data=meltDf.loc[meltDf[region_col] == m],
                    x=score, y="value", hue="variable", marker = regionMarkers[m],
                    s = markersize, palette = customPalette,
                    ax=ax, alpha=alpha, legend=l)
    
    #Add regression line on top if needed
    if regression == True:
        for value, c in zip(valueVars, customPalette):
            sns.regplot(data=meltDf.loc[meltDf['variable'] == value],
                x=score, y="value", # hue="variable", 
                scatter=False, ci=False, order=order,
                 #markers=["o", "X", "s", "D", "v", "P"],
                ax=ax,
                line_kws = {'color':c, 'alpha' :0.85,
                            'linestyle' : 'dashed'})
            
    #Invert the axis to match the inverted data
    #NB: just flips, so if you run it twice, the axis is back to normal         
    if type(invert) == list:
        if invert[n] == True:
            ax.invert_xaxis()
    elif invert == True: #invert for all
        ax.invert_xaxis()
            
    #ax.set_ylabel("Nr of paragraphs in topic category", weight="bold")
    ax.set_ylabel("% of al submissions in topic category", weight="bold")
 
    if type(xlabels) == list:
        ax.set_xlabel(xlabels[n], weight="bold")
    elif type(xlabels) == str:
        ax.set_xlabel(xlabels, weight = "bold")
    else: ax.set_xlabel(score, weight="bold")
    
    
    if type(xlog) == list:
        if xlog[n] == 'symlog':
            ax.set_xscale('symlog', linthresh =linthreshx[n], linscale = linscalex[n])
            plt.gca().xaxis.grid(True, which='minor')
        elif xlog[n] == True:
            ax.set_xscale('log')
    elif xlog == 'symlog':
        ax.set_xscale('symlog', linthresh =linthreshx, linscale = linscalex)
        plt.gca().xaxis.grid(True, which='minor')
    elif xlog == True:
        ax.set_xscale('log')
        
    n += 1
        
    #Add a single legend for all
    #ax1 = axes[0]
    
    if legend != False:
        if legend == "default":
            #add empty plots with the right colours/labels
            h1 = ax.scatter([], [], alpha=0, label="Topic category")
            h2 = [ax.scatter([], [], color = c, label=l) for l, c in zip(valueVars, customPalette)]
            he = ax.scatter([], [], alpha=0, label=" ") #empty for spacing 
            h4 = ax.scatter([], [], alpha=0, label="Region")
            h5 = [ax.scatter([], [],  label= m, c = 'k', marker = regionMarkers[m]) for m in regionMarkers]
            handles, labels = ax.get_legend_handles_labels()
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            
        
        ax.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5),  
                   bbox_transform=plt.gcf().transFigure)
    
    fig.suptitle(title, 
                  fontsize=16, weight='bold')
    fig.tight_layout()
    
    
    
    return(fig, ax)

def r_p_values(df, correlation = 'spearman'):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    correlates = dfcols.transpose().join(dfcols, how='outer')
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            if correlation == 'spearman':
                correlates[r][c] = spearmanr(df[r], df[c], nan_policy = 'omit')[0]
                pvalues[r][c] = spearmanr(df[r], df[c], nan_policy = 'omit')[1] 
            elif correlation  == 'pearson':
                correlates[r][c] = pearsonr(df[r], df[c], nan_policy = 'omit')[0]
                pvalues[r][c] = pearsonr(df[r], df[c], nan_policy = 'omit')[1] 
    return(correlates, pvalues)



#%% Calculate the "scores": nr of paragraphs per topic category for each country

scores_dict = {}
for cat in df['topic_category'].unique():
    if cat != 'Junk':
        category_dict = {}
        for country in df['country'].unique():
            nr_in_cat = len(df.loc[(df['country'] == country
                                    ) & (df['topic_category'] == cat)])
            category_dict[country] = nr_in_cat
    scores_dict[cat] = category_dict
scores_df = pd.DataFrame.from_dict(scores_dict)

#Add on the country info again
scores_df = scores_df.merge(df.drop_duplicates(subset=['Country']), 
                            left_index = True, right_on = 'Country',
                            how = 'left' 
                            )

#The food insecurity category title is way too long
scores_df.rename(columns = {'Food insecurity & Food system resilience': 'Food insecurity'}, 
                 inplace=True)
#%% Plot GDP and food insecurity index correlations 


valueVars = ['Food insecurity', 
             #'Agriculture',
             #'UNFSS & Governance', 
             'Climate & Environment',
             'Health & social issues', 
             #'Other sectors',
             ]
scoreCols = ['food_score', '2021_gdp']


fig, ax = plt.subplots(1, 1, dpi = 300, figsize = (8.5, 6))
fig, ax = plot_by_region(fig, ax, scoreCols[0], valueVars,
                       dfs = scores_df,
                       invert=True,
                       xlabels = "Prevalence of Moderate or Severe Food Insecurity 2020-22 (%)",
                       regression = True)

plt.show()

fig2, ax2 = plt.subplots(1, 1, dpi = 300, figsize = (8.5, 6))
fig2, ax2 = plot_by_region(fig2, ax2, scoreCols[1], valueVars,
                       dfs = scores_df,
                       xlabels = "GDP per capita (USD PP)",
                       xlog = True,
                       regression = True)

plt.show()

#%% Results are not very notable. Perhaps better to plot as share of submission rather than absolute NR?

#Just calculate again
share_dict = {}
for cat in df['topic_category'].unique():
    if cat != 'Junk':
        category_dict = {}
        for country in df['country'].unique():
            nr_in_cat = len(df.loc[(df['country'] == country
                                    ) & (df['topic_category'] == cat)])
            #record the share instead as a percentage
            category_dict[country] = nr_in_cat/len(df[df['country'] == country])*100
    share_dict[cat] = category_dict
share_df = pd.DataFrame.from_dict(share_dict)

#Add on the country info again
share_df = share_df.merge(df.drop_duplicates(subset=['Country']), 
                            left_index = True, right_on = 'Country',
                            how = 'left' 
                            )

#The food insecurity category title is way too long
share_df.rename(columns = {'Food insecurity & Food system resilience': 'Food insecurity'}, 
                 inplace=True)

#%% plot
fig, ax = plt.subplots(1, 1, dpi = 300, figsize = (8.5, 6))
fig, ax = plot_by_region(fig, ax, scoreCols[0], valueVars,
                       dfs = share_df,
                       invert=True,
                       xlabels = "Prevalence of Moderate or Severe Food Insecurity 2020-22 (%)",
                       regression = True)

plt.show()

fig2, ax2 = plt.subplots(1, 1, dpi = 300, figsize = (8.5, 6))
fig2, ax2 = plot_by_region(fig2, ax2, scoreCols[1], valueVars,
                       dfs = share_df,
                       xlabels = "GDP per capita (USD PP)",
                       xlog = True,
                       regression = True)

plt.show()

#%% And plot again but with the world bank regions
regionMarkers = { 'East Asia & Pacific': 'o',
                 'Europe & Central Asia': 'X',
                 'North America': 's',
                 'South Asia': 'P',
                 'Sub-Saharan Africa': 'D',
                 'Middle East & North Africa': "v",
                 'Latin America & Caribbean': '*'}

# Absolute
fig, ax = plt.subplots(1, 1, dpi = 300, figsize = (8.5, 6))
fig, ax = plot_by_region(fig, ax, scoreCols[0], valueVars,
                       dfs = scores_df,
                       region_col='World Bank Region', regionMarkers=regionMarkers,
                       invert=True,
                       xlabels = "Prevalence of Moderate or Severe Food Insecurity 2020-22 (%)",
                       regression = True)

plt.show()

fig2, ax2 = plt.subplots(1, 1, dpi = 300, figsize = (8.5, 6))
fig2, ax2 = plot_by_region(fig2, ax2, scoreCols[1], valueVars,
                       dfs = scores_df,
                       region_col='World Bank Region', regionMarkers=regionMarkers,
                       xlabels = "GDP per capita (USD PP)",
                       xlog = True,
                       regression = True)

plt.show()


#%% All categories as a subplot
#Share

valueVars = ['Food insecurity', 
             'Agriculture',
             'UNFSS & Governance', 
             'Climate & Environment',
             'Health & social issues', 
             'Other sectors',
             ]

fig, axes = plt.subplots(1, 2, dpi = 300, figsize = (8.5, 5))
fig, ax1 = plot_by_region(fig, axes[0], scoreCols[0], valueVars[:3],
                       dfs = share_df, 
                       customPalette= sns.color_palette('colorblind')[:3],
                       legend=False,
                       region_col='World Bank Region', regionMarkers=regionMarkers,
                       invert=False,
                       xlabels = "Moderate or Severe Food Insecurity (% of pop.)",
                       regression = True)

fig, ax2 = plot_by_region(fig, axes[1], scoreCols[0], valueVars[3:],
                       dfs = share_df,
                       customPalette= sns.color_palette('colorblind')[3:6],
                       legend=False,
                       region_col='World Bank Region', regionMarkers=regionMarkers,
                       invert=False,
                       xlabels = "Moderate or Severe Food Insecurity (% of pop.)",
                       regression = True)

#The legend now becomes a bit of a problem
regionMarkers = { 'Oceania': 'o',
                  'Europe': 'X',
                  'North America': 's',
                  'Asia': 'P',
                  'Africa': 'D',
                  'South America': '*'}
h1 = ax1.scatter([], [], alpha=0, label="Topic category")
h2 = [ax1.scatter([], [], color = c, label=l) for l, c in zip(valueVars[:3], sns.color_palette('colorblind'))]
he = ax1.scatter([], [], alpha=0, label=" ") #empty for spacing 
h3 = [ax1.scatter([], [], color = c, label=l) for l, c in zip(valueVars[3:], sns.color_palette('colorblind')[3:])]
h4 = ax1.scatter([], [], alpha=0, label="Region")
h5 = [ax1.scatter([], [],  label= m, c = 'k', marker = regionMarkers[m]) for m in list(regionMarkers.keys())[:3]]
he2 = ax1.scatter([], [], alpha=0, label=" ") #empty for spacing 
h6 = h5 = [ax1.scatter([], [],  label= m, c = 'k', marker = regionMarkers[m]) for m in list(regionMarkers.keys())[3:]]
handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles, labels, ncol = 4, loc='upper center', bbox_to_anchor=(0.5, -0.05),  
            bbox_transform=plt.gcf().transFigure)

#Add A and B to the panels
fig.text(0.05, 0.9, "A", weight='bold', size = 12)
fig.text(0.55,0.9, "B", weight='bold', size = 12)
#fig.tight_layout()
plt.show()

# fig2, ax2 = plt.subplots(1, 1, dpi = 300, figsize = (8.5, 6))
# fig2, ax2 = plot_by_region(fig2, ax2, scoreCols[1], valueVars,
#                        dfs = share_df,
#                        region_col='World Bank Region', regionMarkers=regionMarkers,
#                        xlabels = "GDP per capita (USD PP)",
#                        xlog = True,
#                        regression = True)

# plt.show()

plt.show()

# fig2, ax2 = plt.subplots(1, 1, dpi = 300, figsize = (8.5, 6))
# fig2, ax2 = plot_by_region(fig2, ax2, scoreCols[1], valueVars,
#                        dfs = share_df,
#                        region_col='World Bank Region', regionMarkers=regionMarkers,
#                        xlabels = "GDP per capita (USD PP)",
#                        xlog = True,
#                        regression = True)

# plt.show()
    

#%% Stats
corDf_spearman, corDf_sP = r_p_values(share_df)
corDf_pearson, corDf_pP = r_p_values(share_df)

