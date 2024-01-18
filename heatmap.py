# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:47:01 2023

@author: siets009
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LogNorm, Normalize

sns.set()

from simple_BERTopic import OUTPUT_FOLDER, DATA_FOLDER

#%% Load data
nr_topics = 52
input_df =  os.path.join(OUTPUT_FOLDER, f'{nr_topics}_df.csv')
topic_excel = os.path.join(OUTPUT_FOLDER, f'{nr_topics}_topics_named.xlsx')
countries_table = os.path.join(DATA_FOLDER, 'CountryMappingTable.csv')

df_in = pd.read_csv(input_df, index_col = 0, encoding = 'utf-8')
#Have named the topics in an excel based on  the get_and_save_representative_pages_per_topic output
topics = pd.read_excel (topic_excel)
countries =  pd.read_csv(countries_table, encoding = 'utf-8')

df = df_in.merge(topics[['topic_name', 'topic_category', 'topic_nr']], on = 'topic_nr')

#%% Also, add the country info
#First, check if we have matching names in the country table
# (this is why we should use ISO codes, but oh well...)
no_match = set(df['country'].unique()) - set(countries['Country'].unique())
#Based on this, rename
df.loc[df['country'] == "Naura", 'country'] = "Nauru"
df.loc[df['country'] == "Turkiye", 'country'] = "Türkiye"

assert len(set(df['country'].unique()) - set(countries['Country'].unique())) == 0

# Add the country info
df =df.merge(countries, left_on = 'country', right_on = 'Country', how='left')
#%% get the counts

def plot_heatmap(df, colx, coly, zero_as_nan = True, fig = None, ax = None,
                 col_normalise = True, log_normalise = True,
                 x_sort = True, y_sort =True, y_axis = 'all', x_axis = 'all', title = None):
    
    df.dropna(subset=[colx, coly], inplace=True)
    df = df.astype(str)
    
    # Input for axes is either a user-defined list or 'all'
    # We may want to sort that alphabetically and/or by (multiple) group(s)
    if y_axis == 'all':
        if y_sort == True:
            y_axis = np.sort(df[coly].unique())
        elif isinstance(y_sort, (str, list)):
            df = df.sort_values(y_sort)
            y_axis = df[coly].unique()
        else:
            y_axis = df[coly].unique()
    else: assert isinstance(y_axis, list)
    if x_axis =='all':
        if x_sort == True:
            x_axis = df[colx].unique()
        elif isinstance(x_sort, (str, list)):
            df = df.sort_values(x_sort)
            x_axis = df[colx].unique()
        else:
            x_axis = df[colx].unique()
    else: assert isinstance(x_axis, list)
    
    
    #Create a matrix with counts
    counts = {}
    for xs in x_axis:
        xs_dict = {}
        for ys in y_axis:
            count = len(df.loc[(df[colx] == xs) &
                                             (df[coly] == ys)])
            if zero_as_nan and count == 0:
                xs_dict[ys] = np.nan
            else:
                xs_dict[ys] = count

        counts[xs] = xs_dict
    df_count = pd.DataFrame.from_dict(counts)
    
    #Colours normalised by column -- else, keep absolute
    if col_normalise:
        df_norm = 100*df_count/df_count.sum()
        annot = df_count
        cbar_format = '%.0f%%'
        cbar_label = 'Percentage of paragraphs in column'
    else:
        df_norm = df_count
        annot = False
        cbar_format = '%.0f'
        cbar_label = 'Number of paragraphs'
    if log_normalise:
        norm = LogNorm()
    else:
        norm = None
    
    #Start plot
    if fig == None:
        fig, ax = plt.subplots(1,1, figsize = (10, 10), dpi=200)
    
    #Plot blue to yellow on a white background
    cmap = colormaps.get_cmap('viridis_r')
    cmap.set_bad("w")
    
    ax = sns.heatmap(df_norm, annot=annot, linewidth=.01,
                     norm=norm, 
                     xticklabels=True, yticklabels=True, #Force to display all labels
                     fmt='.0f',
                     mask = df_count.isna(),
                     linecolor = '0.8',
                     cmap = cmap, cbar_kws={"shrink": 0.5,
                                            "pad": 0.1,
                                            'format': cbar_format,
                                            #"extend": 'both',
                                            'label': cbar_label})
    
    plt.xticks(rotation=45, ha='right')
    
    if title  != None:
        ax.set_title(title)
        
    plt.tight_layout()
    return(fig, ax)

#%% Plot all countries

#count_df = counts_for_heatmap(df.sort_values('topic_category'), 'country', 'topic_name')
fig, ax = plt.subplots(1,1, figsize = (32, 12), dpi=200)
fig, ax = plot_heatmap(df, 'country', 'topic_name',
                       fig = fig, ax = ax,
                       col_normalise= False,
                       title = "Heatmap All - No Normalisation")

#%% Probably better if we agregate and/or sort by region 

#Also, flip the axes

fig, ax = plt.subplots(1,1, figsize = (14, 28), dpi=200)
fig, ax = plot_heatmap(df, 'topic_name', 'country',
                       x_sort = ['topic_category', 'topic_name'],
                       y_sort = ['World Bank Region', 'country'],
                       fig = fig, ax = ax,
                       col_normalise= False,
                       title = "Heatmap All - No Normalisation")
#Lets add some black bars around the groups
for x in [9, 16, 25, 38, 48, 49]:
    plt.axvline(x, lw = 2, c = 'k')
for y in [24, 50, 78, 80, 86]:
    plt.axhline(y, lw = 2, c = 'k')

fig.tight_layout()
fig.show()


#%% Now plot the more agregated & set order manually
fig, ax = plt.subplots(1,1, figsize = (7, 6), dpi=200)
y_ordered = ['General & Governance', 'Agriculture', 'Food insecurity & Food system resilience',
             'Climate & Environment','Health & social issues', 'Other sectors', #'Junk'
             ]
x_ordered = ["North America", "Europe & Central Asia", 
             "East Asia & Pacific", "South Asia", 
             "Middle East & North Africa", "Sub-Saharan Africa", "Latin America & Caribbean"]

fig, ax = plot_heatmap(df, 'World Bank Region', 'topic_category',
                       fig = fig, ax = ax,
                       y_axis= y_ordered, x_axis = x_ordered,
                       title = None #"Number of paragraphs per topic category and region - normalised by column"
                       )

#%% Aggregated level - income
fig, ax = plt.subplots(1,1, figsize = (7, 6), dpi=300)
y_ordered = ['General & Governance', 'Agriculture', 'Food insecurity & Food system resilience',
             'Climate & Environment','Health & social issues', 'Other sectors', #'Junk'
             ]
x_ordered = ['High', 'Upper-middle', 'Lower-middle', 'Low']

fig, ax = plot_heatmap(df, 'World Bank Income', 'topic_category',
                       fig = fig, ax = ax,
                       log_normalise= False,
                       y_axis= y_ordered, x_axis = x_ordered,
                       title = None#"Number of paragraphs per topic category and region - normalised by column"
                       )


#%% All topics - region

fig, ax = plt.subplots(1,1, figsize = (9, 14), dpi=300)

#Ordering by hand is a bit of a pain, but easier than writing multi-level sort
y_ordered = ['Addressing food challenges', 'Dialogues', 'Food security strategy', 'Food system', 'Governance', 'Governance process', 'Government', 'Government action', 'SDGs', 'Spatial data', 'Timeline', 'UNFSS', 'UNFSS Action tracks', 
             'Agricultural development', 'Agricultural digitalization', 'Agricultural employment', 'Agricultural innovation', 'Agricultural market challenges', 'Agricultural production', 'Domestic production', 'International agricultural cooperation', 'Livestock breeding', 
             'Civil protection', 'COVID-19', 'Economic vulnerabilities', 'Food system challenges', 'Food system development', 'Fortification', 'Humanitarian aid', 'Imports', 'Undernutrition', 
             'Agri-environmental measures', 'Climate change', 'Climate impacts', 'Climate resilience', 'Environment', 'Waste management', 'Water management',
             'Consumer protection', 'Consumption statistics', 'Finance and loans', 'Gender', 'Healthy nutrition', 'Obesity', 'One health', 'Safety', 'School meals', 'SME', 
             'Fisheries & Aquaculture', 'Research and education']

x_ordered = ["North America", "Europe & Central Asia", 
             "East Asia & Pacific", "South Asia", 
             "Middle East & North Africa", "Sub-Saharan Africa", "Latin America & Caribbean"]
fig, ax = plot_heatmap(df, 'World Bank Region', 'topic_name', y_sort = 'topic_category',
                       fig = fig, ax = ax,
                       log_normalise= False,
                       y_axis= y_ordered, x_axis = x_ordered,
                       title = None#"Number of paragraphs per topic category and region - normalised by column"
                       )

#Add lines around the groups of topics
for y in [13, 22, 31, 38, 48]:
    plt.axhline(y, lw = 2, c = 'k')
#And add them as text 
#plt.subplots_adjust(left=0.5)
plt.text(7.1, 6.5, "General & Governance", va = 'center', rotation=270, weight='bold')
plt.text(7.1, 17.5, "Agriculture", va = 'center', rotation=270, weight='bold')
plt.text(7.1, 26.5, "Food Insecurity", va = 'center', rotation=270, weight='bold')
plt.text(7.1, 34.5, "Environment", va = 'center', rotation=270, weight='bold')
plt.text(7.1, 43, "Health & Social", va = 'center', rotation=270, weight='bold')
plt.text(7.1, 49, "Other", va = 'center', rotation=270, weight='bold')
   
#fig.tight_layout()
fig.show()

#%% All topics - income

fig, ax = plt.subplots(1,1, figsize = (9, 14), dpi=300)


x_ordered = ['High', 'Upper-middle', 'Lower-middle', 'Low']

fig, ax = plot_heatmap(df, 'World Bank Income', 'topic_name', y_sort = 'topic_category',
                       fig = fig, ax = ax,
                       y_axis= y_ordered, x_axis = x_ordered,
                       title = "Number of paragraphs per topic category and income group - normalised by column")


#Add lines around the groups of topics
for y in [13, 22, 31, 38, 48]:
    plt.axhline(y, lw = 2, c = 'k')
#And add them as text 
#plt.subplots_adjust(left=0.5)
plt.text(4.1, 6.5, "General & Governance", va = 'center', rotation=270, weight='bold')
plt.text(4.1, 17.5, "Agriculture", va = 'center', rotation=270, weight='bold')
plt.text(4.1, 26.5, "Food Insecurity", va = 'center', rotation=270, weight='bold')
plt.text(4.1, 34.5, "Environment", va = 'center', rotation=270, weight='bold')
plt.text(4.1, 43, "Health & Social", va = 'center', rotation=270, weight='bold')
plt.text(4.1, 49, "Other", va = 'center', rotation=270, weight='bold')
    
fig.tight_layout()
fig.show()


#%% Plot fully agregated





# #%% Plot continental
# #Plot on a white background
# cmap = colormaps.get_cmap('viridis_r')
# cmap.set_bad("w")


# df_counts_continental  = counts_for_heatmap(df, 'UN continental', 'topic_name')
# plt.figure(figsize=(10,15), dpi = 200)
# ax = sns.heatmap(df_counts_continental, annot=False, fmt=".1f", linewidth=.01,
#                  norm=LogNorm(), cmap = cmap)
# plt.tight_layout()

# #%% Plot worldbank region
# #Plot on a white background
# cmap = colormaps.get_cmap('viridis_r')
# cmap.set_bad("w")

# df_counts_continental  = counts_for_heatmap(df, 'World Bank Region', 'topic_name')
# # df_norm_continental = 100*df_counts_continental/df_counts_continental.sum() #col
# # bar_ticks = [2, 5, 10, 25]
# df_norm_continental = 100*df_counts_continental.divide(df_counts_continental.sum(axis = 1), axis = 0) #row
# bar_ticks = [2, 5, 15, 45, 90]
# plt.figure(figsize=(10,15), dpi = 200)
# ax = sns.heatmap(df_norm_continental, annot=df_counts_continental, linewidth=.01,
#                  norm=LogNorm(),
#                  cmap = cmap, cbar_kws={"shrink": 0.5})
# cbar = ax.collections[0].colorbar

# cbar.set_ticks(bar_ticks)
# cbar.set_ticklabels([f"{t}%" for t in bar_ticks])
# ax.set_title("Normalised by row (topic)")
# plt.tight_layout()



# #%% Plot worldbank region
# #Plot on a white background
# cmap = colormaps.get_cmap('viridis_r')
# cmap.set_bad("w")

# df_counts_continental  = counts_for_heatmap(df, 'UN statistical', 'topic_name')
# # df_norm_continental = 100*df_counts_continental/df_counts_continental.sum() #col
# # bar_ticks = [2, 5, 10, 25]
# df_norm_continental = 100*df_counts_continental.divide(df_counts_continental.sum(axis = 1), axis = 0) #row
# bar_ticks = [2, 5, 15, 45, 90]
# plt.figure(figsize=(10,15), dpi = 200)
# ax = sns.heatmap(df_norm_continental, annot=df_counts_continental, linewidth=.01,
#                  norm=LogNorm(),
#                  cmap = cmap, cbar_kws={"shrink": 0.5})
# cbar = ax.collections[0].colorbar

# cbar.set_ticks(bar_ticks)
# cbar.set_ticklabels([f"{t}%" for t in bar_ticks])
# ax.set_title("Normalised by row (topic)")
# plt.tight_layout()

