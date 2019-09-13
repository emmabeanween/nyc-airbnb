#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.patches as mpatches
from IPython.core.display import display, HTML
from scipy.stats import iqr
from colour import Color



path  = r"C:\Users\ejgoldbe\Downloads\airbnbnyc.csv"
airbnb_nyc = pd.read_csv(path)
display(airbnb_nyc.isna().sum())
airbnb_nyc = airbnb_nyc.drop(['host_name', 'name', 'last_review', 'reviews_per_month', 'id', 'host_id', 'latitude', 'longitude'], axis = 1)
airbnc_nyc = airbnb_nyc.dropna()
display(airbnb_nyc.head(5))
only_numeric_data = airbnb_nyc._get_numeric_data()
#a correlation plot and matrix of the important numerical variables
corr = only_numeric_data.corr()
display(corr)
plt.figure(figsize = (15, 5))
plt.matshow(corr)
range_cols = range(only_numeric_data.shape[1])
plt.xticks(range_cols, only_numeric_data.columns, rotation=40)
plt.yticks(range_cols, only_numeric_data.columns)
colorbar = plt.colorbar()
colorbar.ax.tick_params(labelsize=14)
plt.show()
#most variables are not correlated, although minimum_nights and availability have a small correlation - plot
plt.figure(figsize = (18, 5))
plt.scatter(airbnb_nyc.availability_365, airbnb_nyc.minimum_nights, color = 'coral')
plt.show()

#individual plots of numerical variables
colors = ['blue', 'green', 'yellow', 'purple', 'pink']
fig, axes = plt.subplots(1, 5, figsize = (20, 5))
for idx in range(only_numeric_data.shape[1]):
    col = only_numeric_data.iloc[:, idx]
    ninety_nine = np.percentile(col, 99)
    axes[idx].set_xlim([0, ninety_nine])
    axes[idx].hist(col, bins=40, color = colors[idx])
    axes[idx].set_title(only_numeric_data.columns[idx])
    
    
#what is our most common neighborhood?
neighborhood_counts = airbnb_nyc['neighbourhood_group'].value_counts().sort_values(ascending=True)
plt.figure(figsize = (16, 6))
plt.bar(neighborhood_counts.index, neighborhood_counts.values, color = 'blue')
plt.xlabel("Neighborhood")
plt.ylabel("Number of Listings")
plt.title("Neighborhood Distribution of NYC AirBNB Data")
plt.show()


#what is our distribution of sub neighborhoods?
sub_neighborhood_counts =  pd.crosstab( airbnb_nyc['neighbourhood_group'], airbnb_nyc['neighbourhood']).replace(0, np.nan)
sub_neighborhood_counts = sub_neighborhood_counts.stack().sort_values(ascending=False)[0:20]
#the 20 most popular sub neighborhoods within the 5 bouroughs for our dataset
fig, ax = plt.subplots(1, 1, figsize = (30, 10))
sub_unstacked = pd.DataFrame(sub_neighborhood_counts.unstack())
colors = ['blue', 'green', 'purple', 'pink', 'red', 'yellow', 'orange', 'black', 'gold', 'gray'
         , 'brown', 'cyan', 'olive', 'maroon', 'lavender', 'aqua', 'coral', 'skyblue', 'lime', 'tan']
sub_unstacked.plot(ax = ax, kind='bar', width=1, color = colors) 
plt.xlabel("Neighborhood")
plt.ylabel("Number of Listings")
plt.title("Most Popular Neighborhoods For AirBNB Listings, NYC")
plt.rcParams.update({'font.size': 10})
plt.legend(loc='best', prop={'size': 15})
plt.show()


#are rooms in certain neighborhoods pricier?
grouped_prices = airbnb_nyc.groupby([ 'neighbourhood']).agg({ 'price': ['size', 'mean'] })
only_size = grouped_prices.loc[:, pd.IndexSlice[:, 'size']]
only_size = only_size[only_size.values >= 10]
#extract the neighbourhoods with greater than or equal to 10 listings, in order to accurately assess the priciest ones
neighbourhoods_greater_ten = only_size.index.tolist()
grouped_prices = grouped_prices.loc[neighbourhoods_greater_ten]
grouped_prices = grouped_prices.sort_values(by = ('price', 'mean'), ascending=False)[0:20]
#extract only the price column to plot
grouped_prices = pd.DataFrame(grouped_prices.xs('mean', level=1, axis=1))
fig, ax = plt.subplots(1, 1, figsize = (15, 5))
grouped_prices.plot(kind = 'barh', ax=ax)
ax.set_xlabel("Mean Price")
ax.set_ylabel("Neighbourhood")
ax.set_title("NYC Neighbourhoods with Highest Average AirBNB listing price")


#most common room type?
plt.figure(figsize = (18, 5))
grouped_type = airbnb_nyc.groupby(['room_type']).size().plot(kind = 'bar', color = ['lightgreen', 'forestgreen', 'seagreen'])
plt.xlabel("Room Type")
plt.ylabel("Number Listings")
#does a certain room type get more reviews?
grouped_type_reviews = airbnb_nyc.groupby(['room_type']).agg({'minimum_nights': ['max', 'min', 'mean']})
display(grouped_type_reviews)


#do certain neighbourhoods get reviewed more or have more availability?
grouped_neighbourhoods = airbnb_nyc.groupby(['neighbourhood_group']).agg({'number_of_reviews': ['mean', 'max']
                                                            , 'availability_365': ['mean']})
grouped_neighbourhoods.plot(kind = 'bar', color = ['blue', 'green', 'red'])
mean_reviews = mpatches.Patch(color='blue', label='Mean Number of Reviews')
max_reviews = mpatches.Patch(color='green', label='Max Number of Reviews')
mean_availability = mpatches.Patch(color='red', label='Mean Availability Days')
plt.legend(handles=[mean_reviews, max_reviews, mean_availability], loc='best')
plt.xlabel("Bourough")
plt.ylabel("Number")
plt.show()



# In[ ]:




