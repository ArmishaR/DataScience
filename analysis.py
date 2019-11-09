import pandas as pd
import seaborn as sns

#Metadata Headers: 'site_id', 'building_id', 'primary_use', 'square_feet',
#'year_built', 'floor_count'
metadata = pd.read_csv('~/Desktop/Data Science/project/ashrae-energy-prediction/DataScience/building_metadata.csv')

#Weather Headers: 'site_id', 'timestamp', 'air_temperature', 'cloud_coverage',
#'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed'
weather = pd.read_csv('~/Desktop/Data Science/project/ashrae-energy-prediction/DataScience/weather_train.csv')

metadata.info()
weather.info()

#%%
#Setting weather into datetime format
weather_time     = []
for x in weather['timestamp']:
    weather_time.append(pd.to_datetime(x))
weather['datetime'] = weather_time
#%% didn't use!
#Weather File Splitting by Hour
byHour = weather.groupby(lambda x: weather['datetime'][x].hour)
#print(byHour.size())

#Removing 'site_id' and 'timestamp'
byHour = byHour[['air_temperature', 'cloud_coverage','dew_temperature', 'precip_depth_1_hr',
                 'sea_level_pressure', 'wind_direction', 'wind_speed']]

#%%
#Adding Hour and Month column
hours = []
for x in weather['datetime']:
    hours.append(x.hour)

months = []
for x in weather['datetime']:
    months.append(x.month)

weather['hour'] = hours
weather['month'] = months

########################### BOXPLOTS ########################################
#%%
#Box Plots By Hour for Weather
weather.boxplot(by = 'hour', figsize=(8,8), column= 'air_temperature')
weather.boxplot(by = 'hour', figsize=(8,8), column= 'cloud_coverage')
weather.boxplot(by = 'hour', figsize=(8,8), column= 'dew_temperature')
weather.boxplot(by = 'hour', figsize=(8,8), column= 'precip_depth_1_hr')
weather.boxplot(by = 'hour', figsize=(8,8), column= 'sea_level_pressure')
weather.boxplot(by = 'hour', figsize=(8,8), column= 'wind_direction')
weather.boxplot(by = 'hour', figsize=(8,8), column= 'wind_speed')

#%%
#Box Plot by Month for Weather
weather.boxplot(by = 'month', figsize=(8,8), column= 'air_temperature')
weather.boxplot(by = 'month', figsize=(8,8), column= 'cloud_coverage')
weather.boxplot(by = 'month', figsize=(8,8), column= 'dew_temperature')
weather.boxplot(by = 'month', figsize=(8,8), column= 'precip_depth_1_hr')
weather.boxplot(by = 'month', figsize=(8,8), column= 'sea_level_pressure')
weather.boxplot(by = 'month', figsize=(8,8), column= 'wind_direction')
weather.boxplot(by = 'month', figsize=(8,8), column= 'wind_speed')

#%%
#Box Plot by Site for Weather
weather.boxplot(by = 'site_id', figsize=(8,8), column= 'air_temperature')
weather.boxplot(by = 'site_id', figsize=(8,8), column= 'cloud_coverage')
weather.boxplot(by = 'site_id', figsize=(8,8), column= 'dew_temperature')
weather.boxplot(by = 'site_id', figsize=(8,8), column= 'precip_depth_1_hr')
weather.boxplot(by = 'site_id', figsize=(8,8), column= 'sea_level_pressure')
weather.boxplot(by = 'site_id', figsize=(8,8), column= 'wind_direction')
weather.boxplot(by = 'site_id', figsize=(8,8), column= 'wind_speed')

#%%
#Box Plot by site for Metadata
metadata.boxplot(by = 'site_id', figsize=(8,8), column= 'square_feet')
metadata.boxplot(by = 'site_id', figsize=(8,8), column= 'year_built')
metadata.boxplot(by = 'site_id', figsize=(8,8), column= 'floor_count')

#%%
#Box Plot by Primary Use for Metadata
metadata.boxplot(by = 'primary_use', figsize=(10,10), rot = 90, column= 'square_feet')
metadata.boxplot(by = 'primary_use', figsize=(10,10), rot = 90, column= 'year_built')
metadata.boxplot(by = 'primary_use', figsize=(10,10), rot = 90, column= 'floor_count')
metadata.boxplot(by = 'primary_use', figsize=(10,10), rot = 90, column= 'site_id')
############################ BAR CHART ###############################################

#%%
#Bar Chart of Primary Use
byUse = metadata.groupby('primary_use')
byUse.size().plot(kind='bar')

############################## HISTOGRAMS ########################################
#%% DEFAULT BINS ARE 10
#Histogram by Primary Use for Metadata
metadata.hist(by = 'primary_use', figsize=(10,10),  column= 'square_feet')
metadata.hist(by = 'primary_use', figsize=(10,10),  column= 'year_built')
metadata.hist(by = 'primary_use', figsize=(10,10),  column= 'floor_count')
metadata.hist(by = 'primary_use', figsize=(10,10),  column= 'site_id')

#%%
#Histogram by site for Metadata
metadata.hist(by = 'site_id', figsize=(10,10), column= 'square_feet', bins = 15)
metadata.hist(by = 'site_id', figsize=(10,10), column= 'year_built', bins = 20)
metadata.hist(by = 'site_id', figsize=(10,10), column= 'floor_count', bins = 15)
#%%
#Histogram by hour for Weather
weather.hist(by = 'hour', figsize=(10,10), xrot = 90, column= 'air_temperature', bins=20)
weather.hist(by = 'hour', figsize=(10,10), xrot = 90, column= 'cloud_coverage')
weather.hist(by = 'hour', figsize=(10,10), xrot = 90, column= 'dew_temperature', bins = 20)
weather.hist(by = 'hour', figsize=(10,10), xrot = 90, column= 'precip_depth_1_hr', bins = 15)
weather.hist(by = 'hour', figsize=(10,10), xrot = 90, column= 'sea_level_pressure', bins = 15)
weather.hist(by = 'hour', figsize=(10,10), xrot = 90, column= 'wind_direction', bins = 30)
weather.hist(by = 'hour', figsize=(10,10), xrot = 90, column= 'wind_speed', bins = 20)

#%%
#Histogram by Month for weather
weather.hist(by = 'month', figsize=(10,10), xrot = 90, column= 'air_temperature')
weather.hist(by = 'month', figsize=(10,10), xrot = 90, column= 'cloud_coverage')
weather.hist(by = 'month', figsize=(10,10), xrot = 90, column= 'dew_temperature')
weather.hist(by = 'month', figsize=(10,10), xrot = 90, column= 'precip_depth_1_hr')
weather.hist(by = 'month', figsize=(10,10), xrot = 90, column= 'sea_level_pressure')
weather.hist(by = 'month', figsize=(10,10), xrot = 90, column= 'wind_direction')
weather.hist(by = 'month', figsize=(10,10), xrot = 90, column= 'wind_speed')

#%%
#Histogram by Site for Weather
weather.hist(by = 'site_id', figsize=(10,10), xrot = 90, column= 'air_temperature', bins = 15)
weather.hist(by = 'site_id', figsize=(10,10), xrot = 90, column= 'cloud_coverage')
weather.hist(by = 'site_id', figsize=(10,10), xrot = 90, column= 'dew_temperature', bins = 20)
weather.hist(by = 'site_id', figsize=(10,10), xrot = 90, column= 'precip_depth_1_hr')
weather.hist(by = 'site_id', figsize=(10,10), xrot = 90, column= 'sea_level_pressure', bins = 20)
weather.hist(by = 'site_id', figsize=(10,10), xrot = 90, column= 'wind_direction', bins = 15)
weather.hist(by = 'site_id', figsize=(10,10), xrot = 90, column= 'wind_speed', bins = 15)

############################## JITTER PLOT ################################################
#%%
#Metadata features
sns.set(style="whitegrid")
sns.stripplot(x = metadata['year_built'])
sns.stripplot(x = metadata['square_feet'])
sns.stripplot(x = metadata['floor_count'])
#%%
#Weather Features
sns.stripplot(x = weather['air_temperature'])
sns.stripplot(x = weather['cloud_coverage'])
sns.stripplot(x = weather['dew_temperature'])
sns.stripplot(x = weather['precip_depth_1_hr'])
sns.stripplot(x = weather['sea_level_pressure'])
sns.stripplot(x = weather['wind_direction'])
sns.stripplot(x = weather['wind_speed'])

#################################### CORRELATION ##########################################
#%%
#Metadata Features
metadata.corr(method = 'pearson').to_csv("/Users/Armisha/Desktop/DS_Images/correlations/Metadata_pearson.csv")
metadata.corr(method = 'kendall').to_csv("/Users/Armisha/Desktop/DS_Images/correlations/Metadata_kendall.csv")
metadata.corr(method = 'spearman').to_csv("/Users/Armisha/Desktop/DS_Images/correlations/Metadata_spearman.csv")

#Weather Features
weather.corr(method = 'pearson').to_csv("/Users/Armisha/Desktop/DS_Images/correlations/Weather_pearson.csv")
weather.corr(method = 'kendall').to_csv("/Users/Armisha/Desktop/DS_Images/correlations/Weather_kendall.csv")
weather.corr(method = 'spearman').to_csv("/Users/Armisha/Desktop/DS_Images/correlations/Weather_spearman.csv")
