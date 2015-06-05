import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
from scipy import stats
import statsmodels.api as sm
from datetime import datetime 

path = r'D:\Toms\BigData\UdaCity\IntroToDataScience' + '\\'

def nyc_trunstile_data_with_weather():
	'''
	Return the original dataset as a DataFrame
	'''
	return pd.read_csv(path + 'turnstile_data_master_with_weather.csv')
    
def nyc_trunstile_V2():
	'''
	Return the enhanced dataset as a DataFrame
	'''	
	
	return pd.read_csv(path + 'turnstile_weather_v2.csv')

def add_date_support_columns(data):
	data['day_week'] = data['DATEn'].map(lambda x: datetime.strptime(x, '%m/%d/%Y').weekday())
	data['holiday'] = data['DATEn'].map(lambda x: int(x == '5/30/2011'))
	data['weekday'] = data['day_week'].map(lambda x: int(x > 4))
	return
	
def workdays_from(data):
	'''
	This function uses the turnstile_weather data frame and returns the following tuple 
		1. The mean of the rain entries
		2. The mean of the no rain entries
		3. the Mann-Whitney U-Statistic and P-value for the rain and no-rain samples
	'''
	workdays = data[(data['holiday'] == 0) &  (data['weekday'] == 0)]
	
	return workdays

def weekends_from(data):
	'''
	This function uses the turnstile_weather data frame and returns the following tuple 
		1. The mean of the rain entries
		2. The mean of the no rain entries
		3. the Mann-Whitney U-Statistic and P-value for the rain and no-rain samples
	'''
	weekends = data[(data['holiday'] == 0) &  (data['weekday'] == 1)]
	
	return weekends

def holidays_from(data):
	'''
	This function uses the turnstile_weather data frame and returns the following tuple 
		1. The mean of the rain entries
		2. The mean of the no rain entries
		3. the Mann-Whitney U-Statistic and P-value for the rain and no-rain samples
	'''
	holidays = data[(data['holiday'] == 1)]
	
	return holidays
	
def print_stats_for(data, title):
	print('Stats: ' + title)
	set = data['ENTRIESn_hourly']
	mean = np.mean(set)
	var = np.var(set)
	count = len(set)
	std = np.std(set)
	print '\tBase' + '\n\t\tcount: ' + str(count) + '\n\t\tmean: ' + str(mean) + '\n\t\tvar: ' + str(var), '\n\t\tstd: ' + str(std)
	set = data[data['rain'] == 0]['ENTRIESn_hourly']
	mean = np.mean(set)
	var = np.var(set)
	count = len(set)
	std = np.std(set)
	print '\tno rain' + '\n\t\tcount: ' + str(count) + '\n\t\tmean: ' + str(mean) + '\n\t\tvar: ' + str(var), '\n\t\tstd: ' + str(std)
	set = data[data['rain'] == 1]['ENTRIESn_hourly']
	mean = np.mean(set)
	var = np.var(set)
	count = len(set)
	std = np.std(set)
	print '\train' + '\n\t\tcount: ' + str(count) + '\n\t\tmean: ' + str(mean) + '\n\t\tvar: ' + str(var), '\n\t\tstd: ' + str(std)
	
	return
	
def rain_work_day_no_holiday_from(data):
	'''
	This function uses the turnstile_weather data frame and returns the following tuple 
		1. The mean of the rain entries
		2. The mean of the no rain entries
		3. the Mann-Whitney U-Statistic and P-value for the rain and no-rain samples
	'''
	rain = data[(data['holiday'] == 0) &  (data['weekday'] == 0) & (data['rain'] == 1) ]
	
	return rain

def norain_work_day_no_holiday_from(data):
	'''
	This function uses the turnstile_weather data frame and returns the following tuple 
		1. The mean of the rain entries
		2. The mean of the no rain entries
		3. the Mann-Whitney U-Statistic and P-value for the rain and no-rain samples
	'''
	rain = data[(data['holiday'] == 0) &  (data['weekday'] == 0) & (data['rain'] == 0) ]
	
	return rain
		
	

		
def scatter_plot_ridership_by_day(data):
	'''
	Return a scatter plot of Ridership by Day  
	It will highlight the memorial holiday 
	The legend could not be added because was obscuring the 
	Friday data points.
	'''
	
	grp = data.groupby(['day_week', 'DATEn'])['ENTRIESn_hourly']
	entriesByDayWithOutHoliday = {}
	entriesByDay = {}
	holidaysByDay = {}	
	holidays = []
	holidaysDays = []
	entries = []
	entryDays = []
	for (k1, k2), values in grp:
		if k2 != '05-30-11':
			entriesByDayWithOutHoliday.setdefault(k1, []).append(np.sum(values))
		entriesByDay.setdefault(k1, []).append(np.sum(values))
		if k2 == '05-30-11':
			holidaysByDay.setdefault(k1, []).append(np.sum(values))
		holidaysByDay.setdefault(k1, [])
		#print(k1, k2, k2 == '05-30-11', '  ', len(values), '  ', np.sum(values), '  ', np.mean(values))
	#print(.setdefault(k1, []).append[np.sum(values)])
	meanRidershipByDay = []
	days = [0 ,1, 2, 3, 4, 5, 6]
	labels = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturdary', 'Sunday']
	labels2 = []
	holidayLabels = []
	for day in days:
		meanRidershipByDay.append(np.mean(entriesByDayWithOutHoliday[day]))
		dayEntries = entriesByDayWithOutHoliday[day]
		for dayEntry in dayEntries:
			entries.append(dayEntry)
			entryDays.append(day)
			labels2.append(labels[day])
			
		holidayDayEntries = holidaysByDay[day]
		if len(holidayDayEntries) == 0:
			holidays.append(None)
			holidaysDays.append(day)
			holidayLabels.append(labels[day])
		for dayEntry in holidayDayEntries:
			holidays.append(dayEntry)
			holidaysDays.append(day)
			holidayLabels.append(labels[day])
	
	#print(entryDays)		
	#print(entries)
	
	plot = plt.figure() 
	plt.scatter(entryDays, entries, alpha=0.5, s=10)
	plt.title("Daily Ridership by Week Day", fontsize=20)
	plt.xticks(entryDays, labels2, rotation='vertical')
	plt.scatter(holidaysDays, holidays, alpha=0.5, color='red', s=10)
	# ytikcs
	locs,labels = plt.yticks()
	plt.yticks(locs, map(lambda x: "%.1f" % x, locs/1e6))
	plt.ylabel('Entries (millions)')
	plt.margins(0.2)
	plt.subplots_adjust(bottom=0.25)
	return plt

def histograms_entries_hourly_rain_norain_for_all(data):
	title = 'Histogram of Turnstile Entries Hourly'
	return histograms_entries_hourly_rain_norain_2_subplots(data, title)

def histograms_entries_hourly_rain_norain_for_workdays(data):
	title = 'Histogram of Workday Turnstile Entries Hourly'
	return histograms_entries_hourly_rain_norain_2_subplots(data, title)

def histograms_entries_hourly_rain_norain_for_weekends(data):
	title = 'Histogram of Weekend Turnstile Entries Hourly'
	return histograms_entries_hourly_rain_norain_2_subplots(data, title)

def histograms_entries_hourly_rain_norain_for_holidays(data):
	title = 'Histogram of Holiday Turnstile Entries Hourly'
	return histograms_entries_hourly_rain_for(data, title)

	
def histograms_entries_hourly_rain_norain_2_subplots(data, title):
	'''
	Return a plot containing 2 histogram subplots of entries 
	hourly for sample rain and no-rain.
	The x-axis is clipped at 6000.
	Typical titles 
	Histogram of Turnstile Entries Hourly
	Histogram of Work Day Turnstile Entries Hourly
	Histogram of Weekend Turnstile Entries Hourly
	Histogram of Holiday Turnstile Entries Hourly
	'''
	
	rain_df = data[data['rain'] == 1]
	noRain_df = data[data['rain'] == 0]
	fig = plt.figure() 
	ax1 = plt.subplot(2,1,1)
	rain_df['ENTRIESn_hourly'].hist(bins=250, color='red')
	plt.xlim(0,6000)
	red_patch = mpatches.Patch(color='red', label='Rain')
	plt.legend(handles=[red_patch])
	plt.title(title, fontsize=20)

	ax2= plt.subplot(2,1,2)
	noRain_df['ENTRIESn_hourly'].hist(bins=250, color='blue')
	#plt.ylim(0,15000)
	plt.xlim(0,6000)
	#ax.ylabel('Frequency')
	plt.xlabel('Entries Hourly', fontsize= 16)
	fig.text(0.03, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize=16)
	#ax2.set_title("No Rain")
	blue_patch = mpatches.Patch(color='blue', label='No Rain')
	plt.legend(handles=[blue_patch])

	return plt
	
def histograms_entries_hourly_rain_for(data, title):
	'''
	Return a plot containing 2 histogram subplots of entries 
	hourly for sample rain and no-rain.
	The x-axis is clipped at 6000.
	Typical titles 
	Histogram of Turnstile Entries Hourly
	Histogram of Work Day Turnstile Entries Hourly
	Histogram of Weekend Turnstile Entries Hourly
	Histogram of Holiday Turnstile Entries Hourly
	'''
	
	rain_df = data[data['rain'] == 1]
	noRain_df = data[data['rain'] == 0]
	fig = plt.figure() 
	ax1 = plt.subplot(1,1,1)
	rain_df['ENTRIESn_hourly'].hist(bins=250, color='red')
	plt.xlim(0,6000)
	red_patch = mpatches.Patch(color='red', label='Rain')
	plt.legend(handles=[red_patch])
	plt.title(title, fontsize=20)
	plt.xlabel('Entries Hourly', fontsize= 16)
	return plt
	
def rain_norain_Mann_Whitney_U_statistic_for(data):
	'''
	This function uses the turnstile_weather data frame and returns the following tuple 
		1. The mean of the rain entries
		2. The mean of the no rain entries
		3. the Mann-Whitney U-Statistic and P-value for the rain and no-rain samples
	'''
	
	rain = data[data['rain'] == 1]['ENTRIESn_hourly']
	no_rain = data[data['rain'] == 0]['ENTRIESn_hourly']
	
	no_rain_mean = np.mean(no_rain)
	rain_mean = np.mean(rain)
	#print no_rain_mean, rain_mean
	stats = scipy.stats.mannwhitneyu(rain, no_rain)
	m_u = len(rain) *len(no_rain)/2.0
	sigma_u = np.sqrt(len(rain)*len(no_rain)* (len(rain) + len(no_rain) + 1)/ 12)
	u = stats[0]
	z = (u - m_u)/sigma_u
	pval = 2*scipy.stats.norm.cdf(z)
	return len(rain), rain_mean, len(no_rain), no_rain_mean, stats[0], stats[1], z
	
def rainVsNoRainMann_Whitney_U_statistic(trunstile_weather):
	'''
	This function uses the turnstile_weather data frame and returns the following tuple 
		1. The mean of the rain entries
		2. The mean of the no rain entries
		3. the Mann-Whitney U-Statistic and P-value for the rain and no-rain samples
	'''
	
	rain = trunstile_weather[trunstile_weather['rain'] == 1]['ENTRIESn_hourly']
	no_rain = trunstile_weather[trunstile_weather['rain'] == 0]['ENTRIESn_hourly']
	
	no_rain_mean = np.mean(no_rain)
	rain_mean = np.mean(rain)
	#print no_rain_mean, rain_mean
	stats = scipy.stats.mannwhitneyu(rain, no_rain)
	m_u = len(rain) *len(no_rain)/2.0
	sigma_u = np.sqrt(len(rain)*len(no_rain)* (len(rain) + len(no_rain) + 1)/ 12)
	u = stats[0]
	z = (u - m_u)/sigma_u
	pval = 2*scipy.stats.norm.cdf(z)
	return rain_mean, no_rain_mean, stats[0], stats[1], z
	
	
def ols_estimate_prediction(weather_turnstile):
	'''
	Return a tuple of the OLS estimate and its prediction
	''' 
	
	weather_turnstile['holiday'] = weather_turnstile['DATEn'].map(lambda x: int(x == '05-30-11'))
	variables = ['rain', 'precipi', 'hour', 'meantempi', 'holiday', 'fog']

	features = weather_turnstile[variables]
	#print(features)
    # Add UNIT to features using dummy variables
	dummy_units = pd.get_dummies(weather_turnstile['UNIT'], prefix='unit')
	dummy_days = pd.get_dummies(weather_turnstile['day_week'], prefix='day')
	#print(dummy_units)
	features = features.join(dummy_units)
	features = features.join(dummy_days)
	#print(features)
	y = weather_turnstile['ENTRIESn_hourly']
	X = sm.add_constant(features)
 	est = sm.OLS(y, X)
	est = est.fit()
	#print(variables)
	#print(est.summary())
	predictions = est.predict(X)
	
	return est, predictions 
	
def histogram_residuals(turnstile_weather, predictions):
	'''
	Using the same methods that we used to plot a histogram of entries
	per hour for our data, why don't you make a histogram of the residuals
	(that is, the difference between the original hourly entry data and the predicted values).
	Try different binwidths for your histogram.

	Based on this residual histogram, do you have any insight into how our model
	performed?  Reading a bit on this webpage might be useful:

	http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
	'''
    
	plt.figure()
	ax1 = plt.subplot(1,1,1)
	x = turnstile_weather['ENTRIESn_hourly'] - predictions
	x.hist(bins=80, normed=True)
	ax1.set_title('Histogram of OLS Residuals', fontsize= 20)
	ax1.set_xlabel('Residual of Entries Hourly')
	ax1.set_ylabel('Proportion')
	return plt
	
def residuals_normal_probability_plot(turnstile_weather, predictions):
	'''
	Using the same methods that we used to plot a histogram of entries
	per hour for our data, why don't you make a histogram of the residuals
	(that is, the difference between the original hourly entry data and the predicted values).
	Try different binwidths for your histogram.
	Based on this residual histogram, do you have any insight into how our model
	performed?  Reading a bit on this webpage might be useful:

	http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
	'''
	plt.figure()
	ax1 = plt.subplot(1,1,1)
	x = turnstile_weather['ENTRIESn_hourly'] - predictions
	stats.probplot(x, plot=ax1)
	plt.title('OLS Residual Normal Probability Plot', fontsize=20)
	return plt
def residuals_vs_predictions_plot(turnstile_weather, predictions):
	'''
	Using the same methods that we used to plot a histogram of entries
	per hour for our data, why don't you make a histogram of the residuals
	(that is, the difference between the original hourly entry data and the predicted values).
	Try different binwidths for your histogram.
	Based on this residual histogram, do you have any insight into how our model
	performed?  Reading a bit on this webpage might be useful:

	http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
	'''
	
	x = turnstile_weather['ENTRIESn_hourly'] - predictions
	plt.figure() 
	plt.scatter(x, predictions)
	plt.title("OLS Residual vs Predictions", fontsize=20)
	
	return plt
