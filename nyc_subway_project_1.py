import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm

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
	
def histograms_entries_hourly_rain_norain_2_subplots(data):
	'''
	Return a plot containing 2 histogram subplots of entries 
	hourly for sample rain and no-rain.
	The x-axis is clipped at 6000.
	'''
	
	rain_df = data[data['rain'] == 1]
	noRain_df = data[data['rain'] == 0]
	fig = plt.figure() 
	ax1 = plt.subplot(2,1,1)
	rain_df['ENTRIESn_hourly'].hist(bins=250, color='red')
	plt.xlim(0,6000)
	red_patch = mpatches.Patch(color='red', label='Rain')
	plt.legend(handles=[red_patch])
	plt.title('Histrogram of Turnstile Entries Hourly', fontsize=20)

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
	prediction = est.predict(X)
	
	return est, prediction 
