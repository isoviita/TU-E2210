#load libraries
import pandas as pd
import numpy as np
import scipy.stats as si
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

make_figures = False
data = None
#function for data loading

def loadData(filename, sheet_index):
	global data
	xls = pd.ExcelFile(filename)
	data = xls.parse(xls.sheet_names[sheet_index])
	data.columns = ['TTM', *data.columns[1:-3], 'SP', 'r', 'obs_date']
	data.set_index(list(data)[0], inplace=True)
	data.head()

#Find implied volatility using Newton's method
def impliedVolatility(C, S, K, T, r, max_error, max_iterations):
	sigma = 0.5
	for i in range(0, max_iterations):
		d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
		C1 = euroCallBS(S, K, T, r, sigma)
		vega = S * np.sqrt(T) * si.norm.cdf(d1, 0.0, 1.0)
		diff = C - C1
		if (abs(diff) < max_error):
			return sigma
		sigma = sigma + diff / vega
	return sigma

#Black-Scholes for an non-dividend paying European call option
def euroCallBS(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call

#returns mean squared error for delta hedging in given worksheet, TTM, hedging frequency and strike price.
def deltaHedge(time_to_maturity, hedging_frequency, strike_price):

	K = strike_price
	subset = data.loc[time_to_maturity:, [strike_price, 'SP', 'r']]

	#initialize in-function variables
	first_day = True
	A, OP, RE, deltas = [], [], [], []
	days_to_hedge, i, last_delta, last_delta = 0, 0, 0, 0

	for index, row in subset.iterrows():
		S = row['SP']
		C = row[strike_price]
		T = index/365
		r = row['r']/1000

		# calculate new delta, if days to hedge are 0:
		if days_to_hedge == 0:
			# print ('Hedge!')
			days_to_hedge = hedging_frequency
			sigma = impliedVolatility(C, S, strike_price, T, r, 0.001, 100)
			d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
			delta = si.norm.cdf(d1, 0.0, 1.0)
			last_delta = delta
			last_sigma = sigma

		#else, use last calculated values
		else:
			sigma = last_sigma
			delta = last_delta

		OP.append(C)
		RE.append(S * delta)
		deltas.append(delta)

		#compute Ai, if not in first day
		if first_day:
			first_day = False
		else:
			diff_A = OP[i] - OP[i-1] - (RE[i] - RE[i-1])
			A.append(diff_A)

		#increase/decrease counters
		days_to_hedge = days_to_hedge - 1
		i = i+1

	return np.nanmean(np.square(A)), np.nanstd(np.square(A))


def deltaHedgePortfolio(time_to_maturity, hedging_frequency, strike_prices):
	As = {}
	for strike_price in strike_prices:
		As[str(strike_price)] = []
	for strike_price in strike_prices:
		K = strike_price
		subset = data.loc[time_to_maturity:, [strike_price, 'SP', 'r']]

		#initialize in-function variables
		first_day = True
		A, OP, RE, deltas = [], [], [], []
		days_to_hedge, i, last_delta, last_delta = 0, 0, 0, 0

		for index, row in subset.iterrows():
			S = row['SP']
			C = row[strike_price]
			T = index/365
			r = row['r']/1000

			# calculate new delta, if days to hedge are 0:
			if days_to_hedge == 0:
				# print ('Hedge!')
				days_to_hedge = hedging_frequency
				sigma = impliedVolatility(C, S, strike_price, T, r, 0.001, 100)
				d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
				delta = si.norm.cdf(d1, 0.0, 1.0)
				last_delta = delta
				last_sigma = sigma

			#else, use last calculated values
			else:
				sigma = last_sigma
				delta = last_delta

			OP.append(C)
			RE.append(S * delta)
			deltas.append(delta)

			#compute Ai, if not in first day
			if first_day:
				first_day = False
			else:
				diff_A = OP[i] - OP[i-1] - (RE[i] - RE[i-1])
				A.append(diff_A)
				As[str(strike_price)].append(diff_A)

			#increase/decrease counters
			days_to_hedge = days_to_hedge - 1
			i = i+1
	totalAs = As[str(strike_prices[0])]
	for j in range(1, len(strike_prices)):
		totalAs = np.add(totalAs, As[str(strike_prices[j])])
	return np.nanmean(np.square(totalAs)), np.nanstd(np.square(totalAs))

def deltaVegaHedge(T1, T2, hedging_frequency):

	#load replicating option series:
	loadData('isx2010C.xls', T2[2])
	replicating = data.loc[T2[0]:, T2[1]]

	#load original option series:
	loadData('isx2010C.xls', T1[2])
	original = data.loc[T1[0]:, [T1[1], 'SP', 'r']]

	#initialize in-function variables
	first_day = True
	K = T1[1]
	A, stocks, options, replicatings = [0], [], [], []
	days_to_hedge, i, last_delta_bs, last_kappa_bs, last_sigma, last_delta_rep, last_kappa_rep = 0, 0, 0, 0, 0, 0, 0
	alphas, etas = [], []
	for index, row in original.iterrows():
		S = row['SP']
		C = row[K]
		T = index/365
		r = row['r']/1000

		C_rep = replicating.iloc[i]

		# calculate new parameters if days to hedge are 0:
		if days_to_hedge == 0:
			days_to_hedge = hedging_frequency
			sigma = impliedVolatility(C, S, K, T, r, 0.001, 100)
			sigma_rep = impliedVolatility(C_rep, S, K, T, r, 0.001, 100)
			d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
			d1_rep = (np.log(S / K) + (r + 0.5 * sigma_rep ** 2) * T) / (sigma * np.sqrt(T))
			
			delta_bs = si.norm.cdf(d1, 0.0, 1.0)
			kappa_bs = S * np.sqrt(T) * si.norm.pdf(d1, 0.0, 1.0)

			delta_rep = si.norm.cdf(d1_rep, 0.0, 1.0)
			kappa_rep = S * np.sqrt(T) * si.norm.pdf(d1_rep, 0.0, 1.0)

			last_delta_bs, last_delta_rep = delta_bs, delta_rep
			last_sigma = sigma
			last_kappa_bs, last_kappa_rep = kappa_bs, kappa_rep

		#else, use last calculated values
		else:
			sigma = last_sigma
			delta_bs, delta_rep = last_delta_bs, last_delta_rep
			kappa_bs, kappa_rep = last_kappa_bs, last_kappa_rep

		alpha = -1 * delta_bs + ( kappa_bs / kappa_rep ) * delta_rep
		eta = -1 * kappa_bs / kappa_rep
		alphas.append(alpha)
		etas.append(eta)

		stocks.append(alpha * S)
		options.append(C)
		replicatings.append(eta * C_rep)

		#compute Ai, if not in first day
		if first_day:
			first_day = False
		else:
			diff_A = options[i] - options[i-1] - (stocks[i] - stocks[i-1]) - (replicatings[i] - replicatings[i-1])
			A.append(diff_A)

		#increase/decrease counters
		days_to_hedge = days_to_hedge - 1
		i = i+1

	return np.nanmean(np.square(A)), np.nanstd(np.square(A)), alphas, etas

loadData('isx2010C.xls', 0)

if make_figures:
	#create fig 1
	hfs, avg, std = [], [], []
	for hf in range(1,31):
		a, s = deltaHedge(42, hf, 510)
		avg.append(a)
		std.append(s)
		hfs.append(hf)

	xi = list(range(len(hfs)))
	plt.figure(1)
	plt.subplot(211)
	plt.plot(avg)
	plt.title('Average and SD of E in single 42-day ATM option delta hedging')
	plt.ylabel('Avg of E ($)')
	plt.subplot(212)
	plt.plot(hfs, std)
	plt.ylabel('SD of E ($)')
	plt.xlabel('Hedging frequecy')
	plt.xticks(xi, hfs)
	plt.savefig('fig-1.eps', dpi=None, facecolor='w', edgecolor='w',
	        orientation='portrait', papertype=None, format=None,
	        transparent=False, bbox_inches=None, pad_inches=0.1,
	        metadata=None)

	#create figure 2
	ttms = [42,41,40,39,38]
	std_scatters = {}
	avg_scatters = {}
	avgs = []
	hfs = []
	for t in ttms:
		avg_scatters[str(t)] = []
		std_scatters[str(t)] = []
	for hf in range(1,31):
		avgs.append(0)
		for t in ttms:
			a, s = deltaHedge(t, hf, 510)
			avg_scatters[str(t)].append(a)
			std_scatters[str(t)].append(s)
			avgs[hf-1] = avgs[hf-1]+a/len(ttms)
		hfs.append(hf)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for t in ttms:
		ax.scatter(hfs, avg_scatters[str(t)], label='TTM: '+str(t))
	avr = fig.add_subplot(111)
	avr.plot(avgs)
	plt.title('Average and of error in single option delta hedging with various TTMs')
	plt.ylabel('Avg of E ($)')
	plt.xlabel('Hedging frequecy')
	plt.legend(loc='upper left');
	plt.savefig('fig-2.eps', dpi=None, facecolor='w', edgecolor='w',
	        orientation='portrait', papertype=None, format=None,
	        transparent=False, bbox_inches=None, pad_inches=0.1,
	        metadata=None)

	ttm = 30
	ts = 12 #nro of sheets
	#for each sheet, calculate atm value at ttm
	avgs, avg_scatters, std_scatters = {}, {}, {}
	hfs = [1,2,3,4,5,6,7,8,9,10,12,15,20]
	for t in range(0,ts):
		avg_scatters[str(t)] = []
		std_scatters[str(t)] = []
		
	for hf in hfs:
		avgs[str(hf)] = 0
		for t in range(0,ts):
			try:
				loadData('isx2010C.xls', t)
				atm = int(round(data.loc[ttm,'SP']/5)*5)
				print(ttm, hf, atm)
				a, s = deltaHedge(ttm, hf, atm)
				avg_scatters[str(t)].append(a)
				std_scatters[str(t)].append(s)
				avgs[str(hf)] = avgs[str(hf)]+a/ts
			except:
				print('Failure')

	fig = plt.figure()
	ax = fig.add_subplot(111)
	for t in range(0,ts):
		ax.scatter(hfs, avg_scatters[str(t)], c='black', label='TTM: '+str(t))
	avr = fig.add_subplot(111)
	arr = list(avgs.values())
	avr.plot(hfs, arr)
	plt.title('Average of error in single option delta hedging with TTM='+str(ttm))
	plt.ylabel('Avg of E ($)')
	plt.xlabel('Hedging frequecy')
	plt.savefig('fig-3.eps', dpi=None, facecolor='w', edgecolor='w',
	        orientation='portrait', papertype=None, format=None,
	        transparent=False, bbox_inches=None, pad_inches=0.1,
	        metadata=None)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	for t in range(0,ts):
		ax.scatter(hfs, std_scatters[str(t)], c='black', label='TTM: '+str(t))
	avr = fig.add_subplot(111)
	arr = list(avgs.values())
	avr.plot(hfs, arr)
	plt.title('SD of error in single option delta hedging with TTM='+str(ttm))
	plt.ylabel('SD of E ($)')
	plt.xlabel('Hedging frequecy')
	plt.savefig('fig-4.eps', dpi=None, facecolor='w', edgecolor='w',
	        orientation='portrait', papertype=None, format=None,
	        transparent=False, bbox_inches=None, pad_inches=0.1,
	        metadata=None)

	loadData('isx2010C.xls', 9)
	ttm = 34
	K = 510

	ATM = { 'std': [], 'avg' : []}
	ATM5 = { 'std': [], 'avg' : []}
	ATM10 = { 'std': [], 'avg' : []}
	ATM15 = { 'std': [], 'avg' : []}
	hfs = [2,3,4,5,6]
	for hf in hfs:
		atm = deltaHedgePortfolio(ttm, hf, [K])
		atm5 = deltaHedgePortfolio(ttm, hf, [K- 5, K, K+5])
		atm10 = deltaHedgePortfolio(ttm, hf, [K - 10 , K + 10])
		atm15 = deltaHedgePortfolio(ttm, hf, [K + 5, K + 10])	

		ATM['avg'].append(atm[0])
		ATM['std'].append(atm[1])
		ATM5['avg'].append(atm5[0])
		ATM5['std'].append(atm5[1])
		ATM10['avg'].append(atm10[0])
		ATM10['std'].append(atm10[1])
		ATM15['avg'].append(atm15[0])
		ATM15['std'].append(atm15[1])
	plt.figure()
	plt.plot(ATM['avg'], label='ATM')
	plt.plot(ATM5['avg'], label='ATM, ATM+5, ATM-5')
	plt.plot(ATM10['avg'], label='ATM+10, ATM-10')
	plt.plot(ATM15['avg'], label='ATM+5, ATM+10')
	plt.xticks(np.arange(len(hfs)), hfs),
	plt.title('AVG of error in option portfolio delta hedging with TTM='+str(ttm))
	plt.ylabel('AVG of E ($)')
	plt.xlabel('Hedging frequecy')
	plt.legend()
	plt.savefig('fig-7.eps', dpi=None, facecolor='w', edgecolor='w',
		        orientation='portrait', papertype=None, format=None,
		        transparent=False, bbox_inches=None, pad_inches=0.1,
		        metadata=None)

	plt.figure()
	plt.plot(ATM['std'], label='ATM')
	plt.plot(ATM5['std'], label='ATM, ATM+5, ATM-5')
	plt.plot(ATM10['std'], label='ATM+10, ATM-10')
	plt.plot(ATM15['std'], label='ATM+5, ATM+10')
	plt.xticks(np.arange(len(hfs)), hfs),
	plt.title('SD of error in option portfolio delta hedging with TTM='+str(ttm))
	plt.ylabel('SD of E ($)')
	plt.xlabel('Hedging frequecy')
	plt.legend()
	plt.savefig('fig-8.eps', dpi=None, facecolor='w', edgecolor='w',
		        orientation='portrait', papertype=None, format=None,
		        transparent=False, bbox_inches=None, pad_inches=0.1,
		        metadata=None)

	loadData('isx2010C.xls', 10)

	ATM = { 'std': [], 'avg' : []}
	ATM5 = { 'std': [], 'avg' : []}
	ATM10 = { 'std': [], 'avg' : []}
	ATM15 = { 'std': [], 'avg' : []}
	hfs = [2,3,4,5,6]
	for hf in hfs:
		atm = deltaHedgePortfolio(42, hf, [535])
		atm5 = deltaHedgePortfolio(42, hf, [535, 530, 540])
		atm10 = deltaHedgePortfolio(42, hf, [530, 540])
		atm15 = deltaHedgePortfolio(42, hf, [540, 545])	

		ATM['avg'].append(atm[0])
		ATM['std'].append(atm[1])
		ATM5['avg'].append(atm5[0])
		ATM5['std'].append(atm5[1])
		ATM10['avg'].append(atm10[0])
		ATM10['std'].append(atm10[1])
		ATM15['avg'].append(atm15[0])
		ATM15['std'].append(atm15[1])
	plt.figure()
	plt.plot(ATM['avg'], label='ATM')
	plt.plot(ATM5['avg'], label='ATM, ATM+5, ATM-5')
	plt.plot(ATM10['avg'], label='ATM+5, ATM-5')
	plt.plot(ATM15['avg'], label='ATM+5, ATM+10')
	plt.xticks(np.arange(len(hfs)), hfs),
	plt.title('AVG of error in option portfolio delta hedging with TTM=42')
	plt.ylabel('AVG of E ($)')
	plt.xlabel('Hedging frequecy')
	plt.legend()
	plt.savefig('fig-5.eps', dpi=None, facecolor='w', edgecolor='w',
		        orientation='portrait', papertype=None, format=None,
		        transparent=False, bbox_inches=None, pad_inches=0.1,
		        metadata=None)

	plt.figure()
	plt.plot(ATM['std'], label='ATM')
	plt.plot(ATM5['std'], label='ATM, ATM+5, ATM-5')
	plt.plot(ATM10['std'], label='ATM+5, ATM-5')
	plt.plot(ATM15['std'], label='ATM+5, ATM+10')
	plt.xticks(np.arange(len(hfs)), hfs)
	plt.title('SD of error in option portfolio delta hedging with TTM=42')
	plt.ylabel('SD of E ($)')
	plt.xlabel('Hedging frequecy')
	plt.legend()
	plt.savefig('fig-6.eps', dpi=None, facecolor='w', edgecolor='w',
		        orientation='portrait', papertype=None, format=None,
		        transparent=False, bbox_inches=None, pad_inches=0.1,
		        metadata=None)

	T1 = (41, 510, 0) #original option: TTM, strike price, sheet index
	T2 = (65, 510, 1) #replicating option:  TTM, strike price, sheet index
	avgs, stds = [], []
	hfs = [2,4,5,6,7,10,12,14,20]
	for t in hfs:
		(a, s, alphas, etas) = deltaVegaHedge(T1, T2, t)
		avgs.append(a)
		stds.append(s)
	plt.figure()
	plt.plot(stds)
	plt.title('SD of error in delta-vega-hedging a call with strike price='+str(T1[1]))
	plt.ylabel('SD of E ($)')
	plt.xlabel('Hedging frequecy')
	plt.xticks(np.arange(len(hfs)), hfs)
	plt.savefig('fig-9.eps', dpi=None, facecolor='w', edgecolor='w',
		        orientation='portrait', papertype=None, format=None,
		        transparent=False, bbox_inches=None, pad_inches=0.1,
		        metadata=None)

	plt.figure()
	plt.plot(avgs)
	plt.title('Average of error in delta-vega-hedging a call with strike price='+str(T1[1]))
	plt.ylabel('AVG of E ($)')
	plt.xlabel('Hedging frequecy')
	plt.xticks(np.arange(len(hfs)), hfs)
	plt.savefig('fig-10.eps', dpi=None, facecolor='w', edgecolor='w',
		        orientation='portrait', papertype=None, format=None,
		        transparent=False, bbox_inches=None, pad_inches=0.1,
		        metadata=None)
