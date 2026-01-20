# SAMPLE USE OF MONTE CARLO SIMULATIONS
import numpy as np
import matplotlib.pyplot as plt


# Parameters
initial_investment = 10000
interest_rate = 0.07
ir_mean = 0.07 # rate of 7%
ir_std = 0.025 # std deviation of 2.5%
years = 10
simulations = 100

#sim_rates = np.zeros((simulations, years))  


# Output
portfolio_values = [] # #simulation values


for _ in range(simulations):

    this_sim_rates = np.random.normal(ir_mean, ir_std, years) # generates #years random interest rates

    print(this_sim_rates)
    
    fin_val = initial_investment * np.prod(1 + this_sim_rates) # initial_investment * (1 + a) * (1+b) ...
    portfolio_values.append(fin_val)


portfolio_values.sort()

for x in portfolio_values:
    print(x)


def get_freq_distribution(interval, pv):
    cnt = 0
    dif = 0
    freq = []
    for i in range(1, simulations):
    
        dif += pv[i] - pv[i-1]

        if (dif > interval):
            dif = 0
            freq.append(cnt)
            cnt = 0

        cnt += 1

    return freq



i_lo = 500
i_hi = 600
i_len = 50


for s in range(i_lo, i_hi + 1, i_len):
    f = get_freq_distribution(s, portfolio_values)
    X = list(range(1, len(f) + 1))
    plt.plot(X, f, label = f"{(s - 200) / 50}")

plt.show()


