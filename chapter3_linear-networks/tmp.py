import matplotlib.pyplot as plt


year = [1500, 1600, 1700, 1800, 1900, 2000]
pop = [458, 580, 682, 1000, 1650, 6]

fig = plt.figure()

for i in range(1,5):
    plt.plot(year, [c*i for c in pop])
    plt.pause(1)
    fig.show()

