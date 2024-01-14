import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

max_values = []
mean = []
losses = []
df = pd.read_csv('save/PER/records.csv', header=None)
for i in range(df.shape[0]):
    for j in range(50):
        max_values.append(df.iloc[i,j])
        mean.append(df.iloc[i,50])
        if j!=0:    
            losses.append(df.iloc[i,j+50])

fig, axs = plt.subplots(2,1, sharex=True)
sns.lineplot(x = range(1,len(max_values)+1), y = max_values, label='Max score', ax=axs[0])
sns.lineplot(x = range(1,len(mean)+1), y = mean, label='Mean every 50 episodes', ax=axs[0])
sns.lineplot(x = range(1,len(losses)+1), y = losses, ax=axs[1]).set_title("Mean of the Mean Squared Error loss in the training")
axs[0].set_title("Max score by episodes")
plt.show()

    

    

