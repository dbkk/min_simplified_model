#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

params = pickle.load(open('../results/params.pkl', 'rb'))

#  unpack the parameters
for k, v in params.items():
    if isinstance(v, np.ndarray):
        exec(f"{k} = {v.tolist()}")
    else:
        exec(f"{k} = {v}")
    # print(f"{k}={v}")

print(f"dx={dx},  nx={nx}, ny={ny}")

pkl_files=os.listdir('../results')
pkl_files=[f for f in pkl_files if f.endswith('.pkl')]
# exclude the params.pkl
pkl_files=[f for f in pkl_files if f!='params.pkl']

# sort according to the number of the file before .pkl 
pkl_files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

# set the pkl_files (timepoints) to be visualized
pkl_files_list=pkl_files[0:10000000:4]

fig, ax = plt.subplots()
ims = []

for f in pkl_files_list: # tune this part to 
# for f in pkl_files[0:10:10]:
    with open(f'../results/{f}', 'rb') as fs:
        data = pickle.load(fs)

    u3 = data['u3']
    u1=data['u1']

    # u3[corner_cells] are zero

    rect=np.zeros((ny,nx+2))
    # assign the value of u3 to the rectangle edge
    rect[0,1:-1]=u3[:nx][::-1]
    rect[-1,1:-1]=u3[nx+ny:2*nx+ny]
    rect[:,0]=u3[nx:nx+ny]
    rect[:,-1]=u3[2*nx+ny:][::-1]

    timeframe=f.split('.')[0].split('_')[-1]
    time=round(float(timeframe)*dt,2)
    print(f"frame {timeframe}")
    # rect=u1

    # no axes
    ax.axis('off')
    im = ax.imshow(rect,cmap='Blues',vmin=0,vmax=15)

    time_text = ax.text(0.15, 1.05, f"Time: {time} sec", color='black',
                        fontsize=12, ha='left', va='top', transform=ax.transAxes)

    ims.append([im, time_text])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
ani.save('../plots/membrane_example.gif', writer='pillow')

# %%
