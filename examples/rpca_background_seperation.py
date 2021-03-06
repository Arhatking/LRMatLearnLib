from datasets.data_loader import *
from RPCA.algorithms import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

v=load_video("../datasets/videos/escalator.avi")
(n_frames,d1,d2)=v.shape
v=v.reshape(n_frames, d1*d2)

(L,S)=altProjNiave(v, 2,100*n_frames)

L=L.reshape(n_frames,d1,d2)
S=S.reshape(n_frames,d1,d2)
v=v.reshape(n_frames,d1,d2)
all=np.concatenate((v,L,S), axis=2)

fig = plt.figure()
ims = []
for i in range(n_frames):
    im = plt.imshow(all[i,:,:], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                repeat_delay=10)
plt.show()