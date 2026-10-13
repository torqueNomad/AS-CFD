import sys, trimesh, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
m=trimesh.load(sys.argv[1]); out=sys.argv[2]
fig=plt.figure(figsize=(16,10))
views=[(20,-60),(20,120),(90,-90),(0,0),(0,-90),(-30,45)]
for i,(el,az) in enumerate(views):
    ax=fig.add_subplot(2,3,i+1,projection='3d')
    tri=m.vertices[m.faces]
    n=m.face_normals; l=np.clip(n@np.array([0.4,0.5,0.75]),0.1,1)
    pc=Poly3DCollection(tri,facecolors=plt.cm.Blues(0.3+0.6*l),edgecolor='none')
    ax.add_collection3d(pc)
    b=m.bounds; c=b.mean(0); r=(b[1]-b[0]).max()/2
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    ax.view_init(el,az); ax.set_xlabel('X sway'); ax.set_ylabel('Y heave'); ax.set_zlabel('Z surge'); ax.set_title(f'el={el} az={az}')
plt.tight_layout(); plt.savefig(out,dpi=80)
