"""
Ben: Nearest neighbor search using a kd-tree and sklearn
"""
import numpy as np
import sklearn
from sklearn.neighbors import KDTree

""" for the 3D plot"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.style.use('fivethirtyeight')

#Which particle we want neighbours located around
target=0
#How many neighbors
K=4
debug=0

np.random.seed(0)# make it reproducible everytime it runs

#An array of arrays, each particle has its tuple or array
X = np.random.random((30, 3))  # 30 "particles" in 3D randomly generated
"""

1.6.4.5. Effect of leaf_size
For small sample sizes a brute force search can be more efficient than a tree-based query. 
This fact is accounted for in the ball tree and KD tree by internally 
switching to brute force searches within leaf nodes. 
The level of this switch can be specified with the parameter leaf_size. 
(from sklearn documentation)
"""
tree = KDTree(X, leaf_size=2)          
dist, ind = tree.query([X[target]], k=(K+1))                
print("Indices of the %i closest neighbors" % K,ind)  
print("Distances to the %i closest neighbors (0 is to itself)" % K,dist)  
sklearn.__version__ #only works at the python shell dunno why

"""
Plot everybody
Zip either packs a set of vectors or tuples into a numpy array, or unpacks
Matplotlib and mplot3d need every x,y,z, coordinate of each particle
on a separated array
"""
x,y,z = zip(*X)
ax.scatter(x,y,z, c='m',s=80, alpha=0.30)


#Plot neighbors
#Converts the tuple of one array to several tuples, each with one index of a neighbor
ind2=zip(*ind)

if debug==1:
 print(ind)
 print("ind2=",ind2)
for i in ind2:
       if debug==1: 
          print(list(i))
      #Now i convert the first tuple with the target index to a list and extract the first(only)
      #memeber, to be able to compare in the next if, apples vs apples, i.e.  int vs int
       if list(i)[0]==target:
          print("Skip target itself: sklearn consider the target itself a neighbor")
       else:
          x=X[i][0]
          y=X[i][1]
          z=X[i][2]
          print("i,x,y,z=",list(i),x,y,z)
          #ax.scatter(x,y,z, c='b',s=160,marker='s')
          ax.scatter(x,y,z, c='b',s=160)

#Plot target center particle
#ax.scatter(X[target][0],X[target][1],X[target][2], c='r',s=180,marker='v')
ax.scatter(X[target][0],X[target][1],X[target][2], c='r',s=180,label="target")

#ax.text("ind",color=black')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Nearest neighbors with sklearn+kdtree')
plt.show()

