import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('t9.csv', sep=',')
x = data["word"]
y = data["probability"]


fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('title')
plt.xlabel('x')
plt.ylabel('y')      
plt.xlabel('Probabilidad o pesos')
plt.ylabel('Palabras principales en el Documento')
plt.title('articles1.csv')
#plt.show()

plt.savefig(os.path.join('t9.png'), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures

