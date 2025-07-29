import numpy as np 
import matplotlib.pyplot as plt

a = [0,1,2,3,4,5]

b = np.array(a)

print(b)

x = np.linspace(-10,10)
e = np.e

y1 = 2**x
y2 = e**x # ネイピア数
y3 = 3**x

# plt.plot(x,y1)
# plt.plot(x,y2)
# plt.plot(x,y3)

# plt.show()


def sigmoid(x):
    return  1/(1 + e**-x )

def df_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))
    


dx = 0.1
s = sigmoid(x)
y_d = (sigmoid(x + dx) - sigmoid(x)) /dx
y_df = df_sigmoid(x)

plt.plot(x,s, label="s")
plt.plot(x,y_d, label="ds")
plt.plot(x,y_df, label="df")
plt.show()
