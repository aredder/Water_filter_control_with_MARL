import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 1, 0.00001)
x = [np.exp((2*x-1)**2/(10*(x-1)*x)) for x in t]
plt.figure()
plt.plot(t, x)
plt.xlabel('$s_2$')
plt.ylabel('$r_2(s_2)$')
plt.savefig('reward_function.png')
plt.show()
