import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

t = sp.Symbol('t')
R = (1 + (1.5 * sp.sin(12 * t)))
Phi = ((1.25 * t) + (0.2 * sp.cos(12 * t)))



x = (R * sp.cos(Phi))
y = (R * sp.sin(Phi))
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)

T = np.linspace(0, 10, 1000)

X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
WX = np.zeros_like(T)
WY = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    WX[i] = sp.Subs(Wx, t, T[i])
    WY[i] = sp.Subs(Wy, t, T[i])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-300, 300], ylim=[-300, 300])

ax1.plot(X, Y)
ax1.plot([X.min(), X.max()], [0, 0], 'black')


P, = ax1.plot(X[0], Y[0], marker='o')
VLine, = ax1.plot([X[0], X[0]+VX[0]], [Y[0], Y[0]+VY[0]], 'r')
WLine, = ax1.plot([X[0],X[0] + WX[0]],[Y[0], Y[0] + WY[0]], 'g')

ArrowX = np.array([-0.2*4, 0, -0.2*4])
ArrowY = np.array([0.1*4, 0, -0.1*4])
RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowX+X[0]+VX[0], RArrowY+Y[0]+VY[0], 'r')

def anima(i):
    P.set_data(X[i], Y[i])
    VLine.set_data([X[i], X[i]+VX[i]], [Y[i], Y[i]+VY[i]])
    WLine.set_data([X[i],X[i]+WX[i]], [Y[i], Y[i]+WY[i]])
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowX+X[i]+VX[i], RArrowY+Y[i]+VY[i])
    return P, VLine, VArrow, WLine

anim = FuncAnimation(fig, anima, frames=1000, interval=2, repeat=False)

plt.show()