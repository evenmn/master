import numpy as np
import matplotlib.pyplot as plt

Theta_global       = 45
beta_global        = 0.9
time_steps_global  = 600
num_periods_global = 6

def simulate(
    beta=beta_global,                       # dimensionless parameter
    Theta=Theta_global,                     # initial angle in degrees
    epsilon=0,                              # initial stretch of wire
    num_periods=num_periods_global,         # simulate for num_periods
    time_steps_per_period=time_steps_global,# time step resolution
    plot=True,                              # make plots or not
    ):

    dt = 2*np.pi/time_steps_per_period
    N = num_periods*time_steps_per_period
    Theta = Theta*(np.pi/180)

    x    = np.zeros(N+2)
    y    = np.zeros(N+2)
    t    = np.zeros(N+2)

    x[0] = (1 + epsilon) * np.sin(Theta)
    y[0] = 1 - (1 + epsilon) * np.cos(Theta)
    L    = np.sqrt(x[0]**2 + (y[0] - 1)**2)

    x[1] = x[0] - (dt**2/2) * (beta/(1-beta)) * (1 - beta/L) * x[0]
    y[1] = y[0] - (dt**2/2) * (beta/(1-beta)) * (1 - beta/L) * (y[0] - 1) - beta * (dt**2)/2

    for n in range(N):
        L        = np.sqrt(x[n+1]**2 + (y[n+1] - 1)**2)
        betafraq = (beta/(beta - 1)) * (1 - beta/L)
        x[n+2]   = dt * dt * betafraq * x[n+1] + 2 * x[n+1] - x[n]
        y[n+2]   = dt * dt * betafraq * (y[n+1] -1 ) - beta * dt * dt + 2 * y[n+1] - y[n]
        t[n+2]   = n * dt
    
    theta = np.arctan2(x, (1 - y))

    if plot:
        label_size = {'size' : '16'}
        plt.plot(x, y)
        plt.xlabel(r'$x$', **label_size)
        plt.ylabel(r'$y$', **label_size)
        plt.title('Trajectory of elastic pendulum', **label_size)
        plt.axis('equal')
        plt.grid()
        plt.show()

        plt.plot(t, (180/np.pi) * theta)
        plt.plot(t, (180/np.pi) * Theta*np.cos(t))
        plt.legend(['Numerical', 'Classical'])
        plt.title('Time evolution of the angle')
        plt.xlabel('Time, t')
        plt.ylabel(r'Angel, $\theta$')
        plt.show()
    else:
        None

	return x, y, t, theta


def test_simulate():
    tol = 1e-16
    x, y, t, theta = simulate(Theta=0, epsilon=0, plot=False)
    if abs(max(x)) > tol or abs(max(y)) > tol:
        print "Error!"
    return None


def test_simulate_2(beta = beta_global, 
                    Theta = Theta_global, 
                    time_steps_per_period = time_steps_global,
                    num_periods = num_periods_global,
                    ):

    dt    = 2*np.pi/time_steps_per_period
    N     = num_periods*time_steps_per_period
    Theta = Theta*(np.pi/180)

    y     = np.zeros(N+2)
    t     = np.zeros(N+2)

    y[0]  = 1 - np.cos(Theta)
    y[1]  = y[0] - dt**2

    for n in range(N):
        betafraq = (beta/(1 - beta)) * (1 - beta/np.sqrt((y[n+1] - 1)**2))
        y[n+2]   = -dt * dt * betafraq * (y[n+1] - 1 ) - beta * dt * dt + 2 * y[n+1] - y[n]
        t[n+2]   = n * dt

    y_classic = y[0] * np.cos(np.sqrt(beta/(1 - beta))*t)

    label_size = {'size':'16'}
    plt.plot(t, y)
    plt.plot(t, y_classic)
    plt.title('Time evolution of the motion in y-direction', **label_size)
    plt.xlabel('Time, t', **label_size)
    plt.ylabel('y-direction', **label_size)
    plt.legend(['Numerical', 'Classical'])
    plt.show()

    for i in range(len(y)):
        tol = 10e-2
        if abs(y[i] - y_classic[i]) > tol:
            print "Error!"
            break
    return None


def demo(beta, Theta):
    simulate(beta=beta, 
             Theta=Theta, 
             plot=True, 
             num_periods=3, 
             time_steps_per_period=600)
    return None

demo(beta=0.9, Theta=9)
test_simulate()
test_simulate_2()
