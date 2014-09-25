import numpy as np

def sympy_demo():
    from sympy import *
    # question 2.2
    init_printing
    x = symbols('x1,x2,x3,x4')
    f = pow(x[0],4)*x[1] - x[2]/pow(1+x[1],2) + 100*x[0]*exp(x[2]) + pow(x[3],3)
    gradient = [diff(f, i) for i in x]
    pprint(gradient)
    hessian = [[diff(i, j) for j in x] for i in gradient]
    pprint(hessian)
    hessian1 = np.array([[diff(i, j).evalf(subs={x[0]:1, x[1]:1, x[2]:-5, x[3]:0}) for j in x] for i in gradient])
    print 'Eigenvalues at (1,1,-5,0):', np.linalg.eigvalsh(hessian1)
    hessian2 = np.array([[diff(i, j).evalf(subs={x[0]:1, x[1]:1, x[2]:-5, x[3]:2}) for j in x] for i in gradient])
    print 'Eigenvalues at (1,1,-5,2):', np.linalg.eigvalsh(hessian2)
    hessian3 = np.array([[diff(i, j).evalf(subs={x[0]:1, x[1]:1, x[2]:1, x[3]:2}) for j in x] for i in gradient])
    print 'Eigenvalues at (1,1,1,2):', np.linalg.eigvalsh(hessian3)
    
def three_dim_plots():
    # question 2.1
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    f = lambda x,y: x*x - y*y
    xs, ys = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
    ax.scatter(xs, ys, f(xs, ys), lw=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show(fig)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(xs, ys, f(xs, ys), 20)
    ax.streamplot(xs, ys, 2*xs, -2*ys, color='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(cs, shrink=0.95, extend='both', label='z values')
    plt.show(fig)

def taylor_series():
    # question 4.1
    import matplotlib.pyplot as plt
    f = lambda x: np.sin(x)
    f0 = lambda x: 0*x
    f1 = lambda x: x
    f2 = lambda x: x 
    f3 = lambda x: x - np.power(x, 3)/6
    fs = [f0, f1, f2, f3]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ['g', 'r', 'c', 'm']
    xs = np.linspace(-2, 2, 100)
    ax.plot(xs, f(xs), c='b', label='original fn')
    for i in range(len(fs)):
        ax.plot(xs, fs[i](xs), c=cs[i], label=str(i) +'-order approx')
    ax.legend(loc=2)
    ax.set_xlabel('x', fontsize=32)
    ax.set_ylabel('f(x)', fontsize=32)
    plt.show(fig)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xmin = -.5
    xmax = .5
    xs = np.linspace(xmin, xmax, 100)
    ax.plot(xs, f(xs) - f3(xs), c='b', label=r'$f-f_3$')
    ax.plot(xs, np.power(xs, 4), c='r', label=r'$|x|^4$')
    ax.set_xlim((xmin, xmax))
    ax.legend()
    plt.show(fig)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, np.abs(f(xs) - f3(xs))/np.power(np.abs(xs), 3))
    ax.set_xlim((xmin, xmax))
    ax.set_xlabel('x', fontsize=32)
    ax.set_ylabel(r'$\frac{|f-f_3|}{|x|^3}$', fontsize=32, rotation='horizontal', labelpad=50)
    plt.show(fig)

def taylor_series_ii():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    g = lambda x1,  x2: np.power(x1+x2, 4) - np.power(x1-x2, 3)
    gp = np.array((-256, -256))
    gpp = np.array(((192, 192), (192, 192)))
    # gx1 = -256; gx2 = -256
    # gx1x1 = 192; gx1x2 = 192; gx2x2 = 192
    g1 = lambda x1, x2: 256 + gp[0]*(x1 + 2) + gp[1]*(x2 + 2)
    g2 = lambda x1, x2: 256 + gp[0]*(x1 + 2) + gp[1]*(x2 + 2) + 0.5*(x1 + 2)*(x1 + 2)*gpp[0,0] + 0.5*(x2 + 2)*(x2 + 2)*gpp[1,1] + (x1 + 2)*(x2 + 2)*gpp[0,1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys = np.meshgrid(np.linspace(-3, -1, 50), np.linspace(-3, -1, 50))
    ax.scatter(xs, ys, g(xs, ys), c='b', alpha=0.3, lw=0)
    ax.scatter(xs, ys, g1(xs, ys), c='r', lw=0)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys = np.meshgrid(np.linspace(-3, -1, 50), np.linspace(-3, -1, 50))
    ax.scatter(xs, ys, g(xs, ys), c='b', alpha=0.3, lw=0)
    ax.scatter(xs, ys, g2(xs, ys), c='r', lw=0)
    plt.show()
    
    
    
                
def do_hw():
    three_dim_plots()
    sympy_demo()
    taylor_series()
    taylor_series_ii()

if __name__=='__main__':
    do_hw()

