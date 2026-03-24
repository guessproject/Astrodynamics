import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib import cm
from astrodynamics import *
import pylab as P
import lambert as lb
import math
import numpy as np

def PlotTwoBodyOrbit(initState, dur, nSteps, centralBody):

    mu = GetPlanetMu(centralBody)
    X, Y, Z = PropagateTwoBody(initState, dur, nSteps, mu)
   
    ##### Orbit & Earth Plot #####
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    
    # Adding Figure Title and Labels
    ax.set_title('Orbit in inertial frame')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_zlabel('z (km)')
    
    # plotting orbits
    ax.plot3D(X, Y, Z, c='green', label='Keplerian orbit')
    
    # Plotting Earth
    N = 50
    phi = np.linspace(0, 2 * np.pi, N)
    theta = np.linspace(0, np.pi, N)
    theta, phi = np.meshgrid(theta, phi)
    r_Cb = GetPlanetRadius(centralBody)
    X_Cb  = r_Cb * np.cos(phi) * np.sin(theta)
    Y_Cb  = r_Cb * np.sin(phi) * np.sin(theta)
    Z_Cb  = r_Cb * np.cos(theta)
    ax.plot_surface(X_Cb, Y_Cb, Z_Cb, color='gray', alpha=0.7)
    
    # define axis limits
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim * 3/4)
    
    plt.legend()
    plt.show()
	

def PlotThreeBodyOrbit(initState, tau, nSteps, mu):

    X, Y, Z = PropagateThreeBody(initState, tau, nSteps, mu)
    
	# Constant m1 and m2 Rotational Frame Locations for CR3BP Primaries
    m1_loc = [-mu, 0, 0]
    m2_loc = [(1-mu), 0, 0]
    fig = plt.figure(figsize=(10,7))
    ax = plt.axes(projection='3d')

    # Adding Figure Title and Labels
    ax.set_title('Rotating Frame CR3BP Orbit (\u03BC = ' + str(round(mu, 6)) + ')')
    ax.set_xlabel('x [ND]')
    ax.set_ylabel('y [ND]')
    ax.set_zlabel('z [ND]')

    # Plotting Rotating Frame Positions
    ax.plot3D(X, Y, Z, c='green')
    ax.plot3D(m1_loc[0], m1_loc[1], m1_loc[2], c='black', marker='o', label='Primary')
    ax.plot3D(m2_loc[0], m2_loc[1], m2_loc[2], c='purple', marker='o', label='Secondary')
    ax.legend()
    # Setting Axis Limits
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim * 3 / 4)

    # Setting Axis Limits
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim * 3 / 4)
    plt.show()    
	

def PlotTransfer(centralBody, dep_planet, arr_planet, jd_dep, jd_arr, nSteps):
    
    mu = GetPlanetMu(centralBody)
    
    # flight time (days)
    ft = jd_arr - jd_dep
	
    # get the COEs and the state vectors
    coe1, r1, v1, jd1 = GetPlanetStateVector(GetPlanetId(dep_planet), jd_dep)
    coe2, r2, v2, jd2 = GetPlanetStateVector(GetPlanetId(arr_planet), jd_arr)

    initState1 = np.concatenate((r1, v1))
    initState2 = np.concatenate((r2, v2))
    prop1 = GetPeriodFromSma(GetPlanetOrbitalRadius(dep_planet), GetPlanetMu('Sun'))
    prop2 = GetPeriodFromSma(GetPlanetOrbitalRadius(arr_planet), GetPlanetMu('Sun'))

    X1, Y1, Z1 = PropagateTwoBody(initState1, prop1, nSteps, mu)
    X2, Y2, Z2 = PropagateTwoBody(initState2, prop2, nSteps, mu)
	
	# transfer orbit
    v1_trans, v2_trans = lb.solve(GetPlanetMu('Sun'), r1, r2, ft*86400, 'prograde', 'low', 0)
    initState_trans = np.concatenate((r1, v1_trans[0]))
    X_trans, Y_trans, Z_trans = PropagateTwoBody(initState_trans, ft*86400, nSteps, mu)
	
	# calculate Delta V
    delta_V1 = np.array([[v1_trans[0][0] - v1[0]], [v1_trans[0][1] - v1[1]], [v1_trans[0][2] - v1[2]]])
    delta_V2 = np.array([[v2_trans[0][0] - v2[0]], [v2_trans[0][1] - v2[1]], [v2_trans[0][2] - v2[2]]])
    delta_V1_norm = np.linalg.norm(delta_V1)
    delta_V2_norm = np.linalg.norm(delta_V2)
    print('Flight Time = ' + str(round(ft, 2)) + ' days')
    print('V1∞    = ' + str(delta_V1_norm) + ' km/sec')
    print('V2∞    = ' + str(delta_V2_norm) + ' km/sec')
    print('V∞ tot = ' + str(delta_V1_norm + delta_V2_norm) + ' km/sec')
	
    ##### Orbit Plot #####
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    
    # Adding Figure Title and Labels
    ax.set_title('Transfer orbit')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_zlabel('z (km)')
    
    # plotting orbits
    ax.plot3D(X1, Y1, Z1, c='gray')
    ax.plot3D(X2, Y2, Z2, c='gray')
    ax.plot3D(0, 0, 0, c='black', marker='o', label=centralBody)
    ax.plot3D(X1[0], Y1[0], Z1[0], c='blue', marker='o', label=dep_planet)
    ax.plot3D(X2[0], Y2[0], Z2[0], c='green', marker='o', label=arr_planet)
    ax.plot3D(X_trans, Y_trans, Z_trans, c='red', label='Transfer orbit')
    
    
    # define axis limits
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim * 3/4)
    
    plt.legend()
    plt.show()


def PrintCr3bpEarthMoonParameters():
    # define characteristic quantities
    G = 6.67408E-20                  # Univ. Gravitational Constant [km3 kg-1 s-2]
    mEarth = 5.972365352228322E+24   # Mass of the Earth [kg]
    mMoon = 7.34603167111482e+22    # Mass of the Moon [kg]
    a = 390877.4158212686            # Semimajor axis of the Moon orbit (circular) from STK

    m1 = mEarth
    m2 = mMoon
    Mstar = m1+m2                        # m star
    Lstar = a                            # Length Parameter
    Tstar = 2*math.pi*(Lstar**3/(G*Mstar))**(1/2)  # Time Parameter - Moon rotation period
    Vstar = 2*math.pi*Lstar/Tstar
    mu = m2/Mstar                        # mass parameter
    nd_sma = 1.0                         # nondimensional semimajor axis

    print("Dimensional parameters for the Earth-Moon system:")
    print('Mstar = ' + str(Mstar) + ' kg')
    print('Lstar = ' + str(Lstar) + ' km')
    print('Tstar = ' + str(Tstar/86400) + ' days')
    print('Vstar = ' + str(Vstar) + ' km/sec')
    print('')
    print("Mass parameter for the Earth Moon system:")
    print('\u03BC = ' + str(mu))
	
	
def PrintJacobiConstantLibrationPoints(mu):

 df = GetLibrationPointsCoord(mu)
 l1x = df._get_value(0, 'x') 
 l2x = df._get_value(1, 'x') 
 l3x = df._get_value(2, 'x') 
 l4x = df._get_value(3, 'x') 
 l4y = df._get_value(3, 'y') 
 l5x = df._get_value(4, 'x') 
 l5y = df._get_value(4, 'y') 
 
 l1State = np.array([l1x, 0, 0, 0, 0, 0])
 l2State = np.array([l2x, 0, 0, 0, 0, 0])
 l3State = np.array([l3x, 0, 0, 0, 0, 0])
 l4State = np.array([l4x, l4y, 0, 0, 0, 0])
 l5State = np.array([l5x, l5y, 0, 0, 0, 0])
 
 JcL1 = GetJacobiConstant(l1State, mu)
 JcL2 = GetJacobiConstant(l2State, mu)
 JcL3 = GetJacobiConstant(l3State, mu)
 JcL4 = GetJacobiConstant(l4State, mu)
 JcL5 = GetJacobiConstant(l5State, mu)
 
 print("Jacobi Constant at libration points for μ = " + str(mu))
 print('L1 = ' + str(JcL1))
 print('L2 = ' + str(JcL2))
 print('L3 = ' + str(JcL3))
 print('L4 = ' + str(JcL4))
 print('L5 = ' + str(JcL5))
	
	
def PlotLagrangianPoints(df, mu):

 print("Libration points coordinates for μ = " + str(mu))
 print("")
 print(df)   

 circle = mpath.Path.unit_circle()
 wedge_1 = mpath.Path.wedge(90, 180)
 wedge_2 = mpath.Path.wedge(270, 0)

 verts = np.concatenate([circle.vertices, wedge_1.vertices[::-1, ...], wedge_2.vertices[::-1, ...]])
 codes = np.concatenate([circle.codes, wedge_1.codes, wedge_2.codes])
 center_of_mass = mpath.Path(verts, codes)
 
 # get the coordinates from the dataframe
 l1x = df._get_value(0, 'x') 
 l2x = df._get_value(1, 'x') 
 l3x = df._get_value(2, 'x') 
 l4x = df._get_value(3, 'x') 
 l4y = df._get_value(3, 'y') 
 l5x = df._get_value(4, 'x') 
 l5y = df._get_value(4, 'y') 

 # These give us the coordinates of the orbits of m2 and m1
 x_2 = (1 - mu) * np.cos(np.linspace(0, np.pi, 100))
 y_2 = (1 - mu) * np.sin(np.linspace(0, np.pi, 100))
 x_1 = (-mu) * np.cos(np.linspace(0, np.pi, 100))
 y_1 = (-mu) * np.sin(np.linspace(0, np.pi, 100))

 fig, ax = plt.subplots(figsize=(10,5), dpi=96)
 ax.set_xlabel("$x^*$")
 ax.set_ylabel("$y^*$")

 # Plot the orbits
 ax.axhline(0, color='k')
 ax.plot(np.hstack((x_2, x_2[::-1])), np.hstack((y_2, -y_2[::-1])))
 ax.plot(np.hstack((x_1, x_1[::-1])), np.hstack((y_1, -y_1[::-1])))
 ax.plot([-mu, 0.5 - mu, 1 - mu, 0.5 - mu, -mu], [0, np.sqrt(3)/2, 0, -np.sqrt(3)/2, 0], 'k', ls="--", lw=1)
 # Plot the Lagrange Points and masses
 ax.plot(l1x, 0, 'rv', label="$L_1$")
 ax.plot(l2x, 0, 'r^', label="$L_2$")
 ax.plot(l3x, 0, 'rp', label="$L_3$")
 ax.plot(l4x, l4y, 'rX', label="$L_4$")
 ax.plot(l5x, l5y, 'rs', label="$L_5$")
 ax.plot(0, 0, 'k', marker=center_of_mass, markersize=10)
 ax.plot(-mu, 0, 'bo', label="$m1$")
 ax.plot(1 - mu, 0, 'go', label="$m2$")
 ax.legend()
 plt.legend(loc='upper left')
 ax.set_aspect("equal")
 
 
def PlotMultipleZeroVelocityCurve(mu):

 nd_sma = 1
 # define the boundaries for the Jacobi constant
 CJ=np.linspace(2.5, 4, 10)
 N=512           # N is the number of grid points on an NXN grid. 
 SIZE=3.0*nd_sma  # SIZE is the full width of the plotting area.

 # variables definition
 d1=mu*nd_sma
 d2=(1-mu)*nd_sma
 dx=SIZE/float(N)

 x=[]
 y=[]
 J=[]

 # set up the grid
 for i in range(N):
  x.append(dx*i+dx*0.5-SIZE/2.0)
  y.append(dx*i+dx*0.5-SIZE/2.0)
  J.append([])
  for j in range(N):
    J[i].append(0.0)

 # calculate J for zero velocity 
 for i in range(N):
  for j in range(N):
    r1=math.sqrt((x[i]+d1)**2+y[j]**2)
    r2=math.sqrt((x[i]-d2)**2+y[j]**2)
    J[j][i]=2*((1/2)*(x[i]**2+y[j]**2)+((1-mu)/r1+mu/r2))

 # plot
 P.figure(figsize = (9.5,7))
 P.title('Zero-Velocity Curve for μ = ' + str(mu))
 P.xlabel('X Position (scaled units)')
 P.ylabel('Y Position (scaled units)')
 P.xlim(-SIZE/2.,SIZE/2.)
 P.ylim(-SIZE/2.,SIZE/2.)
 # plot the colored, background levels
 cb=P.contourf(x,y,J,levels=CJ,extend='max')
 # plot the contour lines
 C = P.contour(x,y,J,levels=CJ,colors='black')
 plt.clabel(C, inline=1, fontsize=10)

 # plot m1 and m2
 P.scatter([-d1,d2],[0,0])
 label=["M1","M2"]
 for i,x in enumerate([-d1,d2]):
   P.text(x+10*dx,-10*dx,label[i])
 # add a colorbar
 P.colorbar(cb)
 P.show()
 
 
def PlotSingleZeroVelocityCurve(jacobi, mu):
 
 df = GetLibrationPointsCoord(mu)
 l1x = df._get_value(0, 'x') 
 l2x = df._get_value(1, 'x') 
 l3x = df._get_value(2, 'x') 

 # define the boundaries for the Jacobi constant
 CJ=np.linspace(0, jacobi, 2)
 N=512           # N is the number of grid points on an NXN grid. 
 SIZE=3.0  # SIZE is the full width of the plotting area.

 # variables definition
 d1=mu
 d2=(1-mu)
 dx=SIZE/float(N)

 x=[]
 y=[]
 J=[]

 # set up the grid
 for i in range(N):
  x.append(dx*i+dx*0.5-SIZE/2.0)
  y.append(dx*i+dx*0.5-SIZE/2.0)
  J.append([])
  for j in range(N):
    J[i].append(0.0)

 # calculate J for zero velocity 
 for i in range(N):
  for j in range(N):
    r1=math.sqrt((x[i]+d1)**2+y[j]**2)
    r2=math.sqrt((x[i]-d2)**2+y[j]**2)
    J[j][i]=2*(0.5*(x[i]**2+y[j]**2)+((1-mu)/r1+mu/r2))

 # plot
 P.figure(figsize = (9.5,7))
 P.title('Zero-Velocity Surfaces (Rotating Frame)')
 P.xlabel('X Position (scaled units)')
 P.ylabel('Y Position (scaled units)')
 P.xlim(-SIZE/2.,SIZE/2.)
 P.ylim(-SIZE/2.,SIZE/2.)
 # plot the contour lines
 C = P.contour(x,y,J,levels=CJ,colors='black')
 plt.clabel(C, inline=5, fontsize=10)

 plt.imshow(J, extent=[-1.5, 1.5, -1.5, 1.5], cmap='binary_r', vmin = jacobi, vmax = 3, alpha=0.5)
 plt.colorbar();

 # plot scatter points
 P.scatter([-d1,d2],[0,0])
 P.scatter([l1x,l2x,l3x],[0,0,0])
 label=["L1","L2","L3"]
 for i,x in enumerate([l1x,l2x,l3x]):
   P.text(x+5*dx,-5*dx,label[i])


def GetMu(planetName):
    '''
    Gets the gravitational parameter of the selected planet
    Args:
        planetName: one of the allowed name
    Returns:
        mu: gravitational parameter of the selected planet (km^3/sec^2)
        RETURNS -1 IF WRONG PLANET NAME
    '''
    # planets list  
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter',
               'Saturn', 'Uranus', 'Neptune', 'Pluto']

    radius = -1
    
    if planetName == 'Sun':
        mu = 1.32712440018e11
    if planetName == 'Mercury':
        mu = 2.20319e4
    if planetName == 'Venus':
        mu = 3.24859e5
    if planetName == 'Earth':
        mu = 3.986e5
    if planetName == 'Mars':
        mu = 4.28284e4
    if planetName == 'Jupiter':
        mu = 1.26713e8
    if planetName == 'Saturn':
        mu = 3.79406e7          
    if planetName == 'Uranus':
        mu = 5.79456e6  
    if planetName == 'Neptune':
        mu = 6.83653e6   
    if planetName == 'Pluto':
        mu = 9.755e2
    return mu


def PlotCircularOrbitParameters():
    ## Constants ##
    j2 = 0.0010826267         # non-dimensional
    meanEarthRadius = 6378.14 # km
    mu = 3.986004418E+05      # Earth's gravitational parameter  

    minAltitude = 250 #km
    maxAltitude = 36000 #km
    nSteps = 100

    altitude = np.linspace(minAltitude, maxAltitude, nSteps)
    sma = np.linspace( meanEarthRadius + minAltitude,  meanEarthRadius + maxAltitude, nSteps)
    period = np.array([None]*nSteps)
    velocity = np.array([None]*nSteps)
    energy = np.array([None]*nSteps)
    revsDay = np.array([None]*nSteps)
    for i in range(nSteps):
        period[i]   = (2*math.pi*math.sqrt(math.pow(sma[i],3)/mu))
        velocity[i] = math.sqrt(mu/sma[i])
        energy[i]   = -(mu/(2*sma[i]))
        revsDay[i]  = (2*math.pi/period[i])* 86400 / (2*math.pi)

    fig = plt.figure(figsize = (10,6))
    plt.subplot(2,2,1)
    plt.plot(altitude/1000, period/60, 'black')
    plt.xlabel('Altitude (km*10^3)')
    plt.ylabel('Period (min)')
    plt.grid(True)
    plt.subplot(2,2,2)
    plt.plot(altitude/1000, velocity, 'black')
    plt.xlabel('Altitude (km*10^3)')
    plt.ylabel('Velocity (km/s)')
    plt.grid(True)
    plt.subplot(2,2,3)
    plt.plot(altitude/1000, revsDay, 'black')
    plt.xlabel('Altitude (km*10^3)')
    plt.ylabel('Revs per day')
    plt.grid(True)
    plt.subplot(2,2,4)
    plt.plot(altitude/1000, energy,'black')
    plt.xlabel('Altitude (km*10^3)')
    plt.ylabel('Energy (km^2/s^2)')
    plt.grid(True)
    plt.show()
	
	
def PlotNodalRegression(minAlt, maxAlt, minInc, maxInc, ecc):
 
 inc = np.linspace(minInc, maxInc, 100)
 alt = np.linspace(minAlt, maxAlt, 100)

 fig = plt.figure(figsize = (10,6))
 ax = plt.axes(projection = '3d')

 X, Y = np.meshgrid(alt,inc)
 Z  = NodalRegression(X, Y, ecc)

 surf = ax.plot_surface(X,Y,Z, cmap = cm.autumn, linewidth=1, antialiased=True)

 ax.set_title('Nodal regression rate as function of altitude and inclination (deg/solar day)')

 ax.set_xlabel('alt (km)', labelpad=20)
 ax.set_ylabel('inc (deg)', labelpad=20)
 plt.show()
 
 
def PlotApsidalRotation(minAlt, maxAlt, minInc, maxInc, ecc):
 
 inc = np.linspace(minInc, maxInc, 100)
 alt = np.linspace(minAlt, maxAlt, 100)

 fig = plt.figure(figsize = (10,6))
 ax = plt.axes(projection = '3d')

 X, Y = np.meshgrid(alt,inc)
 Z = ApsidalRotation(X, Y, ecc)

 surf = ax.plot_surface(X,Y,Z, cmap = cm.cool, linewidth=1, antialiased=True)

 ax.set_title('Apsidal regression rate as function of altitude and inclination (deg/day)')

 ax.set_xlabel('alt (km)', labelpad=20)
 ax.set_ylabel('inc (deg)', labelpad=20)
 plt.show()
 
 
def PlotSsoInclination(ecc):
    ## Constants ##
    j2 = 0.0010826267            # non-dimensional
    meanEarthRadius = 6378.14    # km
    mu = 3.986004418E+05         # Earth's gravitational parameter  
    omegaDot = 1.991063853E-7 # nodal drift for SSO orbits

    minAltitude = 250  #km
    maxAltitude = 1000 #km
    nSteps = 400 

    altitude = np.linspace(minAltitude, maxAltitude, nSteps)
    sma = np.linspace( meanEarthRadius + minAltitude,  meanEarthRadius + maxAltitude, nSteps) 
	
    inc = np.array([None]*nSteps)
    for i in range(nSteps):
     inc[i]   =  math.degrees(math.acos(-2*sma[i]**(7/2)*(omegaDot*((1-ecc**2))**2)/(3*meanEarthRadius**2*j2*math.sqrt(mu))))
    
    
    plt.plot(altitude, inc, color = 'black', label = 'inc')
    plt.grid(True)
    plt.xlabel('alt (km)')
    plt.ylabel('inc (deg)')
    leg = plt.legend()
    fig = plt.gcf()
    plt.title('SSO Orbit Inclination for ecc = ' + str(ecc))
    fig.set_size_inches(12,4)
    plt.show()


def PlotAngularExtents(periAlt, departurePlanet):
    
    mu = GetPlanetMu(departurePlanet)
    c3   = np.zeros(80)
    e    = np.zeros(80)
    beta = np.zeros(80)
    i = 1
    for x in range (80):
      c3[x]   = i
      e[x]    = 1 + ((periAlt + 6378.15)*c3[x])/(mu)
      beta[x] = math.degrees(math.acos(1/e[x]))
      i       = i+1

    f1 = plt.figure(figsize = (10,4))
    ax = f1.add_subplot(111)
    ax.set_xlabel('C3 (km^2/sec^2)', labelpad=2)
    ax.set_ylabel('β (deg)', labelpad=2)
    ax.plot(c3, beta, color= 'black', linewidth=2) 
    plt.title('Angular extent')
    plt.grid(True)
    plt.show()
	
	
def PlotHCW(initialState, alt, numRevs):
    def HCW(state, t):
     x = state[0]
     y = state[1]
     z = state[2]
     x_dot = state[3]
     y_dot = state[4]
     z_dot = state[5]
     x_ddot = 2*omega*y_dot + 3*math.pow(omega,2)*x
     y_ddot = -2*omega*x_dot
     z_ddot = -math.pow(omega,2)*z
     dstate_dt = [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]
     return dstate_dt
	 
    rEarth = 6378.137  #km
    #derive data
    omega = math.sqrt(398600.5/pow(alt+rEarth,3)) #rad/sec
    period = (2*math.pi)/omega # sec
    t = np.linspace(0, numRevs * period, 1000)  # ND Time as array

    # Numerically Integrating
    sol = odeint(HCW, initialState, t)
    x = sol[:, 0]
    y = sol[:, 1]
    z = sol[:, 2]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    fig.suptitle('Relative path', fontsize=10)

    # --- Proiezione XY ---
    ax1.plot(y, x, color='blue')
    ax1.set_title('XY')
    ax1.set_xlabel('Y (m)')
    ax1.set_ylabel('X (m)')
    ax1.grid(True)
    ax1.autoscale(enable=True, axis='y')
    #ax1.axis('equal')

    # --- Proiezione XZ ---
    ax2.plot(x, z, color='red')
    ax2.set_title('XZ')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.grid(True)
    ax2.autoscale(enable=True, axis='y')

    # --- Proiezione YZ ---
    ax3.plot(y, z, color='green')
    ax3.set_title('YZ')
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.grid(True)
    ax3.autoscale(enable=True, axis='y')

    # Ottimizzazione degli spazi
    plt.tight_layout()
    plt.show()
	
    C = math.sqrt(math.pow((3*initialState[0] + (2*initialState[4]/omega)),2) + math.pow((initialState[3]/omega),2))
    print('Radial oscillation amplitude = ' + str(math.trunc(C)) + ' m')
    print('Intrack oscillation amplitude = ' + str(math.trunc(2*C)) + ' m')
    print('Critical intrack rate for drift stopping = ' + str(-2*omega*initialState[0]) + ' m/s')
    print('Critical intrack rate for straight line = ' + str(-1.5*omega*initialState[0]) + ' m/s')

    fig2 = plt.figure(figsize=(10,6))
    ax = plt.axes(projection='3d')
    ax.set_title('Relative motion')
    ax.set_xlabel('Y (Intrack - m)')
    ax.set_ylabel('Z (Crosstrack - m)')
    ax.set_zlabel('X (Radial - m)')

    ax.plot3D(y, z, x, c='green')
    ax.plot3D(0, 0, 0, c='black', marker='o')
    plt.show()	
    
    
def PlotHohmannVsBielliptic(r1, r2, rb):
    dv_hohmann = calculate_hohmann_delta_v(r1, r2)
    dv_bielliptic = calculate_bielliptic_delta_v(r1, r2, rb)

    print(f"Hohmann Delta-V: {dv_hohmann}")
    print(f"Bielliptic Delta-V: {dv_bielliptic}")

    # Parametric analysis: vary rb
    # Fix r1 and r2, vary rb from r2 to some multiple, say 100*r2
    rb_ratios = np.linspace(1.01, 10, 100)  # rb / r2 from 1.01 to 100
    rbs = rb_ratios * r2

    dv_bi_list = []
    for rb_val in rbs:
        try:
            dv_bi_list.append(calculate_bielliptic_delta_v(r1, r2, rb_val))
        except ValueError:
            dv_bi_list.append(np.nan)  # If invalid, but should be fine

    dv_hoh_list = [dv_hohmann] * len(rb_ratios)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(rb_ratios, dv_bi_list, label='Bielliptic Delta-V')
    plt.plot(rb_ratios, dv_hoh_list, label='Hohmann Delta-V', linestyle='--')
    plt.xlabel('Intermediate Radius Ratio (rb / r2)')
    plt.ylabel('Total Delta-V')
    plt.title('Hohmann vs Bielliptic Transfer Delta-V')
    plt.legend()
    plt.grid(True)
    plt.show()

def PlotIntrakManEffects(init_sma, init_ecc, delta_v):
    sma_array = np.empty(360)
    ecc_array = np.empty(360)
    aop_array = np.empty(360)
    for i in range(360):
        initStateKep = np.array([init_sma, init_ecc, 0, 0, 0, i]) # keplerian state (sma, ecc, inc, aop, raan, ta)
        mu = GetPlanetMu('Earth')
        initStateSv = keplerian_to_state(initStateKep, mu)

        # compute the original magnitude
        init_v = math.sqrt(initStateSv[3]*initStateSv[3] + initStateSv[4]*initStateSv[4] + initStateSv[5]*initStateSv[5])
        final_v = init_v + delta_v*0.001
        # scaling factor
        k = final_v/init_v

        final_vx = k*initStateSv[3]
        final_vy = k*initStateSv[4]
        final_vz = k*initStateSv[5]

        final_sv = np.array([initStateSv[0], initStateSv[1], initStateSv[2], final_vx , final_vy , final_vz ])
        kep = state_to_keplerian(final_sv, mu)

        sma_array[i] = kep[0] - init_sma
        ecc_array[i] = kep[1] - init_ecc
        if kep[4]>180:
            aop_array[i] = kep[4] -360
        else:
            aop_array[i] = kep[4]
        

    # figsize imposta (larghezza, altezza) in pollici
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    # 3. Plot del primo array
    axs[0].plot(np.arange(360), sma_array, color='blue')
    axs[0].set_title('Delta semimajor axis (Km)')
    axs[0].set_ylabel('km')
    axs[0].set_xticks(np.arange(0, 361, 15))
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.8)
    axs[0].grid(True)

    # 4. Plot del secondo array
    axs[1].plot(np.arange(360), ecc_array, color='red')
    axs[1].set_title('Delta eccentricity')
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.8)
    axs[1].grid(True)

    # 5. Plot del terzo array
    axs[2].plot(np.arange(360), aop_array, color='green')
    axs[2].set_title('Delta argument of periapsis')
    axs[2].set_xlabel('True anomaly at Δv (deg)')
    axs[2].set_ylabel('deg')
    axs[2].grid(True, which='both', linestyle='--', linewidth=0.8)
    axs[2].grid(True)

    # Migliora la spaziatura tra i grafici
    plt.tight_layout()

    # Mostra il risultato
    plt.show()