import numpy as np
from datetime import datetime, timedelta, timezone
from scipy import optimize
from scipy.optimize import newton
from scipy.integrate import odeint
import math
import pandas as pd

########################################################### TWO BODY MODEL ########################################################

def TwoBodyModel(state, t, mu):
	##########################################
	#Implements the equation of motion for the Two Body problem
	#Args:
		# state:       array containing initial position and velocity (km, km/sec)
		# t:           time step 
		# mu:          gravitational parameter of the central body (km^3/sec^2)
	#Returns:
		# dstate_dt:   derivative of the state
	##########################################
	
    x = state[0]
    y = state[1]
    z = state[2]
    x_dot = state[3]
    y_dot = state[4]
    z_dot = state[5]
    x_ddot = -mu * x / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    y_ddot = -mu * y / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    z_ddot = -mu * z / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    dstate_dt = [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]
    return dstate_dt


def PropagateTwoBody(initState, dur, nSteps, mu):
	##########################################
	#Integrates the equation of motion for the Two Body problem
	#Args:
		# initState: numpy array of initial state vector - np.array([[x], [y], [z], [vx], [vy], [vz]]
		# t:         duration of integration interval(sec)
		# mu:        gravitational parameter of the central body (km^3/sec^2)
	#Returns:
		# X, Y, Z:   numpy array: 1x3 matrix which contains the position vector
	##########################################

    t = np.linspace(0, dur ,nSteps) 
    sol = odeint(TwoBodyModel, initState, t, args=(mu,))
    X  = sol[:, 0]  # X-coord [km] of satellite over time interval 
    Y  = sol[:, 1]  # Y-coord [km] of satellite over time interval
    Z  = sol[:, 2]  # Z-coord [km] of satellite over time interval
    #Vx = sol[:, 3]  # X-vel [km/sec] of satellite over time interval 
    #Vy = sol[:, 4]  # Y-vel [km/sec] of satellite over time interval
    #Vz = sol[:, 5]  # Z-vel [km/sec] of satellite over time interval
    return X, Y, Z


########################################################### THREE BODY MODEL ########################################################

def ThreeBodyModel(state, t, mu):
    ##########################################
	#Implements the equation of motion for the Three Body problem
	#Args:
		# state:       array containing initial position and velocity (km, km/sec)
		# t:           time step 
		# mu:          gravitational parameter of the central body (km^3/sec^2)
	#Returns:
		# dstate_dt:   derivative of the state
	##########################################
 x = state[0]
 y = state[1]
 z = state[2]
 x_dot = state[3]
 y_dot = state[4]
 z_dot = state[5]
 x_ddot = x+2*y_dot-((1-mu)*(x+mu))/((x+mu)**2+y**2+z**2)**(3/2)-(mu*(x-(1-mu)))/((x-(1-mu))**2+y**2+z**2)**(3/2)
 y_ddot = y-2*x_dot-((1-mu)*y)/((x+mu)**2+y**2+z**2)**(3/2)-(mu*y)/((x-(1-mu))**2+y**2+z**2)**(3/2)
 z_ddot = -((1-mu)*z)/((x+mu)**2+y**2+z**2)**(3/2)-(mu*z)/((x-(1-mu))**2+y**2+z**2)**(3/2)
 dstate_dt = [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]
 return dstate_dt


def PropagateThreeBody(ndInitialState, tau, nSteps, mu):
	##########################################
	#Integrates the equation of motion for the Two Body problem
	#Args:
		# ndInitState: numpy array of non-dimensional initial state vector - np.array([[x], [y], [z], [vx], [vy], [vz]]
		# tau:         non-dimensional duration of integration interval # 6.28 is the time needed for the secondary body to complete one revolution
		# mu:          mass ratio (secondary mass / (primary mass + secondary mass))
	#Returns:
		# X, Y, Z:     numpy array: 1x3 matrix which contains the position vector
	##########################################

 t = np.linspace(0, tau, nSteps)  # ND Time as array

 # CR3BP Model: user-defined Python functionthat will be used in odeint to numerically integrate our state vector.
 # The odeint function takes the current state vector to create the time-derivative state vector. 
 # Notice we pull the position and velocity components from the state vector to create the state vector time-derivative.

 # Numerically Integrating
 sol = odeint(ThreeBodyModel, ndInitialState, t, args=(mu,))

 # Rotational Frame Position Time History
 X_rot = sol[:, 0]
 Y_rot = sol[:, 1]
 Z_rot = sol[:, 2]

 # Inertial Frame Position Time History
 #X_Iner = sol[:, 0]*np.cos(t) - sol[:, 1]*np.sin(t)
 #Y_Iner = sol[:, 0]*np.sin(t) + sol[:, 1]*np.cos(t)
 #Z_Iner = sol[:, 2]
 
 return X_rot, Y_rot, Z_rot


def GetLibrationPointsCoord(mu):
 ##########################################
 #Implements the equation of motion for the Three Body problem
 #Args:
	# mu: mass parameter of the binary system
 #Returns:
	# df:   dataframe with coordinates
 ##########################################
 def collinear_lagrange(xstar, mu):
  # returns the location of the collinean Lagrange points (L1, L2 or L3)
  first_term = xstar
  second_term = (1 - mu) / np.abs(xstar + mu)**3 * (xstar + mu)
  third_term = mu / np.abs(xstar - 1 + mu)**3 * (xstar - 1 + mu)
  return first_term - second_term - third_term
 
 l1x = newton(func=collinear_lagrange, x0=0, args=(mu,))
 l2x = newton(func=collinear_lagrange, x0=1, args=(mu,))
 l3x = newton(func=collinear_lagrange, x0=-1, args=(mu,))
 l4x = 0.5 - mu
 l4y = np.sqrt(3)/2
 l5x = 0.5 - mu
 l5y = -np.sqrt(3)/2

 data = [["L1", l1x, 0, 0], ["L2", l2x, 0, 0], ["L3", l3x, 0, 0], ["L4", l4x, l4y, 0], ["L5", l5x, l5y, 0]]
 df = pd.DataFrame(data,columns=["Point","x","y","z"])
 return df
 
 
def GetJacobiConstant(state, mu):
 # returns the value of the Jacobi constant given a nondimensional state
 x = state[0]
 y = state[1]
 z = state[2]
 x_dot = state[3]
 y_dot = state[4]
 z_dot = state[5]
 r1=math.sqrt((x+mu)**2+y**2+z**2)
 r2=math.sqrt((x-1+mu)**2+y**2+z**2)
 U=0.5*(x**2+y**2)+((1-mu)/r1+mu/r2)
 J=2*U - (x_dot**2+y_dot**2+z_dot**2)
 return J 


################################################## ORBIT REPRESENTATION CONVERSION ###################################################


def KeplerianToStateVector(kep, mu):
    ##########################################
    #Converts the keplerian elements to position and velocity vector
    #Args:
    # kep(numpy array): a numpy array: np.array([[sma], [ecc], [inc], [aop], [raan], [ta]])
    # kep(0): semi major axis (km)
    # kep(1): eccentricity (number)
    # kep(2): inclination (degrees)
    # kep(3): argument of perigee (degrees)
    # kep(4): right ascension of the ascending node (degrees)
    # kep(5): true anomaly (degrees)
    #Returns:
    #    numpy array: 1x6 matrix which contains the position and velocity vector
    #    position vector (x,y,z) km
    #    velocity vector (vx,vy,vz) km/s
    ##########################################

    r = np.zeros((6, 1))

    # unload orbital elements array
    sma = kep[0]
    ecc = kep[1]
    inc = kep[2]
    inc = math.radians(inc)
    argper = kep[3]
    argper = math.radians(argper)
    raan = kep[4]
    raan = math.radians(raan)
    tanom = kep[5]
    tanom = math.radians(tanom)

    slr = sma * (1 - ecc * ecc)
    rm = slr / (1 + ecc * math.cos(tanom))

    arglat = argper + tanom  # argument of latitude

    sarglat = math.sin(arglat)
    carglat = math.cos(arglat)

    c4 = math.sqrt(mu / slr)
    c5 = ecc * math.cos(argper) + carglat
    c6 = ecc * math.sin(argper) + sarglat

    sinc = math.sin(inc)
    cinc = math.cos(inc)

    sraan = math.sin(raan)
    craan = math.cos(raan)

    # position vector
    x = rm * (craan * carglat - sraan * cinc * sarglat)
    y = rm * (sraan * carglat + cinc * sarglat * craan)
    z = rm * sinc * sarglat

    # velocity vector
    vx = -c4 * (craan * c6 + sraan * cinc * c5)
    vy = -c4 * (sraan * c6 - craan * cinc * c5)
    vz = c4 * c5 * sinc

    return np.array([x, y, z, vx, vy, vz])


def StateVectorToKeplerian(vec, mu):
    ##########################################
    #Converts state vector to orbital elements.
    #Args:
    #    vec (numpy array): state vector

    #Returns:
    #    numpy array: array of the computed keplerian elements
    #    kep(0): semimajor axis (kilometers)
    #    kep(1): orbital eccentricity (non-dimensional)
    #             (0 <= eccentricity < 1)
    #    kep(2): orbital inclination (degrees)
    #    kep(3): argument of perigee (degrees)
    #    kep(4): right ascension of ascending node (degrees)
    #    kep(5): true anomaly (degrees)
    ##########################################
    
    r = np.array([vec[0], vec[1],vec[2]])
    v = np.array([vec[3], vec[4],vec[5]])
    mag_r = np.sqrt(r.dot(r))
    mag_v = np.sqrt(v.dot(v))

    h = np.cross(r, v)
    mag_h = np.sqrt(h.dot(h))

    e = ((np.cross(v, h)) / mu) - (r / mag_r)
    mag_e = np.sqrt(e.dot(e))

    n = np.array([-h[1], h[0], 0])
    mag_n = np.sqrt(n.dot(n))

    true_anom = math.acos(np.clip(np.dot(e,r)/(mag_r * mag_e), -1, 1))
    if np.dot(r, v) < 0:
        true_anom = 2 * math.pi - true_anom
    true_anom = math.degrees(true_anom)

    i = math.acos(np.clip(h[2] / mag_h, -1, 1))
    i = math.degrees(i)

    ecc = mag_e

    raan = math.acos(np.clip(n[0] / mag_n, -1, 1))
    if n[1] < 0:
        raan = 2 * math.pi - raan
    raan = math.degrees(raan)

    per = math.acos(np.clip(np.dot(n, e) / (mag_n * mag_e), -1, 1))
    if e[2] < 0:
        per = 2 * math.pi - per
    per = math.degrees(per)

    a = 1 / ((2 / mag_r) - (mag_v**2 / mu))

    if i >= 360.0:
        i = i - 360
    if raan >= 360.0:
        raan = raan - 360
    if per >= 360.0:
        per = per - 360

    kep = np.zeros(6)
    kep[0] = a
    kep[1] = ecc
    kep[2] = i
    kep[3] = per
    kep[4] = raan
    kep[5] = true_anom
    return kep

def keplerian_to_state(keplerian, mu):
    """
    Convert Keplerian orbital elements to Cartesian state vector (position and velocity).
    
    Parameters:
    keplerian (np.array): Keplerian elements [a, e, i, Omega, omega, nu] with a in km, angles in deg.
    mu (float): Gravitational parameter in km^3/s^2 (default for Earth).
    
    Returns:
    np.array: State vector [x, y, z, vx, vy, vz] in km and km/s.
    
    Assumptions: Elliptical orbit (0 <= e < 1).
    Handles special cases like equatorial (i=0) or circular (e=0) orbits.
    """
    a, e, i, Omega, omega, nu = keplerian

    if e < 0 or e >= 1:
        raise ValueError("Eccentricity must be 0 <= e < 1 for elliptical orbit.")
    if a <= 0:
        raise ValueError("Semi-major axis must be positive.")

    # Convert angles to radians
    i_rad = np.radians(i)
    Omega_rad = np.radians(Omega)
    omega_rad = np.radians(omega)
    nu_rad = np.radians(nu)

    # Semi-latus rectum
    p = a * (1 - e**2)

    # Magnitude of position
    r_mag = p / (1 + e * np.cos(nu_rad))

    # Position in perifocal frame (PQW)
    r_pqw = r_mag * np.array([np.cos(nu_rad), np.sin(nu_rad), 0])

    # Velocity in perifocal frame
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu_rad), e + np.cos(nu_rad), 0])

    # Rotation matrices
    def rotz(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    def rotx(angle):
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

    # Rotation matrix from PQW to ECI: Rz(Omega) * Rx(i) * Rz(omega)
    R = rotz(Omega_rad) @ rotx(i_rad) @ rotz(omega_rad)

    # Transform to ECI
    r_eci = R @ r_pqw
    v_eci = R @ v_pqw

    # Combine into state vector
    state = np.concatenate((r_eci, v_eci))

    return state

def state_to_keplerian(state, mu):
    """
    Convert Cartesian state vector (position and velocity) to Keplerian orbital elements.
    
    Parameters:
    state (np.array): State vector [x, y, z, vx, vy, vz] in km and km/s.
    mu (float): Gravitational parameter in km^3/s^2 (default for Earth).
    
    Returns:
    tuple: (a, e, i, Omega, omega, nu)
        a: Semi-major axis (km)
        e: Eccentricity
        i: Inclination (deg)
        Omega: RAAN (deg)
        omega: Argument of periapsis (deg)
        nu: True anomaly (deg)
    
    Assumptions: Elliptical orbit (e < 1, negative energy).
    Handles special cases like equatorial or circular orbits by setting undefined elements to 0.
    """
    r = state[:3]
    v = state[3:]

    r_mag = np.linalg.norm(r)
    if r_mag == 0:
        raise ValueError("Position vector magnitude is zero.")

    v_mag = np.linalg.norm(v)

    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    if h_mag == 0:
        raise ValueError("Angular momentum is zero (collinear position and velocity).")

    # Specific orbital energy
    energy = v_mag**2 / 2 - mu / r_mag
    if energy >= 0:
        raise ValueError("Orbit is not bound (energy non-negative).")

    # Semi-major axis
    a = -mu / (2 * energy)

    # Eccentricity vector
    e_vec = np.cross(v, h) / mu - r / r_mag
    e = np.linalg.norm(e_vec)
    if e >= 1:
        raise ValueError("Orbit is not elliptical (e >= 1).")

    # Inclination
    i = np.arccos(h[2] / h_mag)

    # Node vector
    k = np.array([0, 0, 1])
    n = np.cross(k, h)
    n_mag = np.linalg.norm(n)

    # Right Ascension of the Ascending Node (RAAN)
    if n_mag == 0:
        Omega = 0.0  # Undefined for equatorial orbit
    else:
        Omega = np.arctan2(n[1], n[0])
        if Omega < 0:
            Omega += 2 * np.pi

    # Argument of periapsis
    if e == 0:
        omega = 0.0  # Undefined for circular orbit
    elif n_mag == 0:
        # Equatorial orbit: measure from x-axis
        omega = np.arctan2(e_vec[1], e_vec[0])
        if omega < 0:
            omega += 2 * np.pi
    else:
        cos_omega = np.dot(n, e_vec) / (n_mag * e)
        sin_omega = np.dot(np.cross(n, e_vec), h) / (n_mag * e * h_mag)
        omega = np.arctan2(sin_omega, cos_omega)
        if omega < 0:
            omega += 2 * np.pi

    # True anomaly
    if e == 0:
        # Circular orbit: true anomaly as argument of latitude
        if n_mag == 0:
            # Equatorial circular: from x-axis
            nu = np.arctan2(r[1], r[0])
        else:
            cos_nu = np.dot(n, r) / (n_mag * r_mag)
            sin_nu = np.dot(np.cross(n, r), h) / (n_mag * r_mag * h_mag)
            nu = np.arctan2(sin_nu, cos_nu)
        if nu < 0:
            nu += 2 * np.pi
    else:
        cos_nu = np.dot(e_vec, r) / (e * r_mag)
        sin_nu = np.dot(np.cross(e_vec, r), h) / (e * r_mag * h_mag)
        nu = np.arctan2(sin_nu, cos_nu)
        if nu < 0:
            nu += 2 * np.pi

    return a, e, np.degrees(i), np.degrees(Omega), np.degrees(omega), np.degrees(nu)

def equinoctial_to_keplerian(state, fr=1):
    """
    Convert equinoctial elements to keplerian
    fr = 1 for prograde orbitd (standard), fr = -1 for retrograde.
    """

    a = state[0]
    h = state[1]
    k = state[2]
    p = state[3]
    q = state[4]
    l = state[5]

    # 1. Eccentricità (e)
    e = np.sqrt(h**2 + k**2)
    
    # 2. Inclinazione (i)
    # Da q = tan(i/2) * cos(Omega) e p = tan(i/2) * sin(Omega)
    tan_i_half = np.sqrt(p**2 + q**2)
    i = 2 * np.arctan(tan_i_half)
    
    # 3. Longitudine del Nodo Ascendente (Omega)
    # Se l'inclinazione è zero, Omega è convenzionalmente 0
    if tan_i_half > 1e-12:
        omega_node = np.arctan2(p, q)
    else:
        omega_node = 0.0
        
    # 4. Argomento del Periasse (omega)
    # varpi = Omega + omega -> omega = varpi - Omega
    varpi = np.arctan2(h, k)
    omega_peri = (varpi - (fr * omega_node))
    
    # 5. Anomalia Media (M)
    # lamba = M + Omega + omega -> M = lamba - varpi
    m_anomaly = (l - varpi)

    # Normalizzazione degli angoli tra 0 e 2pi
    omega_node = np.mod(omega_node, 2 * np.pi)
    omega_peri = np.mod(omega_peri, 2 * np.pi)
    m_anomaly = np.mod(m_anomaly, 2 * np.pi)

    return {
        "Semimajor axis (km)": a,
        "Eccentricity ": e,
        "Inclination (deg)": np.degrees(i),
        "RAAN (deg)": np.degrees(omega_node),
        "Arg of Perigee (deg)": np.degrees(omega_peri),
        "Mean anomaly (deg)": np.degrees(m_anomaly)
    }

def keplerian_to_equinoctial(state, fr = 1):
    """
    Converte gli elementi kepleriani in elementi equinoziali.
    
    Parametri:
    a         : Semiasse maggiore
    e         : Eccentricità
    i_deg     : Inclinazione (in gradi)
    Omega_deg : Longitudine del nodo ascendente (in gradi)
    omega_deg : Argomento del periasse (in gradi)
    M_deg     : Anomalia media (in gradi)
    fr = 1 for prograde orbitd (standard), fr = -1 for retrograde.
    """

    a = state[0]
    e = state[1]
    i_deg = state[2]
    omega_deg = state[4]
    Omega_deg = state[3]
    M_deg = state[5]

    # Conversione in radianti
    i = np.radians(i_deg)
    Omega = np.radians(Omega_deg)
    omega = np.radians(omega_deg)
    M = np.radians(M_deg)
    
    # 1. Semiasse maggiore (rimane invariato)
    a_eq = a
    
    # 2. Calcolo di h e k (Vettore Eccentricità nel piano equinoziale)
    # varpi = Omega + omega (Longitudine del periasse)
    varpi = Omega + omega
    h = e * np.sin(varpi)
    k = e * np.cos(varpi)
    
    # 3. Calcolo di p e q (Vettore Inclinazione)
    # Nota: per orbite retrograde si usa tan(i/2)^fr, qui gestiamo il caso standard
    p = np.tan(i / 2)**fr * np.sin(Omega)
    q = np.tan(i / 2)**fr * np.cos(Omega)
    
    # 4. Longitudine Media (lambda)
    # lambda = M + Omega + omega
    lamba = M + varpi
    
    # Normalizzazione di lambda tra 0 e 2pi
    lamba = np.mod(lamba, 2 * np.pi)
    
    return {
        "a": a_eq,
        "h": h,
        "k": k,
        "p": p,
        "q": q,
        "lambda_deg": np.degrees(lamba)
    }

def CoeToStateVector(coe, mu):
	##########################################
	#Calculate the state vector from classical orbital elements
	#Args:    
	#		h     - Specific angular momentum
	#       e     - eccentricity
	#       i     - orbital inclination
	#       omega - right ascension of the ascending node
	#       w     - argument of perigee
	#       theta - true anomaly
	#       mu    - gravitational parameter
	#
	#Returns:
	#	    r     - The position vector
	#       v     - The velocity vector
	##########################################
	
    # Create the rotation matrix about the x-axis (equation 4.32)
    def rot1(th):
        c, s = np.cos(th), np.sin(th)
        rot1 = np.array([ [1,  0, 0],
                          [0,  c, s],
                          [0, -s, c] ])
        return rot1
    # Create the rotation matrix about the z-axis (equation 4.34)
    def rot3(th):
        c, s = np.cos(th), np.sin(th)
        rot3 = np.array([ [ c, s, 0],
                          [-s, c, 0],
                          [ 0, 0, 1] ])
        return rot3    

    # unpack data
    h, e, omega, i, w, theta, *others = coe    
    # Calculate the position vector in the perifocal frame (equation 4.45)
    rp = (h**2/mu) * ( 1/(1+e*np.cos(theta)) ) * np.array([np.cos(theta), np.sin(theta), 0]).T
    # Calculate the velocity vector in the perifocal frame (equation 4.46)
    vp = (mu/h) * np.array([-np.sin(theta), e + np.cos(theta), 0]).T   
    # Calculate the transform matrix from perifocal to geocentric (equation 4.49)
    Q  = (rot3(w) @ rot1(i) @ rot3(omega)).T
    # Transform from perifocal to geocentric (equations 4.51 - r and v are column vectors)
    r = Q @ rp.T
    v = Q @ vp.T
    # return position, velocity
    return r, v


######################################################### INTERPLANETARY #############################################################

def GetPlanetStateVector(planet_id, jd):
    ##########################################
    #Calculate the orbital elements and the state vector of a planet
    #Args:	
    #   planet_id - the planet number (0 - 8)
    #   jd        - julian day number of the date and time
    #Returns:
    #   coe       - classical orbit elements
    #   r         - position vector
    #   v         - velocity vector  
    #   jd        - JD of the state vector
    ##########################################

    mu = 1.32712440018e11 # Sun gravitational parameter km^3/sec^2

    # returns the inverse tangent (tan^-1) of the elements of X in degrees.
    atand = lambda x: np.rad2deg(np.arctan(x))

    # J2000 orbital elements 
    elements = GetPlanetEphemeris(planet_id, jd)
    a      = elements[0]
    e      = elements[1]
    # angular momentum - equation 2.71:
    h      = np.sqrt(mu*a*(1 - e**2))   
    # reduce the angular elements to within the range 0 - 360 degrees
    incl   = elements[2];
    RA     = np.mod(elements[3],360);
    w_hat  = np.mod(elements[4],360);
    L      = np.mod(elements[5],360);
    w      = np.mod(w_hat - RA ,360);
    M      = np.mod(L - w_hat  ,360);
    # Algorithm 3.1 (for which M must be in radians)
    E      = SolveKepler(e, np.radians(M)); # in rad
    # equation 3.13 (converting the result to degrees):
    TA     = np.mod(2*atand(np.sqrt((1 + e)/(1 - e))*np.tan(E/2)), 360)     
    coe    = [h, e, np.radians(RA), np.radians(incl), np.radians(w),
              np.radians(TA),  a,  w_hat,  L,  M,  E]
    # calculate state vectors from orbital elements - algorithm 4.5:
    [r, v] = CoeToStateVector(coe, mu)
    # return orbital elements(coe), state vectors(r,v) and julian date
    return coe, r, v, jd


def GetPlanetEphemeris(planet_id, jd):
    ##########################################
    #Extract planet's J2000 orbital elements and centennial rates from Table.
    #Args:
    # planet_id      - 0 through 8, for Mercury through Pluto
    # J2000_elements - 9 by 6 matrix of J2000 orbital elements for the nine
    #                planets Mercury through Pluto. The columns of each 
    #                row are: [a, e, i, RA, w_hat, L]

    #Returns
    ##########################################

    #orbital J2000_elements - 9 by 6 matrix of J2000 orbital elements for the nine planets
	#The columns of each row are: [a, e, i, RA, w_hat, L]
	
    J2000_elements = np.array((
    [0.38709927,  0.20563593,  7.00497902,  48.33076593,  77.45779628,  252.25032350],
    [0.72333566,  0.00677672,  3.39467605,  76.67984255, 131.60246718,  181.97909950], 
    [1.00000261,  0.01671123, -0.00001531,   0.0,        102.93768193,  100.46457166], 
    [1.52371034,  0.09339410,  1.84969142,  49.55953891, -23.94362959, 	-4.55343205],
    [5.20288700,  0.04838624,  1.30439695, 100.47390909,  14.72847983, 	34.39644501],
    [9.53667594,  0.05386179,  2.48599187, 113.66242448,  92.59887831, 	49.95424423],
    [19.18916464,  0.04725744,  0.77263783,  74.01692503, 170.95427630,  313.23810451],
    [30.06992276,  0.00859048,  1.77004347, 131.78422574,  44.96476227,  -55.12002969], 
    [39.48211675,  0.24882730, 17.14001206, 110.30393684, 224.06891629,  238.92903833]
    ))  
    J2000_coe = J2000_elements[planet_id]

    # cent_rates     - 9 by 6 matrix of the rates of change of the J2000_elements per Julian century (Cy)
	
    cent_rates = np.array(( 
    [0.00000037,  0.00001906, -0.00594749, -0.12534081,  0.16047689,  149472.67411175], 
    [0.00000390, -0.00004107, -0.00078890, -0.27769418,  0.00268329,  58517.81538729],  
    [0.00000562, -0.00004392, -0.01294668,  0.0,         0.32327364,   35999.37244981],  
    [0.0001847,   0.00007882, -0.00813131, -0.29257343,  0.44441088,   19140.30268499],  
    [-0.00011607, -0.00013253, -0.00183714,  0.20469106, 0.21252668,    3034.74612775], 
    [-0.00125060, -0.00050991,  0.00193609, -0.28867794, -0.41897216,    1222.49362201],
    [-0.00196176, -0.00004397, -0.00242939,  0.04240589,  0.40805281, 	428.48202785], 
    [0.00026291,  0.00005105,  0.00035372, -0.00508664, -0.32241464, 	218.45945325], 
    [-0.00031596,  0.00005170,  0.00004818, -0.01183482, -0.04062942, 	145.20780515]
    ))
    rates = cent_rates[planet_id]
    
    # Convert from AU to km:
    au             = 149597871; 
    J2000_coe[0]   = J2000_coe[0]*au;
    rates[0]       = rates[0]*au;

    # t0 - Julian centuries between J2000 and jd (equation 8.93a)
    t0     = (jd - 2451545)/36525
    # Equation 8.93b:
    elements = J2000_coe + rates*t0
    # return orbital elements
    return elements

	
def GetPlanetMu(planetName):
    ##########################################
    #Gets the gravitational parameter of the selected planet
    #Args:
    #    planetName: one of the allowed name
    #Returns:
    #    mu: gravitational parameter of the selected planet (km^3/sec^2)
    #    RETURNS -1 IF WRONG PLANET NAME
    ##########################################
	
    # planets list  
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']

    radius = -1   
    if planetName == 'Sun':
        mu = 1.32712440018e11
    if planetName == 'Mercury':
        mu = 2.20319e4
    if planetName == 'Venus':
        mu = 3.24859e5
    if planetName == 'Earth':
        mu = 3.986004418e5
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


def GetPlanetRadius(planetName):
    ##########################################
    #Gets the radius of the selected planet
    #Args:
    #    planetName: one of the allowed names
    #Returns:
    #    radius: radius of the selected planet (km)
    #    RETURNS -1 IF WRONG PLANET NAME
    ##########################################
	
    # planets list  
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']

    radius = -1
    
    if planetName == 'Mercury':
        radius = 2.44e3
    if planetName == 'Venus':
        radius = 6.051e3
    if planetName == 'Earth':
        radius = 6.378e3
    if planetName == 'Mars':
        radius = 3.397e3
    if planetName == 'Jupiter':
        radius = 7.1492e4
    if planetName == 'Saturn':
        radius = 6.0268e4          
    if planetName == 'Uranus':
        radius = 2.5559e4 
    if planetName == 'Neptune':
        radius = 2.4764e4  
    if planetName == 'Pluto':
        radius = 1.16e3
    return radius
	
	
def GetPlanetOrbitalRadius(planetName):
    ##########################################
    #Gets the orbit radius of the selected planet
    #Args:
    #    planetName: one of the allowed name
    #Returns:
    #    radius: orbital radius of the selected planet (km)
    #    RETURNS -1 IF WRONG PLANET NAME
    ##########################################
    
	# planets list  
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter',
               'Saturn', 'Uranus', 'Neptune', 'Pluto']

    radius = -1
    
    if planetName == 'Mercury':
        radius = 57.9e6
    if planetName == 'Venus':
        radius = 108.2e6
    if planetName == 'Earth':
        radius = 149.6e6
    if planetName == 'Mars':
        radius = 227.9e6
    if planetName == 'Jupiter':
        radius = 779.3e6
    if planetName == 'Saturn':
        radius = 1427e6          
    if planetName == 'Uranus':
        radius = 2871e6  
    if planetName == 'Neptune':
        radius = 4497e6   
    if planetName == 'Pluto':
        radius = 5913e6
    return radius
	
	
def GetPlanetId(planet_name):
    ##########################################
    #Gets the planet ID
    #Args:
    #    planetName: one of the allowed names
    #Returns:
    #    ind: ID of the selected planet (0 - 8)
    #    RETURNS -1 IF WRONG PLANET NAME
    ########################################## 
	
	# planets list  
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
    try:
        # remove empty spaces and capitalizes the first character
        planet_name = planet_name.strip().capitalize()
        # find the index
        ind = planets.index(planet_name)
        return ind
    except ValueError:
        print( 'error: Planet \'', planet_name, '\' not in list.')        
        return -1
    	

##################################################### GENERAL ASTRODYNAMICS ###########################################################


def HohmannDeltaVs(mu, r1, r2):
    ##########################################
    #Calculates the Delta V at departing and arrival points of an Hohmann transfer
    #Args:
    #    mu: gravitational parameter
    #    r1: radius of the departure orbit (km)
    #    r2: radius of the arrival orbit   (km)
    #Returns:
    #    deltaV1, deltaV2 (km/sec)
    ##########################################
    deltaV1 = math.sqrt(mu/r1)*(math.sqrt(2*r2/(r1+r2))-1)
    deltaV2 = math.sqrt(mu/r2)*(1-math.sqrt(2*r1/(r1+r2)))

    return deltaV1, deltaV2


def HohmannTransferTime(mu, r1, r2):
    ##########################################
    #Calculates the duration of an Hohmann transfer
    #Args:
    #    mu: gravitational parameter
    #    r1: radius of the departure orbit (km)
    #    r2: radius of the arrival orbit   (km)
    #Returns:
    #    t: transfer duration
    ##########################################
    sma = (r1+r2)/2
    return math.pi*math.sqrt(math.pow(sma,3)/mu)


def SolveKepler(e, M):
    # This function uses Newton's method to solve Kepler's 
    # equation  E - e*sin(E) = M  for the eccentric anomaly,
    # given the eccentricity and the mean anomaly.
    # 
    # E  - eccentric anomaly (radians)
    # e  - eccentricity, 
    # M  - mean anomaly (radians)
	
    # Set an error tolerance:
    error = 1.e-8
    # select a starting value for E:
    if M < np.pi: E = M + e/2
    else:         E = M - e/2
    # iterate on Equation 3.17 until E is determined to within
    # the error tolerance:
    ratio = 1
    while np.abs(ratio) > error:
        ratio = (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
        E = E - ratio
    # return eccentric anomaly
    return E


def GetMeanMotionFromPeriod(period):
    ##########################################
    #Args:
    #    period: orbit period (sec)
    #Returns:
     #   mean motion (rad/sec)        
    ##########################################
    return 2*math.pi/period


def GetMeanMotionFromSma(sma, mu):
    ##########################################
    #Args:
    #    sma: semimajor axis (km)
    #   mu: gravitational parameter (km^3/sec^2)
    #Returns:
    #    mean motion (rad/sec)        
    ##########################################
    return math.sqrt(mu/math.pow(sma,3))


def GetPeriodFromSma(sma, mu):
    ##########################################
    #Args:
    #   sma: semimajor axis (km)
    #   mu: gravitational parameter (km^3/sec^2)
    #Returns:
    #   period (sec)        
    ##########################################
    return 2*math.pi*math.sqrt(math.pow(sma,3)/mu)


def GetFlightPathAngle(theta, ecc):
    ##########################################
    #Args:
    #   theta: true anomaly (deg)
    #   ecc: eccentricity
    #Returns:
    #   Flight path angle (deg)        
    ##########################################
    #return math.atan((ecc*math.sin(math.radians(theta)))/(1+ecc*math.cos(math.radians(theta))))
    return math.atan2(ecc*math.sin(math.radians(theta)), (1+ecc*math.cos(math.radians(theta))))

def JdFromGregorianDateTime(year, month, day, hour=0, minute=0, second=0):
    # gregorian date and time to julian day number
	
	# julian day number at 0 UT for any year between 1900 and 2100 using Equation 5.48
    j0 = 367*year - np.fix(7*(year + np.fix((month + 9)/12))/4)  + np.fix(275*month/9) + day + 1721013.5   
    # ut - universal time in fractions of a day
    ut     = (hour + minute/60 + second/3600)/24
    # jd - julian day number of the date and time (equation 5.47)
    jd     = j0 + ut
    # return julian day number
    return jd
	
	
def JdFromGregorianDdMmYyyy(ddmmyyyy):
    # gregorian date and time to julian day number
	# input format shall be dd-mm-yyyy
	
    date   = ddmmyyyy.split("-")
    day    = int(date[0])
    month  = int(date[1])
    year   = int(date[2])
    hour   = 0
    minute = 0
    second = 0
	
	
	# julian day number at 0 UT for any year between 1900 and 2100 using Equation 5.48
    j0 = 367*year - np.fix(7*(year + np.fix((month + 9)/12))/4)  + np.fix(275*month/9) + day + 1721013.5   
    # ut - universal time in fractions of a day
    ut     = (hour + minute/60 + second/3600)/24
    # jd - julian day number of the date and time (equation 5.47)
    jd     = j0 + ut
    # return julian day number
    return jd


def GregorianDateTimeFromJd(jd):
    # convert a decimal Julian Date to gregorian date and time
	
	# convert jdn to gdt    
    jdn = int(jd)
    L = jdn + 68569
    N = int(4 * L / 146_097)
    L = L - int((146097 * N + 3) / 4)
    I = int(4000 * (L + 1) / 1_461_001)
    L = L - int(1461 * I / 4) + 31
    J = int(80 * L / 2447)
    day = L - int(2447 * J / 80)
    L = int(J / 11)
    month = J + 2 - 12 * L
    year = 100 * (N - 49) + I + L    
    # decimal processing
    offset = timedelta(days=(jd % 1), hours=+12)
    dt = datetime(year=year, month=month, day=day, tzinfo=timezone.utc)
    return dt + offset


def DateStringFromJd(jd):
    # create date string from julian day number
	
	#return GregorianDateTimeFromJd(np.float32(jd)).strftime('%d-%m-%Y')   
	return GregorianDateTimeFromJd(float(jd)).strftime('%d-%m-%Y')   


def NodalRegression(alt, inc, ecc):
    ##########################################
	#Calculate the nodal regression due to J2
    #Args:
    #   alt: satellite altitude (km)
    #   inc: inclination (deg)
	#   ecc: eccentricity
    #Returns:
    #   nodal regression: deg per sidereal day        
    ##########################################
	
   meanEarthRadius = 6378.15
   j2 = 0.0010826267
   mu = GetPlanetMu('Earth')
   
   incRad = inc*math.pi/180
   sma = np.array(alt) + meanEarthRadius 
   regRadPerSec = -(3/2)*j2*np.power(meanEarthRadius/(sma*(1-math.pow(ecc,2))),2)*np.sqrt(mu/np.power(sma,3))*np.cos(incRad)
   return regRadPerSec * 57.2958 * 86400 
   

def ApsidalRotation(alt, inc, ecc):
   ##########################################
   #Calculated the apsidal rotation due to J2
   #Args:
   #   alt: satellite altitude (km)
   #   inc: inclination (deg)
   #   ecc: eccentricity
   #Returns:
   #   apsidal rotation: deg per sidereal day        
   ##########################################
   
   meanEarthRadius = 6378.15
   j2 = 0.0010826267
   mu = GetPlanetMu('Earth')
   incRad = inc*math.pi/180
   sma = np.array(alt) + meanEarthRadius  
   apsRot = -(3/2)*j2*np.power(meanEarthRadius/(sma*(1-math.pow(ecc,2))),2)*np.sqrt(mu/np.power(sma,3))*(2 - (5/2)*(np.sin(incRad)**2))
   return apsRot * 57.2958 * 86164 # deg per sidereal day  
   
   
 ##################################################### ORBIT TRANSFERS ###########################################################
def calculate_hohmann_delta_v(r1, r2):
    if r1 >= r2:
        raise ValueError("Initial radius should be less than final radius")
    
    mu = GetPlanetMu('Earth')
    
    v1_circ = np.sqrt(mu / r1)
    v2_circ = np.sqrt(mu / r2)
    
    a_h = (r1 + r2) / 2
    
    dv1 = v1_circ * (np.sqrt(2 * r2 / (r1 + r2)) - 1)
    dv2 = v2_circ * (1 - np.sqrt(2 * r1 / (r1 + r2)))
    
    total_dv = dv1 + dv2
    return total_dv
    
def calculate_bielliptic_delta_v(r1, r2, rb):
    if r1 >= r2 or r2 >= rb:
        raise ValueError("Must have r1 < r2 < rb")
    
    mu = GetPlanetMu('Earth')
    
    a1 = (r1 + rb) / 2
    a2 = (r2 + rb) / 2
    
    # First burn at r1
    dv1 = np.sqrt(mu / r1) * (np.sqrt(2 * rb / (r1 + rb)) - 1)
    
    # Velocities at rb
    v1_rb = np.sqrt(mu * (2 / rb - 1 / a1))
    v2_rb = np.sqrt(mu * (2 / rb - 1 / a2))
    dv2 = v2_rb - v1_rb  # Should be positive
    
    # Velocity at r2 in second ellipse
    v2_r2 = np.sqrt(mu * (2 / r2 - 1 / a2))
    v2_circ = np.sqrt(mu / r2)
    dv3 = v2_r2 - v2_circ  # Magnitude, since deceleration but we take positive
    
    total_dv = dv1 + dv2 + dv3
    return total_dv
