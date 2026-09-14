import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime
import astrodynamics as ad
import lambert as lb


# global constants
mu = 1.32712428e11    # gravitational parameter of the sun (km^3/s^2)


def PlotPorkchop(dep_planet, arr_planet, first_dep_date, last_dep_date, first_arr_date, last_arr_date, plot_type):
	##########################################
	#Main function to create Porkchop plots
	#Args:
	#	dep_planet:     departure planet name
	#	arr_planet:     arrival planet name
	#	first_dep_date: first departure date in yyyy-mm-dd format
	#	last_dep_date:  last departure date in yyyy-mm-dd format
	#	first_arr_date: first arrival date in yyyy-mm-dd format
	#	last_arr_date:  last arrival date in yyyy-mm-dd format
	#	plot_type:      variable to plot ('delv_plot' or 'c3_plot')
	#Returns:
	#	Porkchop plot 
	##########################################
    try:
	 #test the date format
     #test = datetime.datetime.strptime(first_dep_date, '%Y-%m-%d')
     #test = datetime.datetime.strptime(last_dep_date, '%Y-%m-%d')
     #test = datetime.datetime.strptime(first_arr_date, '%Y-%m-%d')
     #test = datetime.datetime.strptime(last_arr_date, '%d-%m-%d')
     test = datetime.datetime.strptime(first_dep_date, '%d-%m-%Y')
     test = datetime.datetime.strptime(last_dep_date, '%d-%m-%Y')
     test = datetime.datetime.strptime(first_arr_date, '%d-%m-%Y')
     test = datetime.datetime.strptime(last_arr_date, '%d-%m-%Y')
    except ValueError:
     #print ('error : wrong date format. [verify as yyyy-mm-dd]')
     print ('error : wrong date format. [verify as dd-mm-yyyy]')
    else:   
	 # derive dates in JD format and time spans
     #dt_first_dep = datetime.datetime.strptime(first_dep_date, '%Y-%m-%d')
     dt_first_dep = datetime.datetime.strptime(first_dep_date, '%d-%m-%Y')
     first_d, first_m, first_y = dt_first_dep.day, dt_first_dep.month, dt_first_dep.year 
     jd_first_dep = ad.JdFromGregorianDateTime(first_y, first_m, first_d)
     #dt_last_dep = datetime.datetime.strptime(last_dep_date, '%Y-%m-%d')
     dt_last_dep = datetime.datetime.strptime(last_dep_date, '%d-%m-%Y')
     last_d, last_m, last_y = dt_last_dep.day, dt_last_dep.month, dt_last_dep.year 
     jd_last_dep = ad.JdFromGregorianDateTime(last_y, last_m, last_d)
    
     #dt_first_arr = datetime.datetime.strptime(first_arr_date, '%Y-%m-%d')
     dt_first_arr = datetime.datetime.strptime(first_arr_date, '%d-%m-%Y')
     first_d, first_m, first_y = dt_first_arr.day, dt_first_arr.month, dt_first_arr.year 
     jd_first_arr = ad.JdFromGregorianDateTime(first_y, first_m, first_d)
     #dt_last_arr = datetime.datetime.strptime(last_arr_date, '%Y-%m-%d')
     dt_last_arr = datetime.datetime.strptime(last_arr_date, '%d-%m-%Y')
     last_d, last_m, last_y = dt_last_arr.day, dt_last_arr.month, dt_last_arr.year 
     jd_last_arr = ad.JdFromGregorianDateTime(last_y, last_m, last_d)
    
     dep_time_span = jd_last_dep - jd_first_dep
     arr_time_span = jd_last_arr - jd_first_arr
    
	 # generate date matrix
     jd_first_dep_list, jd_first_arr_list = _generate_date_matrix_(jd_first_dep, jd_first_arr, dep_time_span, arr_time_span)

     # mu, dep_planet, jd_first_dep, arr_planet, jd_first_arr
     res = _generate_porkchop_plot_data_(dep_planet, jd_first_dep_list, arr_planet, jd_first_arr_list)   
     jd_first_dep_str_list, jd_first_arr_str_list, tof_days_list, c3_dep_1_list, c3_dep_2_list, delv_t_1_list, delv_t_2_list = res
    
     # contour levels    
     c3_levels = [4, 5, 6, 8, 10, 12, 14, 16, 18, 19, 20, 30, 50]
     t_levels  = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    
     # plot    
     if plot_type == 'delv_plot':
        title = 'Porkchop plot (∆V Total = ||$ΔV_{' +dep_planet+'}|| + ||ΔV_{'+ arr_planet +'}$||)' + '\n'
        plot_porkchop(title, jd_first_dep_str_list, jd_first_arr_str_list,
                         c3_dep_1_list, c3_dep_2_list, c3_levels,
                         tof_days_list, t_levels)
     elif plot_type == 'c3_plot':       
        title = 'Porkchop plot (C3-characteristic energy = $v_{id}^{2}$)' + '\n'
        plot_porkchop(title, jd_first_dep_str_list, jd_first_arr_str_list,
                         delv_t_1_list, delv_t_2_list, c3_levels,
                         tof_days_list, t_levels)
     else: raise Exception("error: plot types should be c3_plot or delv_plot")   
	


def _get_lambert_estimates_(v_dep, v_arr, r1, r2, flight_time_secs, orb_type, M, path):
	##########################################
	# Calculate c3 and DeltaV values from lambert solution 
	##########################################
	
	# Solve the problem
    # multiple-solution for m>0 cases not considered here
    v1_list, v2_list = lb.solve(mu, r1, r2, flight_time_secs, orb_type, path, M)
    v1, v2 = v1_list[0], v2_list[0]
     
    # compute v_inf for departure and arrival (subtract planet velocities)
    v_inf_dep = np.linalg.norm(v_dep - v1) 
    v_inf_arr = np.linalg.norm(v_arr - v2)
	
    # characteristic energy. v_inf = orbital velocity when the orbital distance tends to infinity.
    c3_dep = v_inf_dep**2 
    c3_arr = v_inf_arr**2
	
    # ∆V = Vearth(t1) − VT(t1) + VMars (t2) − VT (t2)
    delv_total = v_inf_dep + v_inf_arr    

    return [c3_dep, delv_total]


def _get_porkchop_plot_data_(dep_planet, jd_first_dep, arr_planet, jd_first_arr):
	##########################################
	# Get plot data for a departure arrival combination
	# Args:
	# 	dep_planet:   departure planet name		
	# 	jd_first_dep: departure date as JD
	# 	arr_planet:   arrival planet name
	# 	jd_first_arr: arrival date as JD
    # Returns:
    #    jd_first_dep_str: departure date as string
	# 	jd_first_arr_str: arrival date as string
	# 	flight_time_days: flight time in days
	# 	c3_and_delv_1:    energy and DeltaV at departure
	# 	c3_and_delv_2:    energy and DeltaV at arrival
	# 	coe_dep:          classical elements at departure
	# 	coe_arr:          classical elements at arrival
	##########################################
	
    # Departure and arrival planets id
    dep_planet_id = ad.GetPlanetId(dep_planet)
    arr_planet_id = ad.GetPlanetId(arr_planet)    
    # get state vector
    coe_dep, r_dep, v_dep, jd_d = ad.GetPlanetStateVector(dep_planet_id, jd_first_dep)
    coe_arr, r_arr, v_arr, jd_a = ad.GetPlanetStateVector(arr_planet_id, jd_first_arr)
    # jd string
    jd_first_dep_str, jd_first_arr_str = ad.DateStringFromJd(jd_first_dep), ad.DateStringFromJd(jd_first_arr)
    # time of flight, convert to seconds for lambert's function
    flight_time_days = (jd_first_arr - jd_first_dep)
    flight_time_secs = flight_time_days * (24.0 * 60.0 * 60.0)
    
    # lambert estimation (type-I and type-II)
    orb_type, M, low_path = 'prograde', 0.0, 'low'
    c3_and_delv_1 = _get_lambert_estimates_(v_dep, v_arr, r_dep, r_arr,
                                           flight_time_secs, orb_type, M, low_path)
    
    orb_type, M, low_path = 'prograde', 0.0, 'high'
    c3_and_delv_2 = _get_lambert_estimates_(v_dep, v_arr, r_dep, r_arr,
                                           flight_time_secs, orb_type, M, low_path)
    #return
    out = jd_first_dep_str, jd_first_arr_str, flight_time_days, \
          c3_and_delv_1, c3_and_delv_2, coe_dep, coe_arr
    return out


def _generate_porkchop_plot_data_(dep_planet, jd_first_dep_list, arr_planet, jd_first_arr_list):
	##########################################
	# Get porkchop plot contour data
	# Args:
	# 	dep_planet:        departure planet name		
	# 	jd_first_dep_list: array of departure dates in JD
	# 	arr_planet:        arrival planet name
	# 	jd_first_arr_list: array of arrival dates in JD
    # Returns:
    #     jd_first_dep_str_list: list of departure date as string
	# 	jd_first_arr_str_list: list of arrival date as string
	# 	flight_time_days_list: list of flight time in days
	# 	c3_dep_1_list:         list of energy at departure
	# 	c3_dep_2_list:         list of energy at arrival
	# 	delv_t_1_list:         list of DeltaV at departure
	# 	delv_t_2_list:         list of DeltaV at arrival
	##########################################
	
    # initialize contour lists
    jd_first_dep_str_list = np.empty(jd_first_dep_list.shape, dtype=object)
    jd_first_arr_str_list = np.empty(jd_first_arr_list.shape, dtype=object)
    # 2d array shape   
    contour_shape = (jd_first_arr_list.shape[0], jd_first_dep_list.shape[0])
    tof_days_list = np.zeros( contour_shape, dtype=np.float64)
    c3_dep_1_list = np.zeros( contour_shape, dtype=np.float64)
    c3_dep_2_list = np.zeros( contour_shape, dtype=np.float64)
    delv_t_1_list = np.zeros( contour_shape, dtype=np.float64)
    delv_t_2_list = np.zeros( contour_shape, dtype=np.float64)
    # generate result matrix
    rows, cols = tof_days_list.shape[0], tof_days_list.shape[1]
    for ix in range(0, rows):
        jd_first_arr_i = jd_first_arr_list[ix]
        for iy in range(0, cols):
            jd_first_dep_i = jd_first_dep_list[iy]
            # calculate plot data
            res = _get_porkchop_plot_data_(dep_planet, jd_first_dep_i, arr_planet, jd_first_arr_i)
            # unpack result
            jd_first_dep_str, jd_first_arr_str, flight_time_days, c3_and_delv_1, c3_and_delv_2, coe_dep, coe_arr = res
            c3_dep_1, delv_1_total = c3_and_delv_1
            c3_dep_2, delv_2_total = c3_and_delv_2
            # pack the list
            tof_days_list[ix][iy] = flight_time_days
            c3_dep_1_list[ix][iy] = c3_dep_1
            c3_dep_2_list[ix][iy] = c3_dep_2
            delv_t_1_list[ix][iy] = delv_1_total
            delv_t_2_list[ix][iy] = delv_2_total
            jd_first_arr_str_list[ix] = jd_first_arr_str
            jd_first_dep_str_list[iy] = jd_first_dep_str
        # end for iy
    #end for ix
    # return list
    out = jd_first_dep_str_list, jd_first_arr_str_list, tof_days_list, \
          c3_dep_1_list, c3_dep_2_list, \
          delv_t_1_list, delv_t_2_list
    return out



def _generate_date_matrix_(jd_first_dep, jd_first_arr, dep_time_span, arr_time_span):
	##########################################
	# Args:
	# 	jd_first_dep:  first departure date in JD format
	# 	jd_first_arr:  first arrival date in JD format
	# 	dep_time_span: duration of the departure window in days
	# 	arr_time_span: duration of the arrival window in days
    # Returns:
    #   jd_first_dep_list: array of departure dates
	# 	jd_first_arr_list: array of arrival dates
	##########################################
	
    # check for date range validity
    if (jd_first_arr - (jd_first_dep + dep_time_span)) < 0:         
        raise ValueError("error: the latest departure epoch is greater than the earliest arrival epoch.")

    # grid resolution of the porkchop plot
    dt_dep, dt_arr = 2, 5   
    # generate list of jd's
    jd_first_dep_list = np.array( list(range(int(jd_first_dep), int(jd_first_dep + dep_time_span), int(dt_dep) )) )
    jd_first_arr_list = np.array( list(range(int(jd_first_arr), int(jd_first_arr + arr_time_span), int(dt_arr) )) )
    # return date matrix
    return jd_first_dep_list, jd_first_arr_list


def plot_porkchop(title, xlist, ylist, xy_contour_data_1, xy_contour_data_2, clevels, xy_tof_data, tlevels):
    ##########################################
	# Plot porkchop on a figure
	##########################################
    def set_ticks(ax):
        # major grid
        x_tick_spacing, y_tick_spacing = 5, 3
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_spacing))
        # fontsize of major, minor ticks label
        ax.xaxis.set_tick_params(labelsize=7, rotation=90)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.set_xlabel("Dep Date (dd-mm-yyyy)", fontsize=8)        
        ax.set_ylabel("Arr Date (dd-mm-yyyy)", fontsize=8)

        # Customize the major grid
        ax.grid(which='major', linestyle='dashdot', linewidth='0.5', color='gray')
        # Customize the minor grid
        ax.grid(which='minor', linestyle='dotted', linewidth='0.5', color='gray')       
        # Turn on the minor ticks (minor grid)
        ax.minorticks_on()        
        # Turn off the display of all ticks.
        ax.tick_params(which='both',
                        top='off',
                        left='off',
                        right='off',
                        bottom='off')
        # end function
        return

    # countour text format ( include 'days')
    def tp_fmt(x):
        s = f"{x:.1f}"
        return rf"{s} days"
    
    # init figure    
    plt.figure(figsize=(10,9))

    # find the minimum value with the corresponding dep, arr dates
    # Type-I minimum
    # draw ∆V contours
    cp1 = plt.contour(xlist, ylist, xy_contour_data_1, clevels, cmap="rainbow")
    plt.clabel(cp1, inline=True, fontsize=7)

    # find the minimum value with the corresponding dep, arr dates
    # type-II
    # draw ∆V contours
    cp2 = plt.contour(xlist, ylist, xy_contour_data_2, clevels, cmap="rainbow")
    plt.clabel(cp2, inline=True, fontsize=7)
    
    # draw time-of-flight contours
    tp = plt.contour(xlist, ylist, xy_tof_data, tlevels, colors='k',linestyles=':')
    plt.clabel(tp, inline=True, fmt=tp_fmt, fontsize=7)
    
    # set title and x, y labels
    plt.title(title, fontdict={'fontsize':8})
    
    # set grids and ticks
    set_ticks(plt.gca())
    
    # set aspect ratio and layout
    plt.gca().set_aspect(1) #'auto')   
    plt.tight_layout()
    # show
    plt.show()
    return




