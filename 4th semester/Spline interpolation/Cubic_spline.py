import numpy as np
from aux_func import thomas_for_spline, cs,gs

def cubic_spline(x,y,sd_a,sd_b):
    ms = thomas_for_spline(x,y,sd_a,sd_b)
    c_waves, c_s = cs(x,y,ms)
    spline_vals, ab = gs(x,ms,c_waves,c_s)
    return (spline_vals, ab)

     
