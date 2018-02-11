from constant_solution import *
from plug_wave import *
from standing_undamped_solution import *
from standing_damped_solution import *
from physical_solution import *

Lx      = 1           # Size in x-dir
dx      = 0.01        # Spatial step x-dir
Ly      = Lx          # Size in y-dir
dy      = dx          # Spatial step y-dir
T       = 0.5         # Total time
dt      = 0.0005      # Time step

test_constant           (Lx, Ly, T, dt, dx, dy)
test_pulse              (Lx, Ly, T, dt, dx, dy)
test_standing_undamped  (Lx, Ly, T, dt, dx, dy)
test_standing_damped    (Lx, Ly, T, dt, dx, dy)
physical('gaussian',     Lx, Ly, T, dt, dx, dy)
