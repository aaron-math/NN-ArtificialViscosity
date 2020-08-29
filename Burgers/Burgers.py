from pyranda import pyrandaSim, pyrandaBC
from matplotlib import pyplot as plt
import numpy as np

"""Initialize simulation"""
title = "burger"
xdom = "xdom = (0.0,1.0,200,periodic=True)"
pysim = pyrandaSim(title,xdom)
pysim.addPackage( pyrandaBC(pysim) )#Turn on the pyrandaBC package

"""Set equations of motion"""
eom = """
ddt(:u:) = - :u: * ddx(:u:) + :nu: * ddx(ddx(:u:))  # Viscous Burgers equation
:u: = fbar(:u:)                                     # Conservative filter to prevent high-frequency ringing
:dudx: = ddx( :u: )                                 # Velocity gradient
:nu: = .1 * gbar( ring( :dudx: ) )                  # Artificial viscosity model
"""
pysim.EOM(eom)

"""Evaluate the initial conditions and then update variables"""
ic = """
:u: = exp(-(meshx-.5)**2/(.1**2))"""
pysim.setIC(ic)

"""Integrate in time"""
dt = .001         # Reduced time step to adjust for stability.
time = 0.0

while time < 0.3:
    time = pysim.rk4(time,dt)
