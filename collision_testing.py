# -*- coding: utf-8 -*-
"""Collision Testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1osjjX1ohuPdWQnV1r0QLoNSRA4mjBF_c
"""
import rebound
import numpy as np
import matplotlib.pyplot as plt
import random
import pdb

G = 6.67408*10**(-11)   # gravitational constant 
Mearth = 5.9722*10**24  # kg
Rearth = 6.371*10**6  # meters
Msun = 1.98911*10**30 # kg
Rsun = 6.95*10**8 # meters
Mjup = 1.898*10**27  # kg
Rjup = 7.1492*10**7 # meters
year = 365.25*24*60*60  # years in seconds
AU = (G*Msun*(year**2)/(4.*np.pi**2))**(1./3.)  # meters

import sys
ID = int(sys.argv[1])

def setupSimulation1():
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.G = 6.6743e-11 # m^3 / kg s^2
    #sim.exit_max_distance = 500 * AU
    sim.dt = 24*3600*100
    # sim.ri_ias15.min_dt = 1e-3 * sim.dt
    # sim.ri_mercurius.hillfrac = 4
    sim.add(m=0.62 * Msun)
    sim.add(m=3.15*Mearth, a=0.029 * AU, r=1.59*Rearth, e = 0, inc = np.random.rayleigh(scale = 0.0175), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi)) # we now set collision radii!
    sim.add(m=3.02*Mearth, a=0.048 * AU, r=1.55*Rearth, e = 0, inc = np.random.rayleigh(scale = 0.0175), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    # sim.add(m=3.02*Mearth, a=0.5 * AU, r=1.55*Rearth, e = np.random.rayleigh(scale = 0.035), inc = np.random.rayleigh(scale = 0.0175), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))

    # sim.add(m=3.15*Mearth, a=0.081 * AU, r=1.59*Rearth, e = np.random.rayleigh(scale = 0.035), inc = np.random.rayleigh(scale = 0.0175), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    # sim.add(m=43*Mearth, a=0.1593 * AU, r=2.43*Rearth, e = np.random.rayleigh(scale = 0.035), inc = np.random.rayleigh(scale = 0.0175), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    # sim.add(m=69.8*Mearth, a=0.2095 * AU, r=2.21*Rearth, e = np.random.rayleigh(scale = 0.035), inc = np.random.rayleigh(scale = 0.0175), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    # sim.add(m=60*Mearth, a=5.55 * AU, r=1*Rjup, e = random.uniform(0, 0.07), inc = random.uniform(0, 0.035), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    # sim.add(m=60*Mearth, a=7.01 * AU, r=1*Rjup, e = random.uniform(0, 0.07), inc = random.uniform(0, 0.035), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    # sim.add(m=60*Mearth, a=8.84 * AU, r=1*Rjup, e = random.uniform(0, 0.07), inc = random.uniform(0, 0.035), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    # sim.add(m=60*Mearth, a=11.16 * AU, r=1*Rjup, e = random.uniform(0, 0.07), inc = random.uniform(0, 0.035), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    # sim.add(m=60*Mearth, a=14.08 * AU, r=1*Rjup, e = random.uniform(0, 0.07), inc = random.uniform(0, 0.035), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    # sim.add(m=60*Mearth, a=17.76 * AU, r=1*Rjup, e = random.uniform(0, 0.07), inc = random.uniform(0, 0.035), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    sim.move_to_com()
    return sim

def setupSimulation2():
    sim = rebound.Simulation()
    sim.integrator = "mercurius"
    #sim.G = 6.6743e-11 # m^3 / kg s^2
    #sim.exit_max_distance = 500 * AU
    # sim.dt = 0.00547570157 / 20
    sim.ri_mercurius.safe_mode = 0 # turning safe mode off
    sim.ri_mercurius.hillfrac = 5
    sim.ri_whfast.safe_mode = 0 # turning whfast safe mode off
    sim.add(m=0.62)
    sim.add(m=3.15*Mearth/Msun, a=0.029, r = 1.59*Rearth/AU, e = 0, inc = np.random.rayleigh(scale = 0.0175), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi)) # we now set collision radii!
    sim.add(m=3.02*Mearth/Msun, a=0.048, r = 1.59*Rearth/AU, e = 0, inc = np.random.rayleigh(scale = 0.0175), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    # sim.add(m=3.02*Mearth, a=0.5 * AU, r=1.55*Rearth, e = np.random.rayleigh(scale = 0.035), inc = np.random.rayleigh(scale = 0.0175), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    sim.add(m=3.15*Mearth/Msun, a=0.081, r=1.55*Rearth/AU, e = np.random.rayleigh(scale = 0.035), inc = np.random.rayleigh(scale = 0.0175), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    sim.add(m=43*Mearth/Msun, a=0.1593, r=2.43*Rearth/AU, e = np.random.rayleigh(scale = 0.035), inc = np.random.rayleigh(scale = 0.0175), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    sim.add(m=69.8*Mearth/Msun, a=0.2095, r=2.21*Rearth/AU, e = np.random.rayleigh(scale = 0.035), inc = np.random.rayleigh(scale = 0.0175), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    sim.add(m=60*Mearth/Msun, a=5.55, r=1*Rjup/AU, e = random.uniform(0, 0.07), inc = random.uniform(0, 0.035), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    sim.add(m=60*Mearth/Msun, a=7.01, r=1*Rjup/AU, e = random.uniform(0, 0.07), inc = random.uniform(0, 0.035), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    sim.add(m=60*Mearth/Msun, a=8.84, r=1*Rjup/AU, e = random.uniform(0, 0.07), inc = random.uniform(0, 0.035), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    sim.add(m=60*Mearth/Msun, a=11.16, r=1*Rjup/AU, e = random.uniform(0, 0.07), inc = random.uniform(0, 0.035), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    sim.add(m=60*Mearth/Msun, a=14.08, r=1*Rjup/AU, e = random.uniform(0, 0.07), inc = random.uniform(0, 0.035), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    sim.add(m=60*Mearth/Msun, a=17.76, r=1*Rjup/AU, e = random.uniform(0, 0.07), inc = random.uniform(0, 0.035), omega = random.uniform(0, 2 * np.pi), Omega = random.uniform(0, 2 * np.pi), M = random.uniform(0, 2 * np.pi))
    sim.dt = (1/20)*np.amin([particle.P for particle in sim.particles[1::]])
    sim.ri_ias15.min_dt = 1e-2 * sim.dt
    sim.move_to_com()
    return sim

def perfect_merger(sim_pointer, collided_particles_index):
  orbitalelementschange["change"] = True
  orbitalelementslength["value"] -= 1
  # orbitalelements = np.load("orbitalelements.npy")
  #print(type(orbitalelements))
  #print(literal_eval(orbitalelements))
  #print(type(literal_eval(orbitalelements)))
  sim = sim_pointer.contents
  ps = sim.particles
  print("Time: " + str(sim.t / (60*60*24*365.25)))
  i = collided_particles_index.p1   # p1 < p2 is not guaranteed.    
  j = collided_particles_index.p2 
  if i > j: # Swap
    temp = j
    j = i
    i = temp
  if i < 6 - innersystemcollide["num"] and j - innersystemcollide["num"] < 6:
    innersystemnum["num"] -= 1 # If there is a collision between two planets in the inner system, decrement the multiplicity
    innersystemcollide["num"] += 1 
  for p in range(len(ps)):
      print("Particle " + str(p) + " mass: " + str(sim.particles[p].m))

  op = rebound.OrbitPlot(sim, color=True)
  op1 = rebound.OrbitPlot(sim, xlim = [-1, 1], ylim = [-1, 1], color = True)
  print("Merging particle {} into {}".format(j, i))
  #print("Particle 1: x: " + str(ps[1].x) + " Particle 1: y: " + str(ps[1].y))
  #print("Particle 2: x: " + str(ps[2].x) + " Particle 2: y: " + str(ps[2].y))
  # op.ax.title("Merging particle {} into {}".format(j, i))
  op.ax.text(ps[i].x, ps[i].y, str(i))
  op.ax.text(ps[j].x, ps[j].y, str(j))
  op1.ax.text(ps[i].x, ps[i].y, str(i))
  op1.ax.text(ps[j].x, ps[j].y, str(j))

  # Merging 
  #print(ps[i].m)
  #print(ps[i].v)
  #collision_energy = (min(ps[i].m, ps[j].m) * abs(ps[i].v - ps[j].v)**2) / (2 * (ps[i].m + ps[j].m))
  #print("Energy of collision between particle {} and {}: {}".format(i, j, collision_energy))
  total_mass = ps[i].m + ps[j].m
  merged_planet = (ps[i] * ps[i].m + ps[j] * ps[j].m)/total_mass # conservation of momentum

  # merged radius assuming a uniform density
  merged_radius = (ps[i].r**3 + ps[j].r**3)**(1/3)

  ps[i] = merged_planet   # update p1's state vector (mass and radius will need corrections)
  ps[i].m = total_mass    # update to total mass
  ps[i].r = merged_radius # update to joined radius
  #orbitalelements = orbitalelements.tolist()
  np.delete(orbitalelementsdict["orbitalelements"], j - 1) # remove the particle with index j from the orbital elements array
  #orbitalelementslength = orbitalelementslength - 1
  #print(orbitalelementslength)
  # orbitalelementsdict["orbitalelements"] = np.array(orbitalelementsdict["orbitalelements"]) 
  # np.save('orbitalelements', orbitalelements)

  # sim.remove(j) # remove particle with index j from the simulation

  return 2 # remove particle with index j

# Commented out IPython magic to ensure Python compatibility.
#------------------------------------------------COLLISION CASE-----------------------------------------------------------
# %matplotlib inline
finale = []
finala = []
finali = []

multiplicities = []
for _ in range(1):
  innersystemnum = {"num": 5}
  orbitalelementschange = {"change": False}
  innersystemcollide = {"num": 0}
  integrationtime = 10000000
  #time_step = 2.211*24*3600
  #Nsteps = int(integrationtime / time_step) # determines timestep
  time_between_outputs = 10000
  Noutputs = int(integrationtime / time_between_outputs)
  sim = setupSimulation2()
  sim.collision = "direct"
  ps = sim.particles
  num_pl = len(sim.particles)
  print(num_pl)
  times = np.linspace(0, integrationtime, Noutputs)
  print("bob")
  orbitalelements = np.zeros((num_pl - 1, 7, Noutputs))
  orbitalelementsdict = {"orbitalelements": orbitalelements} #number of parameters +1 because of output times
  orbitalelementsdict["orbitalelements"] = np.array(orbitalelementsdict["orbitalelements"])
  # orbitalelementslength = num_pl - 1

  sim.collision_resolve_keep_sorted = 1

  sim.collision_resolve = perfect_merger # user defined collision resolution function`

  # for p in range(len(ps)):
  #         #print(i)
  #         print("Particle " + str(p) + " mass: " + str(sim.particles[p].m))
  orbitalelementslength = {"value":len(orbitalelementsdict["orbitalelements"])}

  # with open('orbitalelements.txt', 'w') as f:
  #     f.write(str(orbitalelements))
  # f.close()
  # np.save('orbitalelements', orbitalelements)
  # print("Particles in the simulation at t=%6.1f: %d"%(sim.t,sim.N))
  # print("System Mass: {}".format([p.m for p in sim.particles]))
  for i,time in enumerate(times):
      # orbitalelements = np.load("orbitalelements.npy")
      # orbitalelements = orbitalelementsdict["orbitalelements"]
    sim.integrate(time)
    orbits = sim.calculate_orbits()
    print(sim.t/year)
    #print("dt: " + str(sim.dt))
    for a in range(orbitalelementslength['value']):
        orbitalelementsdict["orbitalelements"][a][0][i] = time
        orbitalelementsdict["orbitalelements"][a][1][i] = orbits[a].a
        orbitalelementsdict["orbitalelements"][a][2][i] = orbits[a].e
        orbitalelementsdict["orbitalelements"][a][3][i] = orbits[a].inc
        orbitalelementsdict["orbitalelements"][a][4][i] = orbits[a].omega
        orbitalelementsdict["orbitalelements"][a][5][i] = orbits[a].Omega
        orbitalelementsdict["orbitalelements"][a][6][i] = orbits[a].M
        # np.save('orbitalelements', orbitalelements)
    
  #print("testtttttt")
  #print(len(orbitalelements))
  #print("mass of orbitalelemetns index:")
  #print(orbitalelements)
  fig = plt.figure(figsize=(20,10))
  ax = plt.subplot(221)
  ax.set_xlabel("time")
  ax.set_ylabel("semi-major axis")
  ax.set_title("Semi-Major Axis")
  ax.set_yscale("log")
  plt.plot(times, orbitalelementsdict["orbitalelements"][0][1] / AU, label = "Planet 1");
  plt.plot(times, orbitalelementsdict["orbitalelements"][1][1] / AU, label = "Planet 2");
  plt.plot(times, orbitalelementsdict["orbitalelements"][2][1] / AU, label = "Planet 3");
  plt.plot(times, orbitalelementsdict["orbitalelements"][3][1] / AU, label = "Planet 4");
  plt.plot(times, orbitalelementsdict["orbitalelements"][4][1] / AU, label = "Planet 5");
  plt.plot(times, orbitalelementsdict["orbitalelements"][5][1] / AU, label = "Planet 6");
  plt.plot(times, orbitalelementsdict["orbitalelements"][6][1] / AU, label = "Planet 7");
  plt.plot(times, orbitalelementsdict["orbitalelements"][7][1] / AU, label = "Planet 8");
  plt.plot(times, orbitalelementsdict["orbitalelements"][8][1] / AU, label = "Planet 9");
  plt.plot(times, orbitalelementsdict["orbitalelements"][9][1] / AU, label = "Planet 10");
  plt.plot(times, orbitalelementsdict["orbitalelements"][10][1] / AU, label = "Planet 11");

  plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


  for p in range(len(ps)):
          #print(i)
          print("Particle " + str(p) + " mass: " + str(sim.particles[p].m))
  op = rebound.OrbitPlot(sim, periastron=True, color=True)
  for p in range(len(ps)):
    op.ax.text(ps[p].x, ps[p].y, str(p))
  # for p in range(len(ps) - 1):
  #   finale.append(orbitalelements[p][2][-1])
  #   finala.append(orbitalelements[p][1][-1])
  #   finali.append(orbitalelements[p][3][-1])
  #op1 = rebound.OrbitPlot(sim, xlim = [-1, 1], ylim = [-1, 1], color=True)
  print("Number of planets in the inner system remaining: " + str(innersystemnum["num"]))
  multiplicities.append(innersystemnum["num"])
print(orbitalelementsdict["orbitalelements"])
file1 = open("orbitalelements.txt", "w+")
filtered_orbitalelm = []
for i in range(len(orbitalelementsdict["orbitalelements"])):
  single_filter = np.delete((orbitalelementsdict["orbitalelements"][i]), np.arange(0, orbitalelementsdict["orbitalelements"][0].size, 5))
  filtered_orbitalelm.append(single_filter)
filtered_orbitalelm = np.array(filtered_orbitalelm)
# filtered_orbitalelm = np.delete(orbitalelementsdict["orbitalelements"], np.arange(0, orbitalelementsdict["orbitalelements"].size, 5)) # delete every 5th timestep
orbitalelementsstr = str(filtered_orbitalelm)
file1.write(orbitalelementsstr)
file1.close()
print(multiplicities)

