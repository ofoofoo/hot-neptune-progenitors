# smaller number of outer templates
# keep using progressive/systematic 


import rebound
import numpy as np
import matplotlib.pyplot as plt
import random
import pdb
import pickle
from rebound import hash as h
from spock import FeatureClassifier
from spock import DeepRegressor


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


def perfect_merger(sim_pointer, collided_particles_index):
  # if the boolean in orbitalelementshash is True, that means that the particle is still in the sim, otherwise, False
    sim = sim_pointer.contents
    ps = sim.particles
    print("Time: " + str(sim.t))
    i = collided_particles_index.p1   # p1 < p2 is not guaranteed.    
    j = collided_particles_index.p2 
    for p in range(len(ps)):
        print("Particle " + str(p) + " mass: " + str(sim.particles[p].m))
    hashi = ps[i].hash.value
    hashj = ps[j].hash.value
    print(hashi)
    print(hashj)
    op = rebound.OrbitPlot(sim, color=True)
    op1 = rebound.OrbitPlot(sim, xlim = [-1, 1], ylim = [-1, 1], color = True)
    print("Merging particle {} into {}".format(j, i))
    op.ax.text(ps[i].x, ps[i].y, str(i))
    op.ax.text(ps[j].x, ps[j].y, str(j))
    op1.ax.text(ps[i].x, ps[i].y, str(i))
    op1.ax.text(ps[j].x, ps[j].y, str(j))

    # Merging 
    total_mass = ps[h(hashi)].m + ps[h(hashj)].m
    merged_planet = (ps[h(hashi)] * ps[h(hashi)].m + ps[h(hashj)] * ps[h(hashj)].m)/total_mass # conservation of momentum

    # merged radius assuming a uniform density
    merged_radius = (ps[h(hashi)].r**3 + ps[h(hashj)].r**3)**(1/3)

    ps[h(hashi)] = merged_planet   # update p1's state vector (mass and radius will need corrections)
    ps[h(hashi)].m = total_mass    # update to total mass
    ps[h(hashi)].r = merged_radius # update to joined radius
    if hashj in [h("p1").value, h("p2").value, h("p3").value, h("p4").value, h("p5").value]: # these are the hash arguments of the planets in the inner system
      multiplicity["num"] -= 1
    orbitalelementshash[hashj][1] = False # take the index j particle (larger index) out of the simulation
    return 2 # remove particle with index j



# SIMULATION SETUP

# FOR THE ARGUMENTS FOR THE TEMPLATE FUNCTIONS: roughly 2-3 collisiosn between inner systems sum to the inner planet of TOI-969
# ALSO: Use of SPOCK for narrowing down templates. Draw randomly from inner + outer and then do a SPOCK: if stable, quit
# SPOCK also has instability time estimator

def inner_template(n,m,a):
    sim = rebound.Simulation()
    sim.integrator = "mercurius"
    #sim.G = 6.6743e-11 # m^3 / kg s^2
    #sim.exit_max_distance = 500 * AU
    # sim.dt = 0.00547570157 / 20
    sim.ri_mercurius.safe_mode = 0 # turning safe mode off
    sim.ri_mercurius.hillfrac = 5
    sim.ri_whfast.safe_mode = 0 # turning whfast safe mode off
    # same as outer template!
    Mp_out = np.random.normal(m, m/4) * Mearth
    Mstar = 1*Msun
    sim.add(m = Mstar/Msun)
    A = 0.5*(2*Mp_out/(3*Mstar))**(1./3)
    #K = np.random.uniform(8, 10) # maybe also draw K from some random distribution? e.g. 4 to 6
    K = np.random.uniform(7, 12)
    a_iplus1_a_i = (1+K*A)/(1-K*A)
    a_list = np.zeros(n)
    m_list = np.zeros(n)
    r_list = np.zeros(n)
    e_list = np.zeros(n)
    omega_list = np.zeros(n)
    Omega_list = np.zeros(n)
    M_list = np.zeros(n)
    inc_list = np.zeros(n)
    a_list[round(n / 2) - 1] = np.random.normal(a, a/12)*(1-K*A)
    for i in range(round(n / 2) - 2, -1, -1):
#         print(i)
        a_list[i] = a_list[i + 1]/a_iplus1_a_i
    for i in range(round(n / 2), n):
        a_list[i] = a_list[i - 1]*a_iplus1_a_i
#     print(a_list)
    for i in range(n):
        m_list[i] = np.random.normal(m, m/4) * (Mearth/Msun)
        r_list[i] = 0.7 * m_list[i] ** (0.63) * (Rearth/AU)
        e_list[i] = e = np.random.rayleigh(scale = 0.035)
        omega_list[i] = random.uniform(0, 2 * np.pi)
        Omega_list[i] = random.uniform(0, 2 * np.pi)
        M_list[i] = random.uniform(0, 2 * np.pi)
        inc_list[i] = np.random.rayleigh(scale = 0.0175)
    for i in range(n):
        sim.add(m = m_list[i], 
                a = a_list[i], 
                hash = "inner_p" + str(i), 
                r = r_list[i], 
                e = e_list[i], 
                inc = inc_list[i], 
                omega = omega_list[i], 
                Omega = Omega_list[i], 
                M = M_list[i])
    ps = sim.particles
    return sim
    

def outer_template(simulation, n, m, a):
    # maybe make Mp_out the mass of the middle planet
    # draw masses directly for both the inner and outer system
    sim = simulation
    Mp_out = np.random.normal(m, m/4) * Mearth
    Mstar = 1*Msun
    A = 0.5*(2*Mp_out/(3*Mstar))**(1./3)
    #K = np.random.uniform(4, 6) # maybe also draw K from some random distribution? e.g. 4 to 6
    K = 4
    a_iplus1_a_i = (1+K*A)/(1-K*A)
    a_list = np.zeros(n)
    m_list = np.zeros(n)
    r_list = np.zeros(n)
    e_list = np.zeros(n)
    omega_list = np.zeros(n)
    Omega_list = np.zeros(n)
    M_list = np.zeros(n)
    inc_list = np.zeros(n)
    a_list[0] = random.uniform(a-0.25, a + 0.25)
    for i in range(1, n):
        a_list[i] = a_list[i - 1] * a_iplus1_a_i
    for i in range(n):
        m_list[i] = np.random.normal(m, m/4) *(Mearth/Msun)
        r_list[i] = 1 * Rjup/AU
        e_list[i] = random.uniform(0, 0.07)
        omega_list[i] = random.uniform(0, 2 * np.pi)
        Omega_list[i] = random.uniform(0, 2 * np.pi)
        M_list[i] = random.uniform(0, 2 * np.pi)
        inc_list[i] = random.uniform(0, 0.035)
    for i in range(n):
        sim.add(m = m_list[i], 
                a = a_list[i], 
                hash = "outer_p" + str(i), 
                r = r_list[i],
                e = e_list[i], 
                inc = inc_list[i], 
                omega = omega_list[i], 
                Omega = Omega_list[i], 
                M = M_list[i])
    sim.dt = (1/20)*np.amin([particle.P for particle in sim.particles[1::]])
    sim.ri_ias15.min_dt = 1e-2 * sim.dt
    sim.move_to_com()
    ps = sim.particles
    return sim


feature_model = FeatureClassifier()
deep_model = DeepRegressor()


# # consider n = 4,5 for outer templates


#4,5,6 outer planets maybe, ~3 potential values of sma (2,3,4), keep masses for now... (60, 100)
# indices are: number of planets, mass of planets (earth mass), minimum a (AU)
outer_templates = [[6, 100, 3], 
                  [6, 100, 4],
                  [6, 100, 5],
                  [5, 100, 3], 
                  [5, 100, 4],
                  [5, 100, 5],
                  [4, 100, 3], 
                  [4, 100, 4], 
                  [4, 100, 5],
                  [6, 60, 3], 
                  [6, 60, 4],
                  [6, 60, 5],
                  [5, 60, 3], 
                  [5, 60, 4],
                  [5, 60, 5],
                  [4, 60, 3], 
                  [4, 60, 4], 
                  [4, 60, 5]]

# calculating general masses from the  hot neptune planet diagram: minimum of the lower cloud is around 3.18 M_e, then 5.65 M_e, and higher bound 10.65 M_e, average of 6.5 M_e
# reading off periods/sma (assuming host star has 1 solar mass): min: 0.06 AU, mid: 0.09 AU, high: 0.19 AU: average is 0.11 AU
# start off with planets inthe lower cloud of the hot neptune desert diagram; have them collide up into the desert
inner_templates = [[3, 6, 0.05],
                   [3, 6, 0.1],
                   [3, 6, 0.15],
                   [4, 6, 0.05],
                   [4, 6, 0.1],
                   [4, 6, 0.15],
                   [5, 6, 0.05],
                   [5, 6, 0.1],
                   [5, 6, 0.15],
                   [6, 6, 0.05],
                   [6, 6, 0.1],
                   [6, 6, 0.15]]


in_random = random.choice(inner_templates)
print(in_random)
out_random = random.choice(outer_templates)
print(out_random)
sim = outer_template(inner_template(in_random[0], in_random[1], in_random[2]), out_random[0], out_random[1], out_random[2])
#sim = outer_template(inner_template(3, 3, 0.02), 4, 1750, 2)
#sim.status()
median, lower, upper = deep_model.predict_instability_time(sim, samples=10000)
print(int(median))
stability_p = feature_model.predict_stable(sim)
print(stability_p)
if median > 0 and stability_p < 1:
    multiplicity = {"num": 5}
    integrationtime = 10000000
    time_between_outputs = 0.1
    n_between_writeouts = int((integrationtime / time_between_outputs) / 1000)
    #print(n_between_writeouts)
    Noutputs = int(integrationtime / time_between_outputs)
    #print(Noutputs / n_between_writeouts)
    sim.collision = "direct"
    ps = sim.particles
    num_particles = len(sim.particles) # includes star
    times = np.linspace(0, integrationtime, Noutputs)
    orbitalelementshash = {}
    for i in range(len(ps)):
    #print(ps[i])
        orbitalelementshash[ps[i].hash.value] = [np.zeros((5, int(Noutputs / n_between_writeouts) - 1)), True]
    del orbitalelementshash[0] # getting rid of the star   
    #print(orbitalelementshash)
    sim.collision_resolve_keep_sorted = 1
    
    sim.collision_resolve = perfect_merger # user defined collision resolution function`
    j = 0
    for i,time in enumerate(times):
        sim.integrate(time)
        #orbits = sim.calculate_orbits()
        #print(sim.t)
        if i % n_between_writeouts == 0 and i != 0: # writeout to .npy periodically every n times we update orbitalelements (note that this != every n timesteps!)
            for a in range(len(ps) - 1):
                if ps[h(ps[a + 1].hash.value)].a > 1000:
                    orbitalelementshash[ps[a + 1].hash.value][1] = False # remove any particle that is more than 1000 AU away
            for a in range(len(ps) - 1):
                if ps[h(ps[a + 1].hash.value)].a < 0: # remove any particles that get a negative sma
                    orbitalelementshash[ps[a + 1].hash.value][1] = False
            # check for if any sma go beyond like 500 AU or something
                if orbitalelementshash[ps[a + 1].hash.value][1]:
                    orbitalelementshash[ps[a + 1].hash.value][0][0][j] = time
                    orbitalelementshash[ps[a + 1].hash.value][0][1][j] = ps[h(ps[a + 1].hash.value)].a
                    orbitalelementshash[ps[a + 1].hash.value][0][2][j] = ps[h(ps[a + 1].hash.value)].e
                    orbitalelementshash[ps[a + 1].hash.value][0][3][j] = ps[h(ps[a + 1].hash.value)].inc
                    orbitalelementshash[ps[a + 1].hash.value][0][4][j] = ps[h(ps[a + 1].hash.value)].m
                    #orbitalelementshash[ps[a + 1].hash.value][0][4][i] = ps[h(ps[a + 1].hash.value)].omega
                    #orbitalelementshash[ps[a + 1].hash.value][0][5][i] = ps[h(ps[a + 1].hash.value)].Omega
                    #orbitalelementshash[ps[a + 1].hash.value][0][6][i] = ps[h(ps[a + 1].hash.value)].M
                else:
                    orbitalelementshash[ps[a + 1].hash.value][0][0][j] = time
                    orbitalelementshash[ps[a + 1].hash.value][0][1][j] = 0
                    orbitalelementshash[ps[a + 1].hash.value][0][2][j] = 0
                    orbitalelementshash[ps[a + 1].hash.value][0][3][j] = 0
                    orbitalelementshash[ps[a + 1].hash.value][0][4][j] = 0
                    #orbitalelementshash[ps[a + 1].hash.value][0][4][i] = 0
                    #orbitalelementshash[ps[a + 1].hash.value][0][5][i] = 0
                    #orbitalelementshash[ps[a + 1].hash.value][0][6][i] = 0
            j += 1
          #  np.save(filename, orbitalelementshash)
          #print(len(ps))
    
    #with open('multiplicity_' + str(ID) + '.pickle', 'wb') as handle2:
    #  pickle.dump(multiplicity["num"], handle2, protocol=pickle.HIGHEST_PROTOCOL)
    #print(orbitalelementshash)
    #print(multiplicity["num"])
    with open('orbitalelementshash_' + str(ID) + '.pickle', 'wb') as handle2:
        pickle.dump(orbitalelementshash, handle2, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(out_random, handle2, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(in_random, handle2, protocol=pickle.HIGHEST_PROTOCOL)
else:
    print("not unstable!!!")
