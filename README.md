# Tools for EMTF pT assignment studies


## Background
The Large Hadron Collider currently collides bunches of protons every 25ns at the Compact Muon Solenoid (CMS) experiment. Only a small fraction of these collisions result in interesting events which can be used for physics analyses. Saving everything would mean saving several pedabytes of data every second, so the CMS detector must make quick decisions about what data to keep, and what to discard.

The CMS trigger system attempts to identify and save the most interesting 1000 events out of the 40 million candidates which occur every second. The Level 1 (L1) trigger is implemented in firmware, capable of dealing with high volume, and reduces the rate from 40MHz to 100KHz. These selected events are then sent to the High Level Trigger (HLT), which does a much deeper analysis and reconstruction of the data to choose 1000 most interesting events.

The Endcap Muon Track Finder is part of the L1 trigger. Its job is to identify interesting muons. In most subsystems, interesting particles are particles with high transverse momentum ($p_T$) as these particles are likely the decay products of the heavy particles which CMS analyses most commonly look for. In the case of the Endcap Muon system, we look for single muons with $p_T>22 GeV$, or two muons each with $p_T>8 GeV$.

To do this, EMTF must use the barely-processed information from the detector to determine the trajectory of muons, and then their momentum -- _at 40MHz_. 

CMS uses a superconducting solenoid magnet to create a strong magnetic field along the beamline. This magnetic field bends the trajectories of charged particles traveling purpendicular to the beamline (transversely). The greater the particle's transverse momentum, the less its trajectory will bend in the magnetic field. This is the principle used by all the muon systems to determine the $p_T$ of a muon. The straighter the trajectory, the higher the $p_T$.

We cannot implement an anylitical algorithm to predict the momentum of muons based on the trajectory in firmware at 40MHz. The best solution is to precompute predictions for transverse momentum for many possible muon trajectories, and put the results in a lookup table (LUT). This way momentum can be determined with a single operation. 

Ideally we would have a separate address in the LUT for every possible muon trajectory, and use all of the available information to make a momentum determination, but the hardware on the detector limits us to 30 bits of address space ($2^{30}=1073741824$ possible trajectories can be encoded). One major technical puzzle of EMTF is to determine exactly which 30 bits of information will have the most discriminating power for $p_T$. 

_Part of the purpose of this codebase is to study which bits are most important._

Then there is the question of how exactly to populate the lookup table. As physicists, we know how to predict what will happen to the trajectory of a muon in a magnetic field anylitically, and this is done in the barrel region of the muon system (BMTF). However, the endcap lies at the end of the solenoid magnet, and the magnetic field does hideous things in this region. It is possible to model, but we cannot account for everything. It turns out that using machine learning to predict $p_T$ leads to much better performance. 

_Part of the purpose of this codebase is to test machine learning models and measure their performance._
