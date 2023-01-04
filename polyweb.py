#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import random as rnd
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.constants import Avogadro

class Reactions():
    """Class containing all the possible reactions represented as methods."""
    def __init__(self, a: int, b: int):

        # initial concentrations in units of molars (M), where M = mol/litres, and volume in units of litres
        self.conc_mono = 5 * 10**-3
        self.conc_init = [10**-6, 10**-5][a]
        self.conc_raft = [0, 10**-6, 10**-5][b]
        self.volume = 10**-12

        # initial number of species
        self.monomer = int(self.conc_mono * self.volume * Avogadro)
        self.initiator = self.initiator_0 = int(self.conc_init * self.volume * Avogadro)
        self.numT = int(self.conc_raft * self.volume * Avogadro)
        
        # Polymer radical lists
        self.Rn = []
        self.radTRn = []
        self.TRn = []
        self.adduct = []
        self.product = []
        
        self.reactions = [11, 21, 31, 32, 33, 41, 42, 43, 51, 52, 53, 54]
        
        # rate constant in units of 1/(hours * M)
        self.rate_constants = {
            "k11": 0.36,
            "k21": 3.6 * 10**7,
            "k31": 3.6 * 10**10,
            "k32": 18 * 10**8,
            "k33": 18 * 10**8,
            "k41": 3.6 * 10**10,
            "k42": 18 * 10**8,
            "k43": 18 * 10**8,
            "k51": 3.6 * 10**11,
            "k52": 3.6 * 10**11,
            "k53": 3.6 * 10**11,
            "k54": 3.6 * 10**11
        }
    
    def rxn11(self):
        """
        Initiation reaction, uses up one initiator and creates two R_0 radicals.
        𝐼 → 2𝑅0⋅
        """
        self.initiator -= 1
        self.Rn += [[1], [1]]
    
    def rxn21(self):
        """
        Propagation reaction, where a monomer reacts with a R_n radical to produce a R_n+1 radical.
        A "0" is appended to a random index in the R_n list, representing attaching the radical site
        to a random monomer unit in the chain.
        𝑅𝑛⋅ + 𝑀 → 𝑅𝑛+1⋅
        """
        random_chain = rnd.randint(len(self.Rn))
        self.Rn[random_chain].append(0)
        self.monomer -= 1
    
    # Pre-equilibrium reactions
    def rxn31(self):
        """
        Reaction of a living polymer radical with a RAFT agent to create a TR_n radical.
        𝑅𝑛⋅ + 𝑇 → ⋅𝑇𝑅𝑛
        """
        num_R1 = [i for i in self.Rn if len(i) > 1]
        rdm = rnd.randint(len(num_R1))
        x = [2] + num_R1[rdm] 
        self.radTRn.append(x)
        
        self.Rn.remove(num_R1[rdm])
        self.numT -= 1
    
    def rxn32(self):
        """
        Reverse of reaction 31, where a TR_n radical splits into a R_n radical and a RAFT agent.
        ⋅𝑇𝑅𝑛 → 𝑅𝑛⋅ + 𝑇
        """
        pp = rnd.randint(len(self.radTRn))
        u = self.radTRn[pp][1:] 
        self.Rn.append(u)
        
        del self.radTRn[pp]
        self.numT += 1
    
    def rxn33(self):
        """
        A TR_n radical ejects an R_0 radical and
        produces a terminated TR_n molecule chain.
        ⋅𝑇𝑅𝑛 → 𝑅0⋅ + 𝑇𝑅𝑛
        """
        random_target = rnd.randint(len(self.radTRn)) 
        self.TRn.append(self.radTRn[random_target])
        
        del self.radTRn[random_target]
        self.Rn.append([1])
    
    # Core equilibrium reactions
    def rxn41(self):
        """
        An R_n radical reacts with a TR_m molecule 
        to form the adduct R_n⋅TR_m.
        𝑅𝑛⋅ + 𝑇𝑅𝑚 → 𝑅𝑛⋅𝑇𝑅𝑚
        """
        num_R1 = [i for i in self.Rn if len(i) > 1]
        
        pp1 = rnd.randint(len(num_R1))
        pp2 = rnd.randint(len(self.TRn))
        
        self.adduct.append(num_R1[pp1] + self.TRn[pp2])
        self.Rn.remove(num_R1[pp1])
        del self.TRn[pp2]
        
    def rxn42(self):
        """
        Reverse of reaction 41, where an R_n⋅TR_m adduct splits into
        an R_n radical and a TR_m molecule.
        𝑅𝑛⋅𝑇𝑅𝑚 → 𝑅𝑛⋅ + 𝑇𝑅𝑚
        """
        pp = rnd.randint(len(self.adduct))
        x = self.adduct[pp].index(2)
        self.Rn.append(self.adduct[pp][:x]) 
        self.TRn.append(self.adduct[pp][x:])
        del self.adduct[pp]
    
    def rxn43(self):
        """
        An R_n⋅TR_m adduct splits into a TR_n molecule and a
        R_m⋅ radical.
        𝑅𝑛⋅𝑇𝑅𝑚 → 𝑇𝑅𝑛 + 𝑅𝑚⋅
        """
        pp = rnd.randint(len(self.adduct))
        x = self.adduct[pp].index(2)
        self.TRn.append([2] + self.adduct[pp][:x]) 
        self.Rn.append(self.adduct[pp][x+1:]) 
        del self.adduct[pp]
    
    # Termination reactions
    def rxn51(self):
        """
        An R_n radical and R_m radical combine to form a P_m+n polymer chain.
        𝑅𝑛⋅ + 𝑅𝑚⋅ → 𝑃𝑛+𝑚
        """
        num_R1 = [i for i in self.Rn if len(i) > 1]
        
        pp1 = rnd.randint(len(num_R1))
        pp2 = rnd.randint(len(num_R1) - 1)
        
        self.product.append(num_R1[pp1] + num_R1[pp2])
        self.Rn.remove(num_R1[pp1])
        self.Rn.remove(num_R1[pp2])
    
    def rxn52(self):
        """
        An R_n radical and R_m radical produce a P_n chain and a P_m chain.
        𝑅𝑛⋅ + 𝑅𝑚⋅ → 𝑃𝑛 + 𝑃𝑚
        """
        num_R1 = [i for i in self.Rn if len(i) > 1]
        
        pp1 = rnd.randint(len(num_R1))
        pp2 = rnd.randint(len(num_R1) - 1)
        
        self.product.append(num_R1[pp1])
        self.product.append(num_R1[pp2])
        
        self.Rn.remove(num_R1[pp1])
        self.Rn.remove(num_R1[pp2])

    def rxn53(self):
        """
        An R_n⋅TR_m adduct combines with an R_0 radical to form a P_m+n chain.
        𝑅𝑛⋅𝑇𝑅𝑚 + 𝑅0⋅ → 𝑃𝑛+𝑚
        """
        pp = rnd.randint(len(self.adduct))
        self.product.append(self.adduct[pp] + [1])
        self.Rn.remove([1])
        del self.adduct[pp]
    
    def rxn54(self):
        """
        An R_n⋅TR_m adduct combines with an R_s radical to form a P_m+n+s chain.
        𝑅𝑛⋅𝑇𝑅𝑚 + 𝑅𝑠⋅ → 𝑃𝑛+𝑚+𝑠
        """
        num_R1 = [i for i in self.Rn if len(i) > 1]
        
        pp1 = rnd.randint(len(num_R1))
        pp2 = rnd.randint(len(self.adduct))
        
        self.product.append(self.adduct[pp2] + num_R1[pp1])
        self.Rn.remove(num_R1[pp1])
        del self.adduct[pp2]

    def calc_rates(self) -> dict:
        """Returns a dictionary of each reaction and their rates of reaction."""
        numR0 = self.Rn.count([1])
        alpha = 1/(Avogadro*self.volume) # particle no. to molarity conversion factor [mol.litres^-1]
        self.ratedict = {
            11: 
                self.rate_constants["k11"] * 2 * self.initiator*alpha,
            21: 
                self.rate_constants["k21"] * (len(self.Rn)*alpha) * (self.monomer*alpha),
            31: 
                self.rate_constants["k31"] * (len(self.Rn) - numR0)*alpha * self.numT*alpha,
            32: 
                self.rate_constants["k32"] * len(self.radTRn)*alpha,
            33: 
                self.rate_constants["k33"] * len(self.radTRn)*alpha,
            41: 
                self.rate_constants["k41"] * (len(self.Rn) - numR0)*alpha * len(self.TRn)*alpha,
            42: 
                self.rate_constants["k42"] * len(self.adduct)*alpha,
            43: 
                self.rate_constants["k43"] * len(self.adduct)*alpha,
            51: 
                self.rate_constants["k51"] * (len(self.Rn) - numR0)*alpha * (len(self.Rn) - numR0-1)*alpha,
            52: 
                self.rate_constants["k52"] * (len(self.Rn) - numR0)*alpha * (len(self.Rn) - numR0-1)*alpha,
            53: 
                self.rate_constants["k53"] * len(self.adduct)*alpha *  numR0*alpha,
            54: 
                self.rate_constants["k54"] * len(self.adduct)*alpha * len(self.Rn)*alpha
        }
        return self.ratedict
    
    def run_reaction(self, reaction: int):
        """Runs the selected reaction if conditions match and updates the environment."""
        if reaction == 11 and self.initiator > 0 :
            self.rxn11()
        elif reaction == 21 and len(self.Rn) > 0 and self.monomer > 0:
            self.rxn21()
        elif reaction == 51 and (len(self.Rn) - self.Rn.count([1]) >= 2):
            self.rxn51()
        elif reaction == 52 and (len(self.Rn) - self.Rn.count([1]) > 2):
            self.rxn52()
        elif reaction == 31 and (len(self.Rn) - self.Rn.count([1]) >= 1) and self.numT > 0:
            self.rxn31()
        elif reaction == 32 and len(self.radTRn) > 0:
            self.rxn32()
        elif reaction == 33 and len(self.radTRn) > 0:
            self.rxn33()
        elif reaction == 41 and len(self.TRn) > 0 and (len(self.Rn) - self.Rn.count([1]) > 0):
            self.rxn41()
        elif reaction == 42 and len(self.adduct) > 0:
            self.rxn42()
        elif reaction == 43 and len(self.adduct) > 0:
            self.rxn43()
        elif reaction == 53 and [1] in self.Rn and len(self.adduct) > 0:
            self.rxn53()
        elif reaction == 54 and (len(self.Rn) - self.Rn.count([1]) > 0) and len(self.adduct) > 0:
            self.rxn54()

    def number_of_R(self, num: int) -> int:
        """Returns the number of R(num) radicals where num is the number of monomers."""
        length = len([radical for radical in self.Rn if len(radical) == num+1])
        return length

    def new_constants(self):
        """"
        Changes the values of k31, k32, k51, and their related 
        rate constants to double their current values.
        """
        self.rate_constants["k31"] = self.rate_constants["k41"] = 2 * self.rate_constants["k31"]
        
        self.rate_constants["k32"] = self.rate_constants["k33"] = 2 * self.rate_constants["k32"]
        self.rate_constants["k42"] = self.rate_constants["k43"] = 2 * self.rate_constants["k42"]
        
        self.rate_constants["k51"] = self.rate_constants["k52"] = 2 * self.rate_constants["k51"]
        self.rate_constants["k53"] = self.rate_constants["k54"] = 2 * self.rate_constants["k53"]
        
    def run_simulation(self, maxtime: int):
        """Runs the simulation until sum of reaction rates equal 0 
        or maximum simulation time is reached.
        """
        time = 0
        self.tenMult = 0
        self.width = []

        while time < maxtime:
            # calculate rates of each potential reaction and convert them to a numpy array
            rate_dictionary = self.calc_rates()
            rates = np.array(list(rate_dictionary.values()))

            if sum(rates) < 10**-8:
                print("Reactions stop as total rates equal 0.")
                break

            if self.initiator == self.initiator_0:
                self.run_reaction(11)
            else:
                # Choose a reaction from the reaction list with weights determined by their probabilities
                probabilities = rates / sum(rates)
                reaction = rnd.choice(self.reactions, p=probabilities)
                self.run_reaction(reaction)

            # Increment time by tau, the time passed from the last reaction
            rd = rnd.rand()
            tau = np.log(1 / rd) / sum(rates) / 3600 
            time += tau

            if time > self.tenMult * 10:
                max_length = max([len(chain) for chain in self.Rn]) - 1
                # List of probabilities of finding R_n over every other present chain length
                self.P_n = [self.number_of_R(n)/len(self.Rn) for n in range(max_length)]
                
                # Plot P(n) against n
                x = [i for i in range(len(self.P_n))]
                y = self.P_n
                plt.plot(x, y, 'g-', linewidth = 1.0)
                plt.xlabel('Chain length, n')
                plt.ylabel('Probability')
                plt.title(f'Probability against chain length: {time} s')
                
                # Store width values for plot
                n = np.array([i for i in range(len(self.P_n))])
                self.P_n = np.array(self.P_n)
                n_mean = np.mean(n)
                W = np.sum((n - n_mean)**2 * self.P_n)
                self.width.append(W)
                
                self.tenMult += 1
                
        plt.show()

        fig = plt.figure(figsize=(12, 5))
        fig.suptitle(f'Simulation x results')

        # Plot width against time
        time_label = [i for i in range(self.tenMult)]

        ax0 = fig.add_subplot(2, 2, 1)
        ax0.plot(time_label, self.width, 'r-', linewidth = 1.0)
        ax0.set_xlabel('Time / 10^1 s')
        ax0.set_ylabel('Width')
        ax0.set_title('Molecular weight distribution width against time')

        # Plot the probability density function and histogram of molecular weight
        chain_lengths = np.array([chain.count(0) for chain in self.Rn])
        std = np.std(chain_lengths, ddof=1)
        mean = np.mean(chain_lengths)
        domain = np.linspace(np.min(chain_lengths), np.max(chain_lengths))

        ax1 = fig.add_subplot(2, 2, 2)
        ax1.plot(domain, norm.pdf(domain,mean,std))
        ax1.hist(chain_lengths, bins=int(max(chain_lengths)), density=True)
        ax1.set_xlabel('Molecular weight')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Molecular weight distribution (polydispersity)')

        plt.show()
        
if __name__ == "__main__":
    rxndet = Reactions(a=0, b=0)
    for i in range(3):
        rxndet.run_simulation(maxtime=3600)
        rxndet.new_constants()