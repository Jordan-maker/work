from copy import copy as cp
import sys


class fmp:
    # final mother particles
    def __init__(self, BR_pdg):
        self.BR_pdg = BR_pdg
        self.prop = [i/sum(self.BR_pdg) for i in self.BR_pdg]  #create right proportions

class imp:
    # intermediate mother particles
    def __init__(self, BR_pdg):
        self.BR_pdg = [i[0] for i in BR_pdg]
        self.fmp = [i[1] for i in BR_pdg]

        factor=[1]*len(BR_pdg)
        self.BR_pdg_modified = cp(self.BR_pdg)
        for i in range(len(BR_pdg)):
            if self.fmp[i]:
                for particle in self.fmp[i]:
                    self.BR_pdg_modified[i] *= particle.BR_pdg[0]
                    factor[i]=particle.prop[0]

        normalized=[]
        for i in range(len(BR_pdg)):
            if self.fmp[i]:
                normalized.append(self.BR_pdg_modified[i]/(self.BR_pdg_modified[0]*factor[i]))
            else:
                normalized.append(self.BR_pdg_modified[i]/(self.BR_pdg_modified[0]))

        self.prop=[i/sum(normalized) for i in normalized]   #create right proportions


def main():

    #------ final mother particles --------#

    KstarPlus  = fmp(BR_pdg=[0.333003335])   # ->  K_L0  pi+
    Kstar2Plus = fmp(BR_pdg=[0.167000000])   # ->  K_L0  pi+

    antiDeltaPlus = fmp(BR_pdg=[1.00])        # anti-delta+ ->  anti-n0 pi+

    eta_c   = fmp(BR_pdg=[0.0013500])        # ->  anti-n0  n0
    eta_c2S = fmp(BR_pdg=[0.0010000])        # ->  anti-n0  n0

    #--------- intermediate mother particles --------#

    Jpsi = imp(BR_pdg=[(0.0020900, None),      # ->  n0  anti-n0
                       (0.0003080, None),      # ->  n0  anti-n0 gamma
                       (0.0170000, [eta_c]),   # ->  eta_csig gamma
                       ])

    chi_c0 = imp(BR_pdg=[(0.0002210, None),    # ->  n0  anti-n0
                         (0.0140000, [Jpsi]),  # ->  J/psisig  gamma
                         ])

    chi_c1 = imp(BR_pdg=[(0.0000760, None),    # ->  n0  anti-n0
                         (0.3430000, [Jpsi]),  # ->  J/psisig  gamma
                         ])

    chi_c2 = imp(BR_pdg=[(0.0000733, None),    # ->  n0  anti-n0
                         (0.1900000, [Jpsi]),  # ->  J/psisig  gamma
                         ])

    Psi2S = imp(BR_pdg=[(0.0003060, None),       # ->  n0  anti-n0
                        (0.0000390, None),       # ->  n0  anti-n0 gamma
                        (0.0979000, [chi_c0]),   # ->  chi_c0sig  gamma
                        (0.0975000, [chi_c1]),   # ->  chi_c1sig  gamma
                        (0.0952000, [chi_c2]),   # ->  chi_c2sig  gamma
                        ])

    tau = imp(BR_pdg=[(0.1082000, None),         # ->  pi+  anti-nu_tau
                      (0.0041900, None),         # ->  pi+  anti-nu_tau  K_L0
                      (0.0002350, None),         # ->  pi+  anti-nu_tau  K_L0  K_L0
                      (0.0120000, [KstarPlus]),  # ->  K*+  anti-nu_tau
                      ])

    #--------- B meson ---------#

    Bplus = imp(BR_pdg=[(0.00000007966667, None),                # ->  pi+  nu_e   anti-nu_e
                        (0.00000007966667, None),                # ->  pi+  nu_mu  anti-nu_mu
                        (0.00000007966667, None),                # ->  pi+  nu_tau anti-nu_tau
                        (0.00000306333333, [KstarPlus]),         # ->  K*+  nu_e   anti-nu_e
                        (0.00000306333333, [KstarPlus]),         # ->  K*+  nu_mu  anti-nu_mu
                        (0.00000306333333, [KstarPlus]),         # ->  K*+  nu_tau anti-nu_tau
                        (0.000001620, None),                     # ->  pi+  n0  anti-n0
                        (0.000003600, [KstarPlus]),              # ->  K*+  n0  anti-n0
                        (0.000109000, [tau]),                    # ->  tau+   nu_tau
                        (0.000039200, [Jpsi]),                   # ->  pi+  J/psi
                        (0.000022000, [chi_c1]),                 # ->  pi+  chi_c1
                        (0.000024400, [Psi2S]),                  # ->  pi+  Psi(2S)
                        (0.001430000, [KstarPlus, Jpsi]),        # ->  K*+  J/psi
                        (0.001200000, [KstarPlus, eta_c]),       # ->  K*+  eta_c
                        (0.000480000, [KstarPlus, eta_c2S]),     # ->  K*+  eta_c(2S)
                        (0.000140000, [KstarPlus, chi_c0]),      # ->  K*+  chi_c0
                        (0.000300000, [KstarPlus, chi_c1]),      # ->  K*+  chi_c1
                        (0.000120000, [KstarPlus, chi_c2]),      # ->  K*+  chi_c2
                        (0.000670000, [KstarPlus, Psi2S]),       # ->  K*+  Psi(2S)
                        (0.000011850, None),                     # ->  pi+  K_L0
                        (0.000570000, [Jpsi]),                   # ->  pi+  K_L0  J/psi
                        (0.000100000, [eta_c]),                  # ->  pi+  K_L0  eta_c
                        (0.000040000, [eta_c2S]),                # ->  pi+  K_L0  eta_c(2S)
                        (0.000100000, [chi_c0]),                 # ->  pi+  K_L0  chi_c0
                        (0.000290000, [chi_c1]),                 # ->  pi+  K_L0  chi_c1
                        (0.000058000, [chi_c2]),                 # ->  pi+  K_L0  chi_c2
                        (0.000071440, [Psi2S]),                  # ->  pi+  K_L0  Psi(2S)
                        (0.000500000, [Kstar2Plus, Jpsi]),       # ->  K_2*+  J/psi
                        (0.000002850, [antiDeltaPlus])           # ->  anti-delta+  n0  K_L0
                        ])

    Bzero = imp(BR_pdg=[(0.00000076667, None),        # ->  K_L0  nu_e   anti-nu_e
                        (0.00000076667, None),        # ->  K_L0  nu_mu  anti-nu_mu
                        (0.00000076667, None),        # ->  K_L0  nu_tau anti-nu_tau
                        (0.000445500, [Jpsi]),        # ->  K_L0  J/psi
                        (0.000410000, [eta_c]),       # ->  K_L0  eta_c
                        (0.000165000, [eta_c2S]),     # ->  K_L0  eta_c(2S)
                        (0.000095000, [chi_c0]),      # ->  K_L0  chi_c0
                        (0.000197500, [chi_c1]),      # ->  K_L0  chi_c1
                        (0.000007500, [chi_c2]),      # ->  K_L0  chi_c2
                        (0.000290000, [Psi2S]),       # ->  K_L0  Psi(2S)
                        ])
    #############################
    # execute commands

    for i in Bplus.prop:
        print(i)

    #print(tau.prop)


if __name__ == '__main__':
    main()



