#import matplotlib.pyplot as plt
#from matplotlib import rc
#import time
import pandas as pd
import numpy as np
import math
from copy import copy as cp
import sys
import warnings
import uproot
import ROOT
from ROOT import TLorentzVector, TVector3
from ROOT import RDataFrame
warnings.filterwarnings("ignore")

#----------------------------------------------------------------------------#
def RDF_to_pandas(key:str, inputRootFiles:str, columns:list=None, cut=None):
    """
    :param key: Tree name of RootFiles.
    :param inputRootFiles: {path}/files.root.
    :param columns: Variables of interest.
    :param cut: root-like cut to be applied.
    :return: Pandas dataframe.
    """
    if cut: df = ROOT.RDataFrame(key, inputRootFiles).Filter(cut)
    else: df = ROOT.RDataFrame(key, inputRootFiles)

    if columns: df = df.AsNumpy(columns=columns)
    else: df = df.AsNumpy()

    return pd.DataFrame(df)

def DataFrameToOutputRootFile(df, cut:str=None, columns:list='All', key='tree', output_filename='output.root'):
    """
    :param df: input DataFrame.
    :param cut: like-string Cut.
    :param columns: list of columns/variables contained in the DataFrame to be saved in the output rootfile.
    :param key: Tree name of the output rootfile.
    :param output_filename: Name of output roofile.
    :return: None
    """

    if cut: df = df.query(cut)

    if columns == 'All': columns = df.columns
    else:                columns = df[columns]

    with uproot.recreate(output_filename) as file:
        tree = file.mktree(key, {name:np.float64 for name in columns})
        tree.extend({name:df[name].values for name in columns})

#-----------------------------------------------------------------------#
"""
Particles = ['pi_sig', 'K_tag', 'pi_tag', 'track_tag']
momentaCMS = ['CMS_mcPX', 'CMS_mcPY', 'CMS_mcPZ', 'CMS_mcE']

columns = [f'{particle}_{p}' for particle in Particles for p in momentaCMS]
columns+= ['track_tag_mcPDG']

# Matching Cuts
matchCut_HadronicTag = 'abs(pi_sig_mcPDG) == 211 && abs(pi_tag_mcPDG) == 211 && abs(track_tag_mcPDG) == 211 && abs(K_tag_mcPDG) == 321 && ' \
                       'abs(D_tag_mcPDG) == 421 && abs(B_sig_mcPDG) == 521 && abs(B_tag_mcPDG) == 521 && ' \
                       'abs(pi_sig_genMotherPDG) == 521 && abs(track_tag_genMotherPDG) == 521 && ' \
                       'track_tag_genMotherPDG == D_tag_genMotherPDG && ' \
                       'abs(K_tag_genMotherPDG) == 421 && K_tag_genMotherPDG == pi_tag_genMotherPDG && pi_tag_mcPDG*track_tag_mcPDG < 0 && ' \
                       'abs(B_sig_genMotherPDG) == 300553 && B_sig_genMotherPDG == B_tag_genMotherPDG && B_sig_mcPDG*B_tag_mcPDG < 0'
matchCut_SemilepTag  = 'abs(pi_sig_mcPDG) == 211 && abs(pi_tag_mcPDG) == 211 && (abs(track_tag_mcPDG)==11 || abs(track_tag_mcPDG)==13) && ' \
                      'abs(K_tag_mcPDG) == 321 && abs(D_tag_mcPDG) == 421 && abs(B_sig_mcPDG) == 521 && abs(B_tag_mcPDG) == 521 && ' \
                      'abs(pi_sig_genMotherPDG) == 521 && abs(track_tag_genMotherPDG) == 521 && ' \
                      'track_tag_genMotherPDG == D_tag_genMotherPDG && ' \
                      'abs(K_tag_genMotherPDG) == 421 && K_tag_genMotherPDG == pi_tag_genMotherPDG && pi_tag_mcPDG*track_tag_mcPDG > 0 && ' \
                      'abs(B_sig_genMotherPDG) == 300553 && B_sig_genMotherPDG == B_tag_genMotherPDG && B_sig_mcPDG*B_tag_mcPDG < 0'

#-------------------#
path = '/home/jordan/Documents/Analisis_thesis/invisible_search/merged_nTracks4/merged/merged/signals'

df_HadronicTag = RDF_to_pandas('b', f'{path}/ntuple_m0.0_HadronicTag_0.root', columns=columns, cut=matchCut_HadronicTag)

df_SemilepTag_e  = RDF_to_pandas('b', f'{path}/ntuple_m0.0_SemilepTag_0.root',  columns=columns, cut=f'{matchCut_SemilepTag} && abs(track_tag_mcPDG)==11')
df_SemilepTag_mu = RDF_to_pandas('b', f'{path}/ntuple_m0.0_SemilepTag_0.root',  columns=columns, cut=f'{matchCut_SemilepTag} && abs(track_tag_mcPDG)==13')
df_SemilepTag = pd.concat([df_SemilepTag_e, df_SemilepTag_mu], ignore_index=True)
"""
#--------------------------- Nominal Values ----------------------------#

# Collision energy; mass Upsilon(4S)
Ecms = 10579.4e-3

# Nominal masses
e_mass = 0.51099895e-3
mu_mass= 105.65837e-3
pi_mass= 139.57039e-3
K_mass = 493.677e-3
B_mass = 5279.34e-3

pi0_mass = 134.9768e-3
D0_mass = 1864.83e-3
Dp_mass = 1869.65e-3 # Dplus
D0star_mass = 2006.85e-3
Dpstar_mass = 2010.26e-3


# Function for calculation of invariant mass

def config_momentum(frame):
    if frame == 'CMS':   return 'CMS_mcPX', 'CMS_mcPY', 'CMS_mcPZ'
    elif frame == 'lab': return 'mcPX', 'mcPY', 'mcPZ'
    else: sys.exit("No valid frame.")


def inv_mass(df:object, name:str, particles:list, mass_PDG:list, frame='CMS'):
    """
    :param df: Dataframe
    :param name: Name for the output variable
    :param particles: list of particles in str-format
    :param mass_PDG: list of particles' PDG in int-format
    :param CMS_Frame: bool
    :return: add the output variable to the df
    """
    x, y, z = config_momentum(frame)
    aux = {}; four_vector = {'E': 0, 'px': 0, 'py': 0, 'pz': 0}

    for index, particle in enumerate(particles):

        aux[particle] = {'E': 0,
                         'px': df.eval(f"{particle}_{x}"),
                         'py': df.eval(f"{particle}_{y}"),
                         'pz': df.eval(f"{particle}_{z}")}

        aux[particle]['E'] = np.sqrt(aux[particle]['px']**2 + aux[particle]['py']**2 + aux[particle]['pz']**2 + mass_PDG[index]**2)

        four_vector['E'] += aux[particle]['E']

        for p_i in ['px', 'py', 'pz']:
            four_vector[p_i] += aux[particle][p_i]

    inv_M = np.sqrt(four_vector['E']**2 - (four_vector['px']**2 + four_vector['py']**2 + four_vector['pz']**2))
    df[name]=inv_M

# Function for calculation in between to momentum vectors

def sum_momentum(df:object, name:str, particles:list, frame='CMS'):
    """
    :param df: Dataframe
    :param name: Name for the output variable
    :param particles: list of particles in str-format
    :param CMS_Frame: bool
    :return: add the output 3-momentum to the df
    """

    x, y, z = config_momentum(frame)
    px=0; py=0; pz=0

    for particle in particles:
        px+= df.eval(f'{particle}_{x}')
        py+= df.eval(f'{particle}_{y}')
        pz+= df.eval(f'{particle}_{z}')

    name_x = f'{name}_{x}'  ;  df[name_x]=px
    name_y = f'{name}_{y}'  ;  df[name_y]=py
    name_z = f'{name}_{z}'  ;  df[name_z]=pz


def angle_in_between(df:object, name:str, particles:list, frame='CMS'):
    """
    :param df: Dataframe
    :param particles: list of names of *two* given particles
    :return: add to the df the 3D-Angle in btw *two* given particles
    """

    x, y, z = config_momentum(frame)

    px_1 = df.eval(f'{particles[0]}_{x}')  ;  px_2 = df.eval(f'{particles[1]}_{x}')
    py_1 = df.eval(f'{particles[0]}_{y}')  ;  py_2 = df.eval(f'{particles[1]}_{y}')
    pz_1 = df.eval(f'{particles[0]}_{z}')  ;  pz_2 = df.eval(f'{particles[1]}_{z}')

    mag = lambda px, py, pz: (px**2 + py**2 + pz**2)**(1/2)

    mag_1 = mag(px_1, py_1, pz_1)
    mag_2 = mag(px_2, py_2, pz_2)

    cosTheta = (px_1*px_2 + py_1*py_2 + pz_1*pz_2)/(mag_1*mag_2)
    df[name]=cosTheta
    return [name]

#------------------------- add q^2 variable ----------------------------#

def add_q2(df, s=Ecms**2, m_sig=K_mass, particle='track_sig'):
    """
    Analisis based on B+ -> K+ nunu
    Formula found in next link, page: 28:
    https://docs.belle2.org/record/3785/files/BELLE2-TALK-DRAFT-2023-117.pdf
    """
    df['q2'] = s/4 + m_sig**2 - math.sqrt(s)*df[f'{particle}_CMS_mcP']

#------------- Functions for Ysig_in_XsigRest calculation --------------#

def SetFourMomenta(df, particle, nick:str='track_sig', index:int=0):
    # At generation-level.
    particle.SetPxPyPzE(df.eval(f'{nick}_CMS_mcPX')[index],
                        df.eval(f'{nick}_CMS_mcPY')[index],
                        df.eval(f'{nick}_CMS_mcPZ')[index],
                        df.eval(f'{nick}_CMS_mcE')[index])

def ARGUS(df, nick_sig_fsp:list, nick_tag_fsp:list, X_mass:float, tag:str='Semilep', saveFile=False, nameFile='file'):

    """
    *ARGUS* method: only makes sense for SemilepTag.

    :param df: Dataframe
    :param nick_sig_fsp: list of nicks for final-state particles of signal-side. E.g., ['track_sig']
    :param nick_tag_fsp: list of nicks for final-state particles of tag-side.    E.g., ['pi_tag', 'K_tag', 'track_tag']
    :param X_mass: mass of B (for B-physics) or tau (for tau-physics)
    :param saveFile:bool.
    :param nameFile:str
    :return: The magnitude of 3-momenta of Y_sig in B_sig restFrame.

    Topology:
    e+ e- -> Xsig + Xtag; Xsig -> Ysig (=sum(fsp_sig)) + invisible; Xtag -> sum(fsp_tag)
    """

    fsp_sig = [TLorentzVector()]*len(nick_sig_fsp)
    fsp_tag = [TLorentzVector()]*len(nick_tag_fsp)

    X_sig = TLorentzVector()
    if tag=='Semilep': X_tag_new = TVector3()

    Ysig_p_in_XsigRest=[]
    Ysig_p_in_XsigRest_norm=[]  # normalized
    for i in range(len(df)):

        for k, nick in enumerate(nick_sig_fsp):
            SetFourMomenta(df=df, particle=fsp_sig[k], nick=nick, index=i)

        for k, nick in enumerate(nick_tag_fsp):
            SetFourMomenta(df=df, particle=fsp_tag[k], nick=nick, index=i)

        Y_sig = cp(fsp_sig[0])
        X_tag = cp(fsp_tag[0])

        if len(fsp_sig)>1:
            for j in range(len(fsp_sig)-1):
                Y_sig += fsp_sig[j+1]

        if len(fsp_tag)>1:
            for j in range(len(fsp_tag)-1):
                X_tag += fsp_tag[j+1]

        # ARGUS APPROXIMATION
        if tag == 'Semilep':
            X_tag_new_p = math.sqrt((Ecms/2)**2 - X_mass**2)
            X_tag_new.SetMagThetaPhi(X_tag_new_p, X_tag.Theta(), X_tag.Phi())
            X_sig.SetPxPyPzE(-X_tag_new.Px(), -X_tag_new.Py(), -X_tag_new.Pz(), Ecms/2)
        elif tag == 'Hadronic':
            X_sig.SetPxPyPzE(-X_tag.Px(), -X_tag.Py(), -X_tag.Pz(), X_tag.E())
        else:
            sys.exit("ERROR: Las opciones válidas son 'Hadronic' y 'Semilep'")

        boost_Y_sig = Y_sig.BoostVector()
        boost_X_sig = X_sig.BoostVector()

        Y_sig_inX_sig = cp(Y_sig)
        Y_sig_inX_sig.Boost(-boost_X_sig)

        Ysig_p_in_XsigRest.append(Y_sig_inX_sig.P())
        Ysig_p_in_XsigRest_norm.append(2*Y_sig_inX_sig.E()/X_mass)  # normalized  = 2*E/mX
        #----------------#

    if saveFile:
        file = open(f"{nameFile}.py", "w")
        file.write(f"ARGUS = {Ysig_p_in_XsigRest}")
        file.write(f"ARGUS_norm = {Ysig_p_in_XsigRest_norm}")
        file.close()

    ################
    return Ysig_p_in_XsigRest, Ysig_p_in_XsigRest_norm

#------------- Functions for Mmax2 and Mmin2 calculation ---------------#

normalize_scalar = lambda scalar: scalar/Ecms
normalize_vector = lambda vector: [normalize_scalar(component) for component in vector]

dot = lambda a, b: np.dot(a, b)
cross = lambda a, b: np.cross(a, b)
scalar_dot = lambda scalar, vector: [scalar*component for component in vector]
sum_vectors = lambda a, b: np.add(a, b)
mag = lambda a: math.sqrt(np.dot(a, a))


def solution_equation(A, B, C, sign=+1):
    """
    General solution for 2-degree polynomial equation
    :param sign: (+1) for Mmax and (-1) for Mmin
    :return: Mmax or Mmin
    """
    disc = B**2 -4*A*C

    if disc >= 0:  Mm2 = (Ecms**2)*( -(B/(2*A)) + (sign/(2*A))*math.sqrt(disc)  )
    else:          Mm2 = np.nan
    return Mm2

def Calculation_Mm2(df, nick_sig_fsp:list, nick_tag_fsp:list, X_mass=B_mass):

    """
    :param particle_a: sum of observable particles' 4-momentum in the signal-side, where is emitted the invisible particle.
    :param particle_b: sum of observable particles' 4-momentum in the tag-side.
    :param X_mass: mass of the mother particle [B or tau]
    :return: Mmax^2 and Mmin^2

    This is based on this paper: "New method for beyond the Standard Model invisible particle searches in tau lepton decays"
    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.115001
    """

    FourMomenta_components = ['CMS_mcPX', 'CMS_mcPY', 'CMS_mcPZ', 'CMS_mcE']

    fsp_sig_Info = {nick:[df.eval(f'{nick}_{p}') for p in FourMomenta_components] for nick in nick_sig_fsp}
    fsp_tag_Info = {nick:[df.eval(f'{nick}_{p}') for p in FourMomenta_components] for nick in nick_tag_fsp}

    particle_a = [sum([fsp_sig_Info[nick][i] for nick in nick_sig_fsp]) for i in range(len(FourMomenta_components))]
    particle_b = [sum([fsp_tag_Info[nick][i] for nick in nick_tag_fsp]) for i in range(len(FourMomenta_components))]

    Mmax2 = [];  Mmin2 = []
    X_mass_norm = normalize_scalar(X_mass)

    for i in range(len(df)):

        Momentum3_a = [particle_a[k][i] for k in [0, 1, 2]];  vect_a = normalize_vector(Momentum3_a)
        Momentum3_b = [particle_b[k][i] for k in [0, 1, 2]];  vect_b = normalize_vector(Momentum3_b)

        Energy_a = particle_a[3][i];      z_a = normalize_scalar(Energy_a)
        Energy_b = particle_b[3][i];      z_b = normalize_scalar(Energy_b)

        vect_H = sum_vectors( scalar_dot((z_b ** 2 - z_b - mag(vect_b) ** 2 - 2 * dot(vect_a, vect_b)), vect_a),
                              scalar_dot((z_a ** 2 - z_a + mag(vect_a) ** 2), vect_b) )
        #---------#
        A_1 = mag(vect_b) ** 2
        A_2 = mag(vect_a) ** 2
        A_3 = 2*dot(vect_a, vect_b)

        B_1 = 2*dot(vect_b, vect_H)
        B_2 = 2*dot(vect_a, vect_H)

        C_1 = 4*((mag(cross(vect_a, vect_b)))**2)
        D_1 = mag(vect_H)**2 - C_1*((0.5 - z_a)**2)

        #---------#
        A_0 = cp(A_1)
        B_0 = -B_1 + C_1 - (2*A_1 + A_3)*(X_mass_norm**2)
        C_0 = (A_1 + A_2 + A_3)*(X_mass_norm**4) + (B_1 + B_2)*(X_mass_norm ** 2) + D_1

        #---------#
        Mmax2.append(solution_equation(A_0, B_0, C_0, sign=+1))
        Mmin2.append(solution_equation(A_0, B_0, C_0, sign=-1))

    # NaN values will be also saved.
    df['Mmax2'] = Mmax2
    df['Mmin2'] = Mmin2

#--------------------------------------------------------------------------------#

#df = df.sample(frac=0.1).reset_index()

# add q^2 variable
#add_q2(df=df, m_sig=pi_mass, particle='track_sig')

# add ARGUS variable
#ARGUS(df=df, nick_sig_fsp=['track_sig'], nick_tag_fsp=['pi_tag', 'K_tag', 'track_tag'], X_mass=B_mass, tag='Semilep')

# add Mmin2 y Mmax2 variables
#Calculation_Mm2(df=df, nick_sig_fsp=['track_sig'], nick_tag_fsp=['pi_tag', 'K_tag', 'track_tag'], X_mass=B_mass)

# correlation coeficient
#print(np.corrcoef(df.Mmax2, df.Mmin2)[0,1])

#DataFrameToOutputRootFile(df=df, key='b', output_filename='output.root')





