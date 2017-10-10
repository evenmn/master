import psi4
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def call_psi4(mol_spec, extra_opts = {}):
    mol = psi4.geometry(mol_spec)

    # Set options to Psi4.
    opts = {'basis': 'cc-pVDZ',
                     'reference' : 'uhf',
                     'scf_type' :'direct',
                     'guess' : 'core',
                     'guess_mix' : 'true',
                     'e_convergence': 1e-7}

    opts.update(extra_opts)

    # If you uncomment this line, Psi4 will *not* write to an output file.
    #psi4.core.set_output_file('output.dat', False)
    
    psi4.set_options(opts)

    # Set up Psi4's wavefunction. Psi4 parses the molecule specification,
    # creates the basis based on the option 'basis' and the geometry,
    # and computes all necessary integrals.
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
    
    # Get access to integral objects
    mints = psi4.core.MintsHelper(wfn.basisset())

    # Get the integrals, and store as numpy arrays
    T = np.asarray(mints.ao_kinetic())      # Kinetic energy
    V = np.asarray(mints.ao_potential())    # Potential energy
    H = T + V                               # Hamiltonian
    W = np.asarray(mints.ao_eri())
    S = np.asarray(mints.ao_overlap())      # Overlap matrix

    # Get number of basis functions and number of occupied orbitals of each type
    nbf = wfn.nso()
    nalpha = wfn.nalpha()
    nbeta = wfn.nbeta()


    # Perform a SCF calculation based on the options.
    SCF_E_psi4 = psi4.energy('SCF')
    Enuc = mol.nuclear_repulsion_energy()

    # We're done, return.
    return SCF_E_psi4, Enuc, H,W,S, nbf, nalpha, nbeta


def density(U, N):
    '''
    Density matrix where U is 
    occupied and N is the number of 
    occupied states
    '''
    U = np.matrix(U[:,:N])
    D = np.dot(U, U.H)
    return D
    
    
def fock_operator(H, W, D_up, D_dn):
    '''
    Constructing the fock operators
    F_up = h + J(D_up + D_dn) - K(D_up),
    F_dn = h + J(D_up + D_dn) - K(D_dn)
    given the Hamiltonian H, the matrix
    W, and the density matrices D_up
    and D_down
    '''
    # J(D)_pq = sum_rs (pq|rs) D_sr     Using Mulliken notation
    J = np.einsum('pqrs, sr->pq', W, D_up + D_dn)
    
    # K(D)_pq = sum_rs (ps|rq) D_sr     Using Mulliken notation
    K_up = np.einsum('psrq, sr->pq', W, D_up)
    K_dn = np.einsum('psrq, sr->pq', W, D_dn)
    
    F_up = H + J + K_up
    F_dn = H + J + K_dn
    return F_up, F_dn
    
    
def UHF_solver(mol_geo):
    SCF_E_psi4, Enuc, H, W, S, nbf, n_up, n_dn = \
    call_psi4(mol_geo, extra_opts = {}) 
    
    # Need to find U by solving the generalized eigenvalue problem He=uSe
    e, u = eigh(H, S)
    D_up = density(u, n_up)
    D_dn = density(u, n_dn)
    F_up, F_dn = fock_operator(H, W, D_up, D_dn)

    # Again we have generalized eigenvalue problem 
    epsilon_up, u_up = eigh(F_up, S)
    epsilon_dn, u_dn = eigh(F_dn, S)
    return SCF_E_psi4, epsilon_up, epsilon_dn

if __name__ == '__main__':
    pass

# Water
r_h2o = 1.809
theta_h2o = 104.59

h2o = """
        O
        H 1 {0}
        H 1 {0} 2 {1}
        symmetry c1
        units bohr
""" .format(r_h2o, theta_h2o)

# Hydrogen
r_h2 = 2.0

h2 = """
    0 1
    H
    H 1 {0}
    symmetry c1
    units bohr
""" .format(r_h2)

psi4_energy, e_up, e_dn = UHF_solver(h2o)
print(psi4_energy)
