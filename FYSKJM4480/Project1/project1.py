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


# === UHF ===

def density_matrix(U, N, noise=False):
    '''
    Density matrix where U is 
    occupied and N is the number of 
    occupied states
    '''
    U = np.matrix(U[:,:N])
    D = np.dot(U, U.H)      # D is a symmetric matrix
    if noise:
        X = np.random.rand(len(U), len(U[0]))
        X = X + np.transpose(X)
        D += 0.001*X
    return D


def J_matrix(D, W):
    '''J(D)_pq = sum_rs (pq|rs) D_sr     Using Mulliken notation'''
    return np.einsum('pqrs, sr->pq', W, D)


def K_matrix(D, W):
    '''K(D)_pq = sum_rs (ps|rq) D_sr     Using Mulliken notation'''
    return np.einsum('psrq, sr->pq', W, D)
    
    
def uhf_fock_matrices(H, W, D_up, D_dn):
    '''
    Constructing the fock operators
    F_up = h + J(D_up + D_dn) - K(D_up),
    F_dn = h + J(D_up + D_dn) - K(D_dn)
    given the Hamiltonian H, the matrix
    W, and the density matrices D_up
    and D_down
    '''
    J    = J_matrix(D_up + D_dn, W)
    K_up = K_matrix(D_up, W)
    K_dn = K_matrix(D_dn, W)
    
    F_up = H + J + K_up
    F_dn = H + J + K_dn
    return F_up, F_dn
    
    
def UHF_SCF_solver(mol_geo, plot=False):
    SCF_E_psi4, Enuc, H, W, S, nbf, n_up, n_dn = \
    call_psi4(mol_geo, extra_opts = {}) 
    
    # Need to find U by solving the generalized eigenvalue problem He=uSe
    e, u = eigh(H, S)
    D_up = density_matrix(u, n_up)
    D_dn = density_matrix(u, n_dn)
    F_up, F_dn = uhf_fock_matrices(H, W, D_up, D_dn)

    # Again we have generalized eigenvalue problem 
    epsilon_up = eigh(F_up, S, eigvals_only=True)
    epsilon_dn = eigh(F_dn, S, eigvals_only=True)
    
    # Energy calculations 
    print('Calculated UHF_energy:   ', UHF_energy(D_up, D_dn, H, W))
    print('Exact UHF_energy (Psi4): ', SCF_E_psi4)
    
    # Plots the eigenvalues as a energy spectrum
    if plot:
        plot_energy_spectrum(epsilon_up, epsilon_dn)
    
    return None


def UHF_energy(D_up, D_dn, H, W):
    '''
    Use theory from section 2.1
    E_UHF = Fock_energy (E_F) + Hartree_energy (E_H) - Exchange_energy (E_EX)
    J(D)_{pq} = sum_{rs}(pq|rs)D_{sr}
    K(D)_{pq} = sum_{rs}(ps|rq)D_{sr}
    '''
    J_up = J_matrix(D_up, W)
    J_dn = J_matrix(D_dn, W)
    K_up = K_matrix(D_up, W)
    K_dn = K_matrix(D_dn, W)
    
    # F^s = h + J(D_up + D_dn) - K(D^s) where s is the spin 
    F_up = H + J_matrix(D_up + D_dn, W) - K_up
    F_dn = H + J_matrix(D_up + D_dn, W) - K_dn
    
    # E_F = sum_{i,s}(phi_i^s|h|phi_i^s)
    E_F =   np.einsum('pq, qp->', F_up, D_up) \
          + np.einsum('pq, qp->', F_dn, D_dn)
    
    # E_H = 1/2 sum (ii|jj) - Sum over all possible combinations of J^s, D^s
    E_H =  0.5*(np.einsum('pq, qp->', J_up, D_up)\
              + np.einsum('pq, qp->', J_up, D_dn)\
              + np.einsum('pq, qp->', J_dn, D_up)\
              + np.einsum('pq, qp->', J_dn, D_dn))

    # E_H = 1/2 sum (ii|jj) - Sum over all possible combinations of J^s, D^s
    E_EX = 0.5*(np.einsum('pq, qp->', K_up, D_up)\
              + np.einsum('pq, qp->', K_dn, D_dn))
              
    return E_F + E_H - E_EX
    
    
def plot_energy_spectrum(e_up, e_dn):
    '''Plots a horizontal energy spectrum for mol_geo'''
    for e in e_up:
        plt.axhline(e, linewidth=0.5)
    plt.title('Energy spectrum for $H_2O$')
    plt.ylabel('Energy, [Hartree]')
    plt.show()
    
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

UHF_SCF_solver(h2o, plot=True)
