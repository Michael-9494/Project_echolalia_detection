import os
import sys
import numpy as np
import librosa
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import patches
import paramiko
sys.path.append(r'C:\Users\97254\Downloads\FMP_1.2.3 (1).zip\FMP_1.2.3')
import libfmp.b
import libfmp.c3
import libfmp.c7

@jit(nopython=True)
def compute_accumulated_cost_matrix_subsequence_dtw(C):
    """Given the cost matrix, compute the accumulated cost matrix for
    subsequence dynamic time warping with step sizes {(1, 0), (0, 1), (1, 1)}

    Notebook: C7/C7S2_SubsequenceDTW.ipynb
    using dynamic programming, we compute the accumulated cost matrix  D  and
     the cost-minimizing index  b∗  in the last row of  D
    Args:
        C (np.ndarray): Cost matrix

    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N, M = C.shape
    D = np.zeros((N, M))
    D[:, 0] = np.cumsum(C[:, 0])
    D[0, :] = C[0, :]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
    return D




@jit(nopython=True)
def compute_optimal_warping_path_subsequence_dtw(D, m=-1):
    """Given an accumulated cost matrix, compute the warping path for
    subsequence dynamic time warping with step sizes {(1, 0), (0, 1), (1, 1)}

    Notebook: C7/C7S2_SubsequenceDTW.ipynb
    we derive the optimal warping path  P∗  using backtracking, which determines the
     index  a∗  of the optimal subsequence  Y(a∗:b∗) .
    Args:
        D (np.ndarray): Accumulated cost matrix
        m (int): Index to start back tracking; if set to -1, optimal m is used (Default value = -1)

    Returns:
        P (np.ndarray): Optimal warping path (array of index pairs)
    """
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = D[N - 1, :].argmin()
    P = [(n, m)]

    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n - 1, m - 1], D[n - 1, m], D[n, m - 1])
            if val == D[n - 1, m - 1]:
                cell = (n - 1, m - 1)
            elif val == D[n - 1, m]:
                cell = (n - 1, m)
            else:
                cell = (n, m - 1)
        P.append(cell)
        n, m = cell
    P.reverse()
    P = np.array(P)
    return P





@jit(nopython=True)
def compute_accumulated_cost_matrix_subsequence_dtw_21(C):
    """Given the cost matrix, compute the accumulated cost matrix for
    subsequence dynamic time warping with step sizes {(1, 1), (2, 1), (1, 2)}

    Notebook: C7/C7S2_SubsequenceDTW.ipynb

    Args:
        C (np.ndarray): Cost matrix

    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N, M = C.shape
    D = np.zeros((N + 1, M + 2))
    D[0:1, :] = np.inf
    D[:, 0:2] = np.inf

    D[1, 2:] = C[0, :]

    for n in range(1, N):
        for m in range(0, M):
            if n == 0 and m == 0:
                continue
            D[n + 1, m + 2] = C[n, m] + min(D[n - 1 + 1, m - 1 + 2], D[n - 2 + 1, m - 1 + 2], D[n - 1 + 1, m - 2 + 2])
    D = D[1:, 2:]
    return D


@jit(nopython=True)
def compute_optimal_warping_path_subsequence_dtw_21(D, m=-1):
    """Given an accumulated cost matrix, compute the warping path for
    subsequence dynamic time warping with step sizes {(1, 1), (2, 1), (1, 2)}

    Notebook: C7/C7S2_SubsequenceDTW.ipynb

    Args:
        D (np.ndarray): Accumulated cost matrix
        m (int): Index to start back tracking; if set to -1, optimal m is used (Default value = -1)

    Returns:
        P (np.ndarray): Optimal warping path (array of index pairs)
    """
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = D[N - 1, :].argmin()
    P = [(n, m)]

    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n - 1, m - 1], D[n - 2, m - 1], D[n - 1, m - 2])
            if val == D[n - 1, m - 1]:
                cell = (n - 1, m - 1)
            elif val == D[n - 2, m - 1]:
                cell = (n - 2, m - 1)
            else:
                cell = (n - 1, m - 2)
        P.append(cell)
        n, m = cell
    P.reverse()
    P = np.array(P)
    return P




if __name__ == '__main__':
    X = np.array([3, 0, 6])
    Y = np.array([2, 4, 0, 4, 0, 0, 5, 2])
    N = len(X)
    M = len(Y)

    plt.figure(figsize=(6, 2))
    plt.plot(X, c='k', label='$X$')
    plt.plot(Y, c='b', label='$Y$')
    plt.legend(loc='lower right')
    plt.tight_layout()


    print('Sequence X =', X)
    print('Sequence Y =', Y)
    # We now compute the cost matrix C using the Euclidean distance as local cost measure:

    C =  libfmp.c3.compute_cost_matrix(X, Y, metric='euclidean')
    print('Cost matrix C =', C, sep='\n')

    # Next, using dynamic programming, we compute the accumulated cost matrix D and the cost - minimizing index b∗ in
    # the last row of D.
    D = compute_accumulated_cost_matrix_subsequence_dtw(C)
    print('Accumulated cost matrix D =', D, sep='\n')
    b_ast = D[-1, :].argmin()
    print('b* =', b_ast)
    print('Accumulated cost D[N, b*] = ', D[-1, b_ast])

    # Finally, we derive the optimal warping path P∗  using backtracking, which determines the index a∗ of the optimal
    # subsequence Y(a∗:b∗).

    P = compute_optimal_warping_path_subsequence_dtw(D)
    print('Optimal warping path P =', P.tolist())
    a_ast = P[0, 1]
    b_ast = P[-1, 1]
    print('a* =', a_ast)
    print('b* =', b_ast)
    print('Sequence X =', X)
    print('Sequence Y =', Y)
    print('Optimal subsequence Y(a*:b*) =', Y[a_ast:b_ast + 1])
    print('Accumulated cost D[N, b_ast]= ', D[-1, b_ast])


    # Finally, we visualize the cost matrix  C  and the accumulated cost matrix  D  of our subsequence
    # DTW approach along with the optimal warping path (indicated by the red dots).
    cmap = libfmp.b.compressed_gray_cmap(alpha=-10, reverse=True)

    plt.figure(figsize=(10, 1.8))
    ax = plt.subplot(1, 2, 1)
    libfmp.c3.plot_matrix_with_points(C, P, linestyle='-', ax=[ax], aspect='equal',
                                      clim=[0, np.max(C)], cmap=cmap, title='$C$ with optimal warping path',
                                      xlabel='Sequence Y', ylabel='Sequence X')

    ax = plt.subplot(1, 2, 2)
    libfmp.c3.plot_matrix_with_points(D, P, linestyle='-', ax=[ax], aspect='equal',
                                      clim=[0, np.max(D)], cmap=cmap, title='$D$ with optimal warping path',
                                      xlabel='Sequence Y', ylabel='Sequence X')

    plt.tight_layout()



    # Matching Function
    # Besides revealing the optimal index  b∗ , the last row of  D  (top row in the visualization)
    # provides more information. Each entry  D(N,m)  for an arbitrary  m∈[1:M]  indicates the total cost of aligning
    # X  with an optimal subsequence of  Y  that ends at position  m . This motivates us to define a matching
    # function  ΔDTW:[1:M]→R  by setting ΔDTW(m):=1ND(N,m)
    # for  m∈[1:M] ,where we have normalized the accumulated cost by the length  N  of the query. Each local minimum
    # b∈[1:M]  of  ΔDTW  that is close to zero indicates the end position of a subsequence  Y(a:b)  that has a small
    # DTW distance to  X . The start index  a∈[1:M] as well as the optimal alignment between this subsequence and  X
    # are obtained by a backtracking procedure starting with the cell  q1=(N,b) . The following example shows
    # the matching function of our previous example.
    Delta = D[-1, :] / N

    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.02],
                                              'height_ratios': [1, 1]}, figsize=(6, 4))
    cmap = libfmp.b.compressed_gray_cmap(alpha=-10, reverse=True)
    libfmp.b.plot_matrix(D, title=r'Accumulated cost matrix $\mathbf{D}$', xlabel='Time (samples)',
                         ylabel='Time (samples)', ax=[ax[0, 0], ax[0, 1]], colorbar=True, cmap=cmap)
    rect = patches.Rectangle((-0.45, 1.48), len(Delta) - 0.1, 1, linewidth=3, edgecolor='r', facecolor='none')
    ax[0, 0].add_patch(rect)

    libfmp.b.plot_signal(Delta, ax=ax[1, 0], xlabel='Time (samples)', ylabel='', ylim=[0, np.max(Delta) + 1],
                         title=r'Matching function $\Delta_\mathrm{DTW}$', color='k')
    ax[1, 0].set_xlim([-0.5, len(Delta) - 0.5])
    ax[1, 0].grid()
    ax[1, 1].axis('off')
    plt.tight_layout()

    # Comparison with Diagonal Matching
    N = len(X)
    M = len(Y)
    C = libfmp.c3.compute_cost_matrix(X, Y, metric='euclidean')

    # Subsequence DTW
    D = compute_accumulated_cost_matrix_subsequence_dtw(C)
    Delta_DTW = D[-1, :] / N
    P_DTW = compute_optimal_warping_path_subsequence_dtw(D)
    a_ast = P[0, 1]
    b_ast = P[-1, 1]

    # Diagonal matching
    Delta_Diag = libfmp.c7.matching_function_diag(C)
    m = np.argmin(Delta_Diag)
    P_Diag = []
    for n in range(N):
        P_Diag.append((n, m + n))
    P_Diag = np.array(P_Diag)
    matches_Diag = [(m, N)]

    # Visualization
    fig, ax = plt.subplots(2, 4, gridspec_kw={'width_ratios': [1, 0.05, 1, 0.05],
                                              'height_ratios': [1, 1]},
                           constrained_layout=True, figsize=(9, 4))
    cmap = libfmp.b.compressed_gray_cmap(alpha=-10, reverse=True)
    libfmp.c3.plot_matrix_with_points(C, P_DTW, linestyle='-', ax=[ax[0, 0], ax[0, 1]],
                                      clim=[0, np.max(C)], cmap=cmap, title='$C$ with optimal warping path',
                                      xlabel='Sequence Y', ylabel='Sequence X')
    libfmp.b.plot_signal(Delta_DTW, ax=ax[1, 0], xlabel='Time (samples)', ylabel='', ylim=[0, 5],
                         title=r'Matching function $\Delta_\mathrm{DTW}$', color='k')
    ax[1, 0].set_xlim([-0.5, len(Delta) - 0.5])
    ax[1, 0].grid()
    ax[1, 0].plot(b_ast, Delta_DTW[b_ast], 'ro')
    ax[1, 0].add_patch(patches.Rectangle((a_ast - 0.5, 0), b_ast - a_ast + 1, 7, facecolor='r', alpha=0.2))
    ax[1, 1].axis('off')

    libfmp.c3.plot_matrix_with_points(C, P_Diag, linestyle='-', ax=[ax[0, 2], ax[0, 3]],
                                      clim=[0, np.max(C)], cmap=cmap, title='$C$ with optimal diagonal path',
                                      xlabel='Sequence Y', ylabel='Sequence X')
    libfmp.b.plot_signal(Delta_Diag, ax=ax[1, 2], xlabel='Time (samples)', ylabel='',
                         ylim=[0, 5], title=r'Matching function $\Delta_\mathrm{Diag}$', color='k')
    ax[1, 2].set_xlim([-0.5, len(Delta) - 0.5])
    ax[1, 2].grid()
    ax[1, 2].plot(m, Delta_Diag[m], 'ro')
    ax[1, 2].add_patch(patches.Rectangle((m - 0.5, 0), N, 7, facecolor='r', alpha=0.2))
    ax[1, 3].axis('off');

    # Step Size Condition
    C = libfmp.c3.compute_cost_matrix(X, Y, metric='euclidean')
    D = compute_accumulated_cost_matrix_subsequence_dtw_21(C)
    P = compute_optimal_warping_path_subsequence_dtw_21(D)

    plt.figure(figsize=(9, 1.8))
    ax = plt.subplot(1, 2, 1)
    libfmp.c3.plot_matrix_with_points(C, P, linestyle='-', ax=[ax], aspect='equal',
                                      clim=[0, np.max(C)], cmap=cmap, title='$C$ with optimal warping path',
                                      xlabel='Sequence Y', ylabel='Sequence X')

    ax = plt.subplot(1, 2, 2)
    D_max = np.nanmax(D[D != np.inf])
    libfmp.c3.plot_matrix_with_points(D, P, linestyle='-', ax=[ax], aspect='equal',
                                      clim=[0, D_max], cmap=cmap, title='$D$ with optimal warping path',
                                      xlabel='Sequence Y', ylabel='Sequence X')
    for x, y in zip(*np.where(np.isinf(D))):
        plt.text(y, x, '$\infty$', horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.show()