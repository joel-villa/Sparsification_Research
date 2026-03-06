Something of potential note: bcsstk03 does not scale well, but it has a small
spectral gap, i.e. top two eigenvalues are similar (this was found 
incidentally as the scipy.sparse.linalg.eigs() function converged to two
different eigenvectors - necessitated MatrixChecker.difference() to initialize 
the initial guess for eigs() in order to converge to the same top eigencector)

Assuming bcsstm39 and crystm03 display similar behavior without testing

TODO: is this assumption true?