# Weisfeiler-Lehman-Kernels-on-Protein-Graphs
This repository contains code and example data for running Weisfeiler-Lehamn subgraph kernels on Proteins. It was created by a student of the University of Saarland as a Bachelor thesis.

Run the code Weisfeiler_Lehman_on_PDBs with:

  --d and a file in which the names of pdb files are listed in seperate lines, to download the requested files and the run the subgraph kernel on these PDB       files, there is goining to be a directory called downloads in which the requested PDB files will be safed and there will be a file containing a             matrix with the results of the WL kernel, lastly there is going to be a heat map of the matrix to visualise the results
  
  --l and a directory that contains PDB files to run the kernel on the files directly, there will be a file containing a matrix with the results of the WL       kernel, lastly there is going to be a heat map of the matrix to visualise the results

both arguments can be used simultainiusly but are not required, if there is no argument the code will not run any kernel and just terminate
