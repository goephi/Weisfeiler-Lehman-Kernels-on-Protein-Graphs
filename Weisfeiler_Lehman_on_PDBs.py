from __future__ import print_function


import os
import grakel
from argparse import ArgumentParser
import urllib.request
import numpy as np
import timeit


import matplotlib.pyplot as plt

from grakel.kernels import WeisfeilerLehman, VertexHistogram


def list_to_dict(l):
    """Convert list to dict"""
    return {val: i for i, val in enumerate(l)}


node_encoding = list_to_dict(
    [
        "ala",
        "arg",
        "asn",
        "asp",
        "cys",
        "gln",
        "glu",
        "gly",
        "his",
        "ile",
        "leu",
        "lys",
        "met",
        "phe",
        "pro",
        "ser",
        "thr",
        "trp",
        "tyr",
        "val",
    ]
)


class Atom:
    """Atom class"""

    def __init__(self, line: str) -> None:
        self.serial = int(line[6:11].strip())
        self.name = line[12:16].strip()
        self.altLoc = line[16].strip()
        self.resName = line[17:20].strip()
        self.chainID = line[21].strip()
        self.num = int(line[22:26].strip())
        self.iCode = line[26].strip()
        self.x = float(line[30:38].strip())
        self.y = float(line[38:46].strip())
        self.z = float(line[46:54].strip())
        self.occupancy = float(line[54:60].strip())
        self.tempFactor = float(line[60:66].strip())
        self.element = line[76:78].strip()
        self.charge = line[78:80].strip()

    def __repr__(self) -> str:
        return f"{self.serial} {self.name}"


class Residue:
    """Residue class"""

    def __init__(self, line: str) -> None:
        self.name = line[17:20].strip()
        self.num = int(line[22:26].strip())
        self.chainID = line[21].strip()
        self.atoms = []

    def add_atom(self, atom: Atom) -> None:
        self.atoms.append(atom)

    def __repr__(self) -> str:
        return f"{self.num} {self.name}"


class Structure:
    """Structure class"""

    def __init__(self, filename: str) -> None:
        self.residues = {}
        self.atoms = {}
        self.parse_file(filename)

    def parse_file(self, filename: str) -> None:
        """Parse PDB file"""
        for line in open(filename, "r"):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                atom = Atom(line)
                self.atoms[atom.serial] = atom
                if atom.num not in self.residues:
                    self.residues[atom.num] = Residue(line)
                self.residues[atom.num].add_atom(atom)

    def __repr__(self) -> str:
        return f"{self.residues}"

    def __get__(self, key: int) -> Atom:
        return self.atoms[key]

    def get_coords(self):
        """Get coordinates of all atoms"""
        coords = [np.array([atom.x, atom.y, atom.z]) for atom in self.atoms.values()]
        return coords

    def get_nodes(self):
        """Get features of all nodes of a graph"""
        return [node_encoding[res.name.lower()] + 1 for res in self.residues.values()]

    def get_edges(self, threshold: float):
        """Get edges of a graph using threshold as a cutoff"""
        coords = self.get_coords()
        edges = [[], []]
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                if np.linalg.norm(coords[i]-coords[j]) < threshold:
                    edges[0] += [i, j]
                    edges[1] += [j, i]

        return edges


def get_pdb_files(directory):
    """
    searches for PDB's in the given directory
    directory the directory in which the PDB's should be located
    """
    file_list = []
    for f in os.listdir(directory):
        if f.endswith(".pdb"):
            file_list.append(f)

    file_list = sorted(file_list, key=str.lower)
    return file_list


def get_structures(directory, file_list):
    """
    transforms the PDB's to structures and then graphs
    directory the directory in which all PDB's are located
    file_list names of the individual PDB's that need to be calculated
    """
    graph_list = []
    for file in file_list:
        filename = directory + "/" + file
        structure = Structure(filename)

        my_edges, my_node_labels = dict(), dict()

        sources, targets = structure.get_edges(10)[0], structure.get_edges(10)[1]

        for index in range(len(sources)):
            source, target = sources[index], targets[index]
            if source in my_edges:
                my_edges[source].append(target)
            else:
                my_edges[source] = [target]
            if source >= len(structure.get_nodes()):
                my_node_labels[source] = 0
            else:
                my_node_labels[source] = structure.get_nodes()[source]
        graph_list.append(grakel.Graph(my_edges, node_labels=my_node_labels))

    return graph_list


def run_weisfeiler_lehman_kernels(graph_list):
    """
    runs the main kernel library
    graph_list a list containing all previously calculated protein graphs
    """
    gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
    result = gk.fit_transform(graph_list)

    return result


def create_matrix_file(result, file_list, name):
    """
    creates a file with the resulting matrix of WL regarding all Protein similarities
    result the results from WL
    file_list as names of PDB's
    """
    with open(name+'.txt', 'w') as output_file:
        for index in range(len(file_list)):
            output_file.write(str(file_list[index]) + '\t' + '\t'.join([str(x) for x in result[index]]) + '\n')
        for e in file_list:
            output_file.write('\t' + str(e))


def create_heat_map(result):
    """
    creates the heatmap regarding the WL results
    result are the previously calculated results
    """
    plt.rcParams["figure.figsize"] = (12.5, 12.5)

    plt.pcolormesh(result)
    plt.colorbar()
    plt.show()

def download_PDBs(download):
    """
    downloads the required PDB's
    downloads filename containing the PDB's to download
    """
    os.mkdir('downloads')
    with open(download) as f:
        lines = f.read().splitlines()

    for e in lines:
        if not e.endswith(".pdb"):
            e = e + ".pdb"
        fullfilename = os.path.join('downloads/', e)
        urllib.request.urlretrieve('https://files.rcsb.org/download/' + e, fullfilename)
    return 'downloads/'


def get_argument_parser():
    """
    helps with the input
    d for files that first need to be downloaded
    l for files that are already in a directory
    """
    p = ArgumentParser(description="standart WL subgraph kernel on the given date")
    p.add_argument("--d", default='emptyxyz', help="give a file that contains the pdb's that need to bee downloaded and run on, this file must not have any '' or ' ' laso every pdb should be in it's own line")
    p.add_argument("--l", default='emptyxyz', help="give a directory containing the pdb's that you want to run on but do not need to be downloaded")
    return p

if __name__ == '__main__':
    p = get_argument_parser()
    args = p.parse_args()
    d, l = args.d, args.l

    if d != 'emptyxyz':
        downloaded_directory = download_PDBs(d)
        file_names = get_pdb_files(downloaded_directory)
        structures = get_structures(downloaded_directory, file_names)
        results = run_weisfeiler_lehman_kernels(structures)
        create_matrix_file(results, file_names, "Downloaded")
        create_heat_map(results)


    if l != 'emptyxyz':
        start = timeit.default_timer()
        file_names = get_pdb_files(l)
        structures = get_structures(l, file_names)
        after_get_structures = timeit.default_timer()
        results = run_weisfeiler_lehman_kernels(structures)
        after_wl = timeit.default_timer()
        create_matrix_file(results, file_names, "Local")
        create_heat_map(results)
        end = timeit.default_timer()

        print("runtime structures:" + str(after_get_structures-start))
        print("runtime wl:" + str(after_wl-after_get_structures))
        print("runtime showing:" + str(end-after_wl))





