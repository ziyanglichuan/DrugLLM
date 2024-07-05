
from rdkit import Chem
from rdkit.Chem import MolStandardize
from collections import deque
from collections import defaultdict

def convert_smiles(smiles_1, smiles_2):
    smiles_1_mol = Chem.MolFromSmiles(smiles_1)
    smiles_2_mol = Chem.MolFromSmiles(smiles_2)

    new_smiles_1 = Chem.MolToSmiles(smiles_1_mol)
    new_smiles_2 = Chem.MolToSmiles(smiles_2_mol)

    return new_smiles_1, new_smiles_2

smiles_1 = "c1(Cl)c(Cl)c(N)ccc1c1nnc(N)nc1(N)"
smiles_2 = "O=[SH](=O)N1CCCCC1"

new_smiles_1, new_smiles_2 = convert_smiles(smiles_1, smiles_2)

print(new_smiles_1)
print(new_smiles_2)










def bfs(mol, start_atom):
    visited = set()
    order = []
    queue = deque([(start_atom, 0)])
    while queue:
        atom, level = queue.popleft()
        if atom.GetIdx() not in visited:
            visited.add(atom.GetIdx())
            if len(order) <= level:
                order.append([])
            order[level].append((atom.GetSymbol()))
            neighbors = list(atom.GetNeighbors())
            
            all_neighbors_visited = True
            for neighbor in neighbors:
                if neighbor.GetIdx() not in visited:
                    all_neighbors_visited = False
                    break
            if all_neighbors_visited:
                order[level].append(("0"))
                
            for neighbor in neighbors:
                queue.append((neighbor, level + 1))
    for level in order:
        level.sort()
    return order
    
def get_bond_types(atom):
    bond_types = []
    for bond in atom.GetBonds():
        bond_types.append(str(bond.GetBondType()))
    bond_types.sort()
    return bond_types



def find_same_orders(mol):
    orders = defaultdict(list)
    for i, atom in enumerate(mol.GetAtoms()):
        order = bfs(mol, atom)
                        
        print(order)
        atom.SetAtomMapNum(i + 1)
        orders[tuple(map(tuple, order))].append(atom.GetIdx())
    test_smiles = Chem.MolToSmiles(mol)
    print(test_smiles)
    for order, atoms in orders.items():
        if len(atoms) > 1:
            print(f'Order: {order}, Atoms: {atoms}')

#smiles = "c2ccc3ccc[nH+]c3c2"
#mol = Chem.MolFromSmiles(smiles)

#find_same_orders(mol)
    
    
    
    
    


