from rdkit import Chem
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
from collections import namedtuple
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import MolSurf, Crippen
import math
import json
import re
import sascorer
import networkx as nx

QEDproperties = namedtuple('QEDproperties', 'MW,ALOGP,HBA,HBD,PSA,ROTB,AROM,ALERTS')
ADSparameter = namedtuple('ADSparameter', 'A,B,C,D,E,F,DMAX')

WEIGHT_MAX = QEDproperties(0.50, 0.25, 0.00, 0.50, 0.00, 0.50, 0.25, 1.00)
WEIGHT_MEAN = QEDproperties(0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95)
WEIGHT_NONE = QEDproperties(1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00)

adsParameters = {
  'MW': ADSparameter(A=2.817065973, B=392.5754953, C=290.7489764, D=2.419764353, E=49.22325677,
                     F=65.37051707, DMAX=104.9805561),
  'ALOGP': ADSparameter(A=3.172690585, B=137.8624751, C=2.534937431, D=4.581497897, E=0.822739154,
                        F=0.576295591, DMAX=131.3186604),
  'HBA': ADSparameter(A=2.948620388, B=160.4605972, C=3.615294657, D=4.435986202, E=0.290141953,
                      F=1.300669958, DMAX=148.7763046),
  'HBD': ADSparameter(A=1.618662227, B=1010.051101, C=0.985094388, D=0.000000001, E=0.713820843,
                      F=0.920922555, DMAX=258.1632616),
  'PSA': ADSparameter(A=1.876861559, B=125.2232657, C=62.90773554, D=87.83366614, E=12.01999824,
                      F=28.51324732, DMAX=104.5686167),
  'ROTB': ADSparameter(A=0.010000000, B=272.4121427, C=2.558379970, D=1.565547684, E=1.271567166,
                       F=2.758063707, DMAX=105.4420403),
  'AROM': ADSparameter(A=3.217788970, B=957.7374108, C=2.274627939, D=0.000000001, E=1.317690384,
                       F=0.375760881, DMAX=312.3372610),
  'ALERTS': ADSparameter(A=0.010000000, B=1199.094025, C=-0.09002883, D=0.000000001, E=0.185904477,
                         F=0.875193782, DMAX=417.7253140),
}

def ads(x, adsParameter):
  """ ADS function """
  p = adsParameter
  exp1 = 1 + math.exp(-1 * (x - p.C + p.D / 2) / p.E)
  exp2 = 1 + math.exp(-1 * (x - p.C - p.D / 2) / p.F)
  dx = p.A + p.B / exp1 * (1 - 1 / exp2)
  return dx / p.DMAX


def properties(mol):
  """
  Calculates the properties that are required to calculate the QED descriptor.
  """
  AcceptorSmarts = [
      '[oH0;X2]',
      '[OH1;X2;v2]',
      '[OH0;X2;v2]',
      '[OH0;X1;v2]',
      '[O-;X1]',
      '[SH0;X2;v2]',
      '[SH0;X1;v2]',
      '[S-;X1]',
      '[nH0;X2]',
      '[NH0;X1;v3]',
      '[$([N;+0;X3;v3]);!$(N[C,S]=O)]'
    ]
  Acceptors = [Chem.MolFromSmarts(hba) for hba in AcceptorSmarts]
  AliphaticRings = Chem.MolFromSmarts('[$([A;R][!a])]')
  StructuralAlertSmarts = [
  '*1[O,S,N]*1',
  '[S,C](=[O,S])[F,Br,Cl,I]',
  '[CX4][Cl,Br,I]',
  '[#6]S(=O)(=O)O[#6]',
  '[$([CH]),$(CC)]#CC(=O)[#6]',
  '[$([CH]),$(CC)]#CC(=O)O[#6]',
  'n[OH]',
  '[$([CH]),$(CC)]#CS(=O)(=O)[#6]',
  'C=C(C=O)C=O',
  'n1c([F,Cl,Br,I])cccc1',
  '[CH1](=O)',
  '[#8][#8]',
  '[C;!R]=[N;!R]',
  '[N!R]=[N!R]',
  '[#6](=O)[#6](=O)',
  '[#16][#16]',
  '[#7][NH2]',
  'C(=O)N[NH2]',
  '[#6]=S',
  '[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]',
  'C1(=[O,N])C=CC(=[O,N])C=C1',
  'C1(=[O,N])C(=[O,N])C=CC=C1',
  'a21aa3a(aa1aaaa2)aaaa3',
  'a31a(a2a(aa1)aaaa2)aaaa3',
  'a1aa2a3a(a1)A=AA=A3=AA=A2',
  'c1cc([NH2])ccc1',
  '[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,' +
  'Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]',
  'I',
  'OS(=O)(=O)[O-]',
  '[N+](=O)[O-]',
  'C(=O)N[OH]',
  'C1NC(=O)NC(=O)1',
  '[SH]',
  '[S-]',
  'c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]',
  'c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]',
  '[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1',
  '[CR1]1[CR1][CR1]cc[CR1][CR1]1',
  '[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1',
  '[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1',
  '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
  '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
  'C#C',
  '[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]',
  '[$([N+R]),$([n+R]),$([N+]=C)][O-]',
  '[#6]=N[OH]',
  '[#6]=NOC=O',
  '[#6](=O)[CX4,CR0X3,O][#6](=O)',
  'c1ccc2c(c1)ccc(=O)o2',
  '[O+,o+,S+,s+]',
  'N=C=O',
  '[NX3,NX4][F,Cl,Br,I]',
  'c1ccccc1OC(=O)[#6]',
  '[CR0]=[CR0][CR0]=[CR0]',
  '[C+,c+,C-,c-]',
  'N=[N+]=[N-]',
  'C12C(NC(N1)=O)CSC2',
  'c1c([OH])c([OH,NH2,NH])ccc1',
  'P',
  '[N,O,S]C#N',
  'C=C=O',
  '[Si][F,Cl,Br,I]',
  '[SX2]O',
  '[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)',
  'O1CCCCC1OC2CCC3CCCCC3C2',
  'N=[CR0][N,n,O,S]',
  '[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2',
  'C=[C!r]C#N',
  '[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1',
  '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1',
  '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])',
  '[OH]c1ccc([OH,NH2,NH])cc1',
  'c1ccccc1OC(=O)O',
  '[SX2H0][N]',
  'c12ccccc1(SC(S)=N2)',
  'c12ccccc1(SC(=S)N2)',
  'c1nnnn1C=O',
  's1c(S)nnc1NC=O',
  'S1C=CSC1=S',
  'C(=O)Onnn',
  'OS(=O)(=O)C(F)(F)F',
  'N#CC[OH]',
  'N#CC(=O)',
  'S(=O)(=O)C#N',
  'N[CH2]C#N',
  'C1(=O)NCC1',
  'S(=O)(=O)[O-,OH]',
  'NC[F,Cl,Br,I]',
  'C=[C!r]O',
  '[NX2+0]=[O+0]',
  '[OR0,NR0][OR0,NR0]',
  'C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]',
  '[CX2R0][NX3R0]',
  'c1ccccc1[C;!R]=[C;!R]c2ccccc2',
  '[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]',
  '[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]',
  '[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]',
  '[*]=[N+]=[*]',
  '[SX3](=O)[O-,OH]',
  'N#N',
  'F.F.F.F',
  '[R0;D2][R0;D2][R0;D2][R0;D2]',
  '[cR,CR]~C(=O)NC(=O)~[cR,CR]',
  'C=!@CC=[O,S]',
  '[#6,#8,#16][#6](=O)O[#6]',
  'c[C;R0](=[O,S])[#6]',
  'c[SX2][C;!R]',
  'C=C=C',
  'c1nc([F,Cl,Br,I,S])ncc1',
  'c1ncnc([F,Cl,Br,I,S])c1',
  'c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])',
  '[#6]S(=O)(=O)c1ccc(cc1)F',
  '[15N]',
  '[13C]',
  '[18O]',
  '[34S]'
]

  StructuralAlerts = [Chem.MolFromSmarts(smarts) for smarts in StructuralAlertSmarts]
  if mol is None:
    raise ValueError('You need to provide a mol argument.')
  mol = Chem.RemoveHs(mol)
  qedProperties = QEDproperties(
    MW=rdmd._CalcMolWt(mol),
    ALOGP=Crippen.MolLogP(mol),
    HBA=sum(len(mol.GetSubstructMatches(pattern)) for pattern in Acceptors
            if mol.HasSubstructMatch(pattern)),
    HBD=rdmd.CalcNumHBD(mol),
    PSA=MolSurf.TPSA(mol),
    ROTB=rdmd.CalcNumRotatableBonds(mol, rdmd.NumRotatableBondsOptions.Strict),
    AROM=rdmd.CalcNumAromaticRings(mol),
    #AROM=Chem.GetSSSR(Chem.DeleteSubstructs(Chem.Mol(mol), AliphaticRings)),
    ALERTS=sum(1 for alert in StructuralAlerts if mol.HasSubstructMatch(alert)),
  )
  # The replacement
  # AROM=Lipinski.NumAromaticRings(mol),
  # is not identical. The expression above tends to count more rings
  # N1C2=CC=CC=C2SC3=C1C=CC4=C3C=CC=C4
  # OC1=C(O)C=C2C(=C1)OC3=CC(=O)C(=CC3=C2C4=CC=CC=C4)O
  # CC(C)C1=CC2=C(C)C=CC2=C(C)C=C1  uses 2, should be 0 ?
  return qedProperties


def qed(mol, w=WEIGHT_MEAN, qedProperties=None):
  if qedProperties is None:
      qedProperties = properties(mol)

  d = [ads(pi, adsParameters[name]) for name, pi in qedProperties._asdict().items()]
  t = sum(wi * math.log(di) for wi, di in zip(w, d))
  return math.exp(t / sum(w))

class ESOLCalculator:
    def __init__(self):
        self.aromatic_query = Chem.MolFromSmarts("a")
        self.Descriptor = namedtuple("Descriptor", "mw logp rotors ap")

    def calc_ap(self, mol):
        """
        Calculate aromatic proportion #aromatic atoms/#atoms total
        :param mol: input molecule
        :return: aromatic proportion
        """
        matches = mol.GetSubstructMatches(self.aromatic_query)
        return len(matches) / mol.GetNumAtoms()

    def calc_esol_descriptors(self, mol):
        """
        Calcuate mw,logp,rotors and aromatic proportion (ap)
        :param mol: input molecule
        :return: named tuple with descriptor values
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        ap = self.calc_ap(mol)
        return self.Descriptor(mw=mw, logp=logp, rotors=rotors, ap=ap)

    def calc_esol_orig(self, mol):
        """
        Original parameters from the Delaney paper, just here for comparison
        :param mol: input molecule
        :return: predicted solubility
        """
        # just here as a reference don't use this!
        intercept = 0.16
        coef = {"logp": -0.63, "mw": -0.0062, "rotors": 0.066, "ap": -0.74}
        desc = self.calc_esol_descriptors(mol)
        esol = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw + coef["rotors"] * desc.rotors \
               + coef["ap"] * desc.ap
        return esol

    def calc_esol(self, mol):
        """
        Calculate ESOL based on descriptors in the Delaney paper, coefficients refit for the RDKit using the
        routine refit_esol below
        :param mol: input molecule
        :return: predicted solubility
        """
        intercept = 0.26121066137801696
        coef = {'mw': -0.0066138847738667125, 'logp': -0.7416739523408995, 'rotors': 0.003451545565957996, 'ap': -0.42624840441316975}
        desc = self.calc_esol_descriptors(mol)
        esol = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw + coef["rotors"] * desc.rotors \
               + coef["ap"] * desc.ap
        return esol

def penalized_logp(mol):
    if mol is None: return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    #return normalized_log_p + normalized_SA + normalized_cycle
    return normalized_SA

class PropertyCalculator:
    def __init__(self):
        self.prop_names = {
            qed: 'QED',
            Lipinski.NumRotatableBonds: 'NRBonds',
            Descriptors.TPSA: 'TPSA',
            Descriptors.MolLogP: 'LogP',
            penalized_logp: 'SA',
            Lipinski.NumHAcceptors: 'NHAcceptors',
            ESOLCalculator().calc_esol: 'Esol',
            Lipinski.FractionCSP3: 'FCSP3'
        }

    def ifSuccess(self, smiles1, smiles2, comparison, prop1=None, prop2=None):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            print("mol error")
            return 0

        ref_prop1, mol_prop1 = (prop1(mol1), prop1(mol2)) if prop1 is not None else (None, None)
        ref_prop2, mol_prop2 = (prop2(mol1), prop2(mol2)) if prop2 is not None else (None, None)

        comp_dict = {'increase': '>', 'decrease': '<'}
        for i, comp in enumerate(comparison):
            if i == 0 and prop1 is not None:
                if not eval(f'{mol_prop1}{comp_dict[comp]}{ref_prop1}'):
                    return 0
            elif i == 1 and prop2 is not None:
                if not eval(f'{mol_prop2}{comp_dict[comp]}{ref_prop2}'):
                    return 0
        if prop2 is None:
            print(f'{mol_prop1}{comp_dict[comparison[0]]}{ref_prop1}')
        else:
            print(f'{mol_prop1}{comp_dict[comparison[0]]}{ref_prop1}   {mol_prop2}{comp_dict[comparison[1]]}{ref_prop2}')
        return 1

if __name__ == '__main__':
    calculator = PropertyCalculator()
    with open('changeInstru_data/change2_offset0_SA.json', 'r') as f:
        data = json.load(f)

    correct_count = 0
    total_count = 0

    with open('Results.txt', 'w') as results_file, open('Accuracy.txt', 'w') as accuracy_file:
        for item in data:
            instruction = item['instruction']
            input_value = item['input'] 
            match = re.search(r'\{([^}]*)\}(?!.*\{)', instruction)
            if match:
                result = match.group(1)
                smiles_before = result

                results_file.write('' + result + '\n')
            else:
                results_file.write('No match found\n')
                total_count += 1
                continue

            output = item['output']
            match = re.search(r'\{([^}]*)\}', output)
            if match:
                result = match.group(1)
                smiles_after = result
                results_file.write('' + result + '\n')
            else:
                results_file.write('No match found\n')
                total_count += 1

            prop1_func = next(key for key, value in calculator.prop_names.items() if value == 'SA') 
            correct_count += calculator.ifSuccess(smiles_before, smiles_after, input_value, prop1=prop1_func, prop2=None)

            total_count += 1

            if total_count % 100 == 0:
                accuracy = correct_count / total_count
                accuracy_file.write(f'Accuracy: {accuracy:.2f}\n')

        accuracy = correct_count / total_count
        accuracy_file.write(f'Final accuracy: {accuracy:.2f}\n')


















