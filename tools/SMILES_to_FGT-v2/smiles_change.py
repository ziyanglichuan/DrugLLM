from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import SanitizeFlags
from collections import deque
import itertools
import re
import os
import sys
from collections import defaultdict

from rdkit import RDLogger
# 禁用 RDKit 的错误日志
RDLogger.DisableLog('rdApp.error')

# 动态添加模块路径（
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


# 使用相对路径加载数据文件
merged_smiles_path = os.path.join(current_dir, './vocab/merged_smiles_list.txt')
with open(merged_smiles_path, 'r') as f:
    merged_smiles_list = set(line.strip() for line in f)


def process_molecule(mol):
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol)
    return mol, smiles


def get_atom_groups(mol):
    ssr = Chem.GetSymmSSSR(mol)
    merged_rings = []
    shared_oneAtom = []  # 用于记录共享原子的环和编号信息
    shared_oneBond = []
    atom_ring_count = defaultdict(list)
    # 遍历每个环
    for ring in ssr:
        merged = False
        # 遍历当前的 merged_rings 并合并
        for i, merged_ring in enumerate(merged_rings):
            if set(ring) & merged_ring:  # 如果有交集
                # 获取交集的原子
                intersection = set(ring) & merged_ring
                if len(intersection) == 1 or len(intersection) == 2:
                    # 如果交集包含两个以下的原子，跳过合并
                    continue
                else:
                    merged_rings[i] = set(ring) | merged_ring  # 合并
                    merged = True
                break
        if not merged:
            merged_rings.append(set(ring))  # 将环存储为 set
            
    # 继续检查环是否需要合并
    while True:
        merged = False
        to_remove = []  # 记录需要删除的元素的索引
        for i, ring1 in enumerate(merged_rings):
            for j in range(i + 1, len(merged_rings)):
                ring2 = merged_rings[j]
                # 使用集合交集比较是否有公共元素
                intersection = ring1 & ring2
                if intersection:  # 如果有交集
                    # 如果交集包含两个以下的原子，跳过合并
                    if len(intersection) == 1 or len(intersection) == 2:
                        continue
                    else:
                        merged_rings[i] = ring1 | ring2  # 合并
                        to_remove.append(j)  # 标记要删除的环
                        merged = True
                    break
            if merged:
                break

        # 删除合并的环
        for index in reversed(to_remove):
            if 0 <= index < len(merged_rings):
                merged_rings.pop(index)

        if not merged:
            break
        
    # 如果某个原子位于 3以上的环中，则合并这 3 个环
    for ring_idx, ring in enumerate(merged_rings):
        for atom in ring:
            atom_ring_count[atom].append(ring_idx)

    # 查找被 3 个环共享的原子
    rings_to_merge = set()
    for atom, ring_list in atom_ring_count.items():
        if len(ring_list) >= 3:  # 如果某个原子出现在 3 个不同的环中
            rings_to_merge.update(ring_list)  # 记录这些环的索引

    # 如果有符合条件的环，进行合并
    if rings_to_merge:
        merged_group = set()
        for idx in sorted(rings_to_merge, reverse=True):
            merged_group.update(merged_rings[idx])
            merged_rings.pop(idx)
        merged_rings.append(merged_group)  # 将合并后的基团添加回来
        

    # 将 merged_rings 中的每个环转换为 set，并生成 atom_groups
    atom_groups = [set(ring) for ring in merged_rings]

    # 计算不在环内的原子集合，避免重复创建集合
    merged_ring_atoms = set().union(*merged_rings) if merged_rings else set()  # 合并所有环中的原子
    non_ring_atoms = set(range(mol.GetNumAtoms())) - merged_ring_atoms  # 获取不在环中的原子

    # 筛选不在环中的键
    non_ring_bonds = []
    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()

        # 如果两个原子都不在环中，检查键类型
        if begin_atom_idx in non_ring_atoms and end_atom_idx in non_ring_atoms:
            if bond.GetBondType() not in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]:
                non_ring_bonds.append(bond.GetIdx())  # 将满足条件的键的索引添加到 non_ring_bonds

    # 使用字典加速查找原子所在的组
    atom_to_group = {}
    # 初始化 atom_groups 并将每个非环原子分配到一个单独的组
    for atom_idx in non_ring_atoms:
        new_group = {atom_idx}
        atom_groups.append(new_group)
        atom_to_group[atom_idx] = new_group  # 记录每个原子的组

    # 遍历所有非环键，合并连接的原子组
    for bond_idx in non_ring_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()

        # 查找开始和结束原子的组
        begin_group = atom_to_group.get(begin_atom_idx)
        end_group = atom_to_group.get(end_atom_idx)

        # 如果开始和结束原子分别属于不同组，则合并这两个组
        if begin_group is not None and end_group is not None and begin_group != end_group:
            # 合并两个组
            new_group = begin_group.union(end_group)

            # 从 atom_groups 中移除旧的组
            try:
                atom_groups.remove(begin_group)
                atom_groups.remove(end_group)
            except ValueError:
                # 如果组不存在，跳过
                continue

            # 将新组添加到 atom_groups 中
            atom_groups.append(new_group)
            # 更新 atom_to_group 中的记录
            for atom_idx in new_group:
                atom_to_group[atom_idx] = new_group

    atom_groups = merge_and_check_all_pairs(mol, atom_groups)
    
    # 记录共享原子以及共享键的信息
    for i, ring1 in enumerate(atom_groups):
        for j in range(i + 1, len(atom_groups)):
            ring2 = atom_groups[j]
            # 使用集合交集比较是否有公共元素
            intersection = ring1 & ring2
            if len(intersection) == 1:
                shared_oneAtom.append((ring1, ring2, intersection))  # 记录共享的原子信息
            if len(intersection) == 2:
                shared_oneBond.append((ring1, ring2, intersection))  # 记录共享的键信息

    return atom_groups, shared_oneAtom, shared_oneBond


def merge_and_check_all_pairs(mol, atom_groups):
    """
    遍历 atom_groups 中的所有两两组合，合并并检查是否在 SMILES 列表中。
    如果存在于 merged_smiles_list，则合并原子组并更新 atom_groups。
    
    :param mol: RDKit 分子对象
    :param atom_groups: 原子组列表，每个元素是一个 set
    :param merged_smiles_list: 包含 SMILES 字符串的集合或列表
    """
    merged = True  # 标记是否有合并发生
    while merged:
        merged = False
        new_atom_groups = []
        seen = set()  # 记录已经合并的 group
        
        for i in range(len(atom_groups)):
            if i in seen:
                continue
            for j in range(i + 1, len(atom_groups)):
                if j in seen:
                    continue
                
                group1, group2 = atom_groups[i], atom_groups[j]
                merged_group = group1 | group2
                
                smiles = atom_group_to_smiles(mol, merged_group)
                
                if smiles in merged_smiles_list:
                    new_atom_groups.append(merged_group)
                    seen.add(i)
                    seen.add(j)
                    merged = True
                    break  # 重新开始合并过程
            else:
                # 如果当前组没有被合并，则保留
                if i not in seen:
                    new_atom_groups.append(atom_groups[i])
        
        atom_groups = new_atom_groups  # 更新 atom_groups
    
    return atom_groups


def mark_atoms(mol, group):
    # 第一步：标记原子组中的原子的邻居为 1
    for atom_idx in group:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            # 如果邻居不在原子组中，设置 AtomMapNum 为 1
            if neighbor.GetIdx() not in group:
                neighbor.SetAtomMapNum(1)

    # 第二步：对于所有原子，标记其邻居的 AtomMapNum 为 2，如果邻居在原子组中
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == 1:  # 只有 AtomMapNum 为 1 的原子需要处理
            for neighbor in atom.GetNeighbors():
                # 如果邻居在原子组中，将其 AtomMapNum 设置为 2
                if neighbor.GetIdx() in group:
                    neighbor.SetAtomMapNum(2)


def bfs(mol, start_atom):
    num_atoms = mol.GetNumAtoms()
    #print(num_atoms)
    visited = [False] * num_atoms  # 使用列表代替集合，提高查找速度
    order = []
    queue = deque([(start_atom, 0)])
    start_idx = start_atom.GetIdx()
    visited[start_idx] = True  # 标记起始原子为已访问

    while queue:
        atom, level = queue.popleft()

        # 确保order列表有足够的层级
        if len(order) <= level:
            order.append([])

        # 添加当前原子的信息到对应层级
        order[level].append((atom.GetSymbol(),atom.GetTotalNumHs()))    # get_atomBond(atom)

        # 遍历邻居原子
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True  # 立即标记为已访问，避免重复入队
                queue.append((neighbor, level + 1))

    # 对每一层级内的原子信息进行排序
    for level_atoms in order:
        level_atoms.sort()
    #print(order)
    return order
    
    
def get_atomBond(atom):
    return sorted(str(bond.GetBondType()) for bond in atom.GetBonds())


def get_atom_info(atom: rdchem.Atom) -> tuple:
    charge = atom.GetFormalCharge()
    hybridization = str(atom.GetHybridization())
    spin_multiplicity = atom.GetNumRadicalElectrons() + 1
    valence = atom.GetDegree()
    isotope = atom.GetIsotope()
    aromatic = atom.GetIsAromatic()
    
    mol = atom.GetOwningMol()
    
    ssr = Chem.GetSymmSSSR(mol)
    merged_rings = [ring for ring in ssr]
    
    ring_fragments_smiles = []
    atom_groups = [list(ring) for ring in merged_rings]
    for groups in atom_groups:
        if atom.GetIdx() in groups:
            smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=list(groups))
            test_mol = Chem.MolFromSmiles(smiles, sanitize=False)
            smiles = Chem.MolToSmiles(test_mol)
            ring_fragments_smiles.append(smiles)
    ring_fragments_smiles.sort()
    
    # 获取该原子的 order
    atom_order = bfs(mol, atom)
    
    # 获取邻居原子的 order
    neighbor_orders = []
    for neighbor in atom.GetNeighbors():
        neighbor_order = bfs(mol, neighbor)
        neighbor_orders.append(neighbor_order)
    
    # 合并该原子和邻居原子的 order 并排序
    all_orders = [atom_order] + sorted(neighbor_orders)

    return (
        charge,
        hybridization,
        spin_multiplicity,
        valence,
        isotope,
        ring_fragments_smiles,
        aromatic,
        all_orders,
    )


def get_groups_mol(mol, left_atoms):
    left_groups = set(range(mol.GetNumAtoms())) - set().union(left_atoms)
    left_groups_smiles = atom_group_to_smiles(mol, left_groups)
    groups_mol = Chem.MolFromSmiles(left_groups_smiles)
    return groups_mol


def atom_group_to_smiles(mol, atom_group):
    # 克隆分子以避免修改原始分子
    mol = Chem.Mol(mol)
    
    # 尝试凯库勒化分子
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Chem.KekulizeException:
        print("分子无法凯库勒化，可能包含无法处理的芳香结构。")
        # 根据需求选择是否继续或抛出异常
        raise
    
    # 获取 atom_group 中的原子对象
    submol_atoms = [mol.GetAtomWithIdx(i) for i in atom_group]

    # 创建一个空的可写分子对象
    submol = Chem.RWMol()

    # 记录原子索引的映射（原始分子索引 -> submol 索引）
    atom_idx_map = {}
    for atom in submol_atoms:
        new_atom = Chem.Atom(atom.GetSymbol())
        # 复制原子的形式电荷
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        
        # 保留原始分子上的 AtomMapNum 编号
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        
        new_idx = submol.AddAtom(new_atom)
        atom_idx_map[atom.GetIdx()] = new_idx

    # 添加原子之间的键，避免重复添加
    added_bonds = set()
    for atom in submol_atoms:
        for bond in atom.GetBonds():
            other_atom = bond.GetOtherAtom(atom)
            if other_atom.GetIdx() in atom_idx_map:
                bond_pair = tuple(sorted([atom.GetIdx(), other_atom.GetIdx()]))
                if bond_pair not in added_bonds:
                    bond_type = bond.GetBondType()
                    submol.AddBond(
                        atom_idx_map[bond_pair[0]],
                        atom_idx_map[bond_pair[1]],
                        bond_type
                    )
                    added_bonds.add(bond_pair)

    # 定义键级到数值的映射，忽略芳香键的特殊键级（因为已凯库勒化）
    bond_order_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        # Chem.BondType.AROMATIC: 1.5  # 已凯库勒化，不需要考虑芳香键
    }

    # 定义一个简单的化学价表
    valence_dict = {
        'C': 4, 'O': 2
    }

    # 遍历 atom_group 中的每个原子，计算并添加显式氢原子
    for atom in submol_atoms:
        atom_idx = atom.GetIdx()
        submol_atom = submol.GetAtomWithIdx(atom_idx_map[atom_idx])

        # 获取原子的符号
        symbol = atom.GetSymbol()

        # 获取原子的化学价，如果未在字典中定义，使用 RDKit 提供的方法
        original_valence = valence_dict.get(symbol, atom.GetTotalValence())

        # 计算在 submol 中的键级总和（与 atom_group 内的其他原子）
        bonds_in_group = 0
        for bond in atom.GetBonds():
            other_atom = bond.GetOtherAtom(atom)
            if other_atom.GetIdx() in atom_idx_map:
                bond_type = bond.GetBondType()
                bonds_in_group += bond_order_map.get(bond_type, 1)

        # 计算与组外原子的键级总和
        bonds_outside = 0
        for bond in atom.GetBonds():
            other_atom = bond.GetOtherAtom(atom)
            if other_atom.GetIdx() not in atom_idx_map:
                bond_type = bond.GetBondType()
                bonds_outside += bond_order_map.get(bond_type, 1)

        # 计算需要添加的显式氢原子数量
        num_h = original_valence - bonds_in_group - bonds_outside

        # 确保氢原子数量非负
        num_h = max(int(round(num_h)), 0)

        # 添加显式氢原子
        for _ in range(num_h):
            h = Chem.Atom('H')
            h_idx = submol.AddAtom(h)
            submol.AddBond(atom_idx_map[atom_idx], h_idx, Chem.BondType.SINGLE)

    # 对子分子进行消毒处理
    flags = SanitizeFlags.SANITIZE_CLEANUP
    Chem.SanitizeMol(submol, sanitizeOps=flags)
    # 去除显式氢原子
    submol = Chem.RemoveHs(submol)
    
    smiles = Chem.MolToSmiles(submol)
    submol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(submol, isomericSmiles=True)


def get_bond_type(mol, group, left_atoms):
    for atom_idx in group:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in left_atoms:
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                bond_type = bond.GetBondType()

                if bond_type == Chem.BondType.SINGLE:
                    return 1
                elif bond_type == Chem.BondType.DOUBLE:
                    return 2
                elif bond_type == Chem.BondType.TRIPLE:
                    return 3
                elif bond_type == Chem.BondType.AROMATIC:
                    return 1.5

    return None


def process_connected_atoms(molecule, atom_map_num, connected_atoms):
    # 遍历分子中的所有原子，查找原子映射编号为指定值的原子并处理
    for atom in molecule.GetAtoms():
        if atom.GetAtomMapNum() == atom_map_num:
            atom_need = atom
            atom.SetAtomMapNum(0)  # 清除原子映射编号

            info = get_atom_info(atom_need)  # 获取原子信息
            molecule, check_smiles = process_molecule(molecule)  # 处理分子
            #print(check_smiles)
            for atom_i in molecule.GetAtoms():
                if get_atom_info(atom_i) == info:  # 找到匹配的原子
                    connected_atoms.append(atom_i.GetIdx())  # 记录连接的原子
                    break
                
    return molecule, connected_atoms  # 返回修改后的分子和连接原子列表


def process_connected_bond(molecule, atom_map_num, connected_bonds):
    # 用来存储符合条件的原子信息
    atom_info_list = []

    # 遍历分子中的所有键
    for bond in molecule.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        # 检查两个原子是否都有指定的 atom_map_num
        if begin_atom.GetAtomMapNum() == atom_map_num and end_atom.GetAtomMapNum() == atom_map_num:
            begin_atom_need = begin_atom
            end_atom_need = end_atom

            for atom in molecule.GetAtoms():
                atom.SetAtomMapNum(0)

            begin_atom_info = get_atom_info(begin_atom_need)
            # print("-----------------")
            # print(begin_atom_info)
            end_atom_info = get_atom_info(end_atom_need)

            molecule, _ = process_molecule(molecule)

            # 记录匹配的原子信息
            atom_info_list.append(begin_atom_info)
            atom_info_list.append(end_atom_info)

            break  # 只处理第一个符合条件的键

    connected_bonds = []

    # 遍历分子的键，找到与 atom_info_list 匹配的键
    for bond in molecule.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        # 获取原子信息
        begin_atom_info = get_atom_info(begin_atom)
        # print(begin_atom_info)
        end_atom_info = get_atom_info(end_atom)
        # 检查两个原子信息是否匹配，并且 bond 类型也匹配
        if sorted([begin_atom_info, end_atom_info]) == sorted(atom_info_list):
            connected_bonds.append(bond.GetIdx())  # 记录匹配的键的索引
            break  # 找到第一个匹配的键后停止遍历

    return molecule, connected_bonds


def remove_groups(mol):
    atom_groups, shared_oneAtom, shared_oneBond = get_atom_groups(mol)
    begin_smiles = Chem.MolToSmiles(mol)
    successful_groups = []  # 用于存储成功分离的原子组
    start_mol = None  # 存储处理后的初始分子
    for i in range(1000):
        if i == 999:
            print("The molecule is too large")
            print(begin_smiles)
        removed = False  # 标记当前是否成功移除一个原子组
        
        for group in atom_groups:  # 遍历所有原子组
            group_in_sharedOneAtom = any(group in info for info in shared_oneAtom) if shared_oneAtom else False
            group_in_sharedOneBond = any(group in info for info in shared_oneBond) if shared_oneBond else False
            count = sum(1 for info in shared_oneBond if group in info) if shared_oneBond else 0
            count += sum(1 for info in shared_oneAtom if group in info) if shared_oneAtom else 0

            if count >= 2:
                continue  # 如果在两个或更多 info 中出现，直接跳出外层循环
                        
            # 清除所有原子的原子映射编号
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

            connected_atoms = []  # 用于存储与目标原子连接的原子
            connected_group_atoms = []  # 用于存储与原子组连接的原子

            # 将当前原子组转换为SMILES表示
            start_smiles = atom_group_to_smiles(mol, group)
            start_mol = Chem.MolFromSmiles(start_smiles)

            # 标记分子中属于当前原子组的原子
            if group_in_sharedOneAtom:
                # 遍历shared_oneAtom，找到包含group的元素
                for info in shared_oneAtom:
                    # 如果group属于info[0]或info[1]，则找到对应的info
                    if group == info[0] or group == info[1]:
                        for atom_idx in info[2]:  
                            com_atom = mol.GetAtomWithIdx(atom_idx)  # 获取原子
                            com_atom.SetAtomMapNum(3)  # 设置MapNum为3
                    
            elif group_in_sharedOneBond:
                # 遍历shared_oneBond，找到包含group的元素
                for info in shared_oneBond:
                    # 如果group属于info[0]或info[1]，则找到对应的info
                    if group == info[0] or group == info[1]:
                        for bond_atomIdx in info[2]:
                            com_atom = mol.GetAtomWithIdx(bond_atomIdx)  # 获取原子
                            com_atom.SetAtomMapNum(4)
            else:
                mark_atoms(mol, group)
                
            # 获取剩余原子的索引集合
            left_atoms = set(range(mol.GetNumAtoms())) - set().union(group)
            groups_mol = get_groups_mol(mol, left_atoms)
            
            # 如果group_in_sharedOneAtom为True，则需要从shared_oneAtom中获取额外的原子
            if group_in_sharedOneAtom:
                # 遍历shared_oneAtom，找到包含group的元素
                for info in shared_oneAtom:
                    if group == info[0] or group == info[1]:
                        left_atoms.update(info[2])  # 添加额外的原子
                        
            if group_in_sharedOneBond:
                # 遍历shared_oneAtom，找到包含group的元素
                for info in shared_oneBond:
                    if group == info[0] or group == info[1]:
                        left_atoms.update(info[2])  # 添加额外的原子
                        
            if not left_atoms:  # 如果剩余的原子为空，说明分子已经被完全分割
                return successful_groups, start_mol

            # 将剩余的原子转换为SMILES表示
            left_smiles = atom_group_to_smiles(mol, left_atoms)
            
            # 如果剩余的部分不是孤立的分子（即没有‘.’符号）
            if '.' not in left_smiles:
                #print(left_smiles)
                if group_in_sharedOneAtom:
                    bond_type = 4
                    left_mol = Chem.MolFromSmiles(left_smiles)
                    
                    if left_mol is None:  # 如果无法从SMILES创建分子，抛出异常
                        raise ValueError(f"Failed to create molecule from SMILES: {left_smiles}")
                    
                    # 遍历分子中的所有原子，查找原子映射编号为3的原子并处理                  
                    left_mol, connected_atoms = process_connected_atoms(left_mol, 3, connected_atoms)
                    groups_mol, connected_group_atoms = process_connected_atoms(groups_mol, 3, connected_group_atoms)
                    
                elif group_in_sharedOneBond:
                    bond_type = 5
                    left_mol = Chem.MolFromSmiles(left_smiles)
                    if left_mol is None:  # 如果无法从SMILES创建分子，抛出异常
                        raise ValueError(f"Failed to create molecule from SMILES: {left_smiles}")
                                     
                    left_mol, connected_atoms = process_connected_bond(left_mol, 4, connected_atoms)
                    groups_mol, connected_group_atoms = process_connected_bond(groups_mol, 4, connected_group_atoms)
                    
                    new_mol = add_group(left_mol, groups_mol, connected_atoms, connected_group_atoms, bond_type=bond_type)
                    new_smiles = Chem.MolToSmiles(new_mol)
                    
                    # 复制原始分子mol
                    origin_mol = Chem.Mol(mol)
                    # 清除原子映射编号
                    for atom in origin_mol.GetAtoms():
                        atom.SetAtomMapNum(0)  
                    origin_smiles = Chem.MolToSmiles(origin_mol)
                    
                    # print(new_smiles)
                    # print(origin_smiles)
                    if new_smiles == origin_smiles:
                        bond_type = 5
                    else :
                        bond_type = 6
                        
                else:
                    bond_type = get_bond_type(mol, group, left_atoms)

                    # 获取剩余分子的结构
                    groups_mol = get_groups_mol(mol, left_atoms)
                    left_mol = Chem.MolFromSmiles(left_smiles)
                    
                    if left_mol is None:  # 如果无法从SMILES创建分子，抛出异常
                        raise ValueError(f"Failed to create molecule from SMILES: {left_smiles}")
                    
                    left_mol, connected_atoms = process_connected_atoms(left_mol, 1, connected_atoms)
                    groups_mol, connected_group_atoms = process_connected_atoms(groups_mol, 2, connected_group_atoms)
                                
                # 将成功分离的原子组、连接原子和连接的原子组原子添加到结果中
                successful_groups.append((groups_mol, connected_atoms, connected_group_atoms, bond_type))
                atom_groups, shared_oneAtom, shared_oneBond = get_atom_groups(left_mol)  # 更新原子组
                mol = left_mol
                removed = True  # 标记当前成功移除原子组
                break  # 跳出当前原子组的循环

        if not removed:  # 如果没有成功移除任何原子组，则退出循环
            break

    return successful_groups, start_mol  # 返回成功分离的原子组和初始分子


def replace_nth(pattern, replacement, text, n):
    matches = list(pattern.finditer(text))
    if n > len(matches) or n < 1:
        return text  # 无需替换
    match = matches[n-1]
    start, end = match.span()
    return text[:start] + replacement + text[end:]


def add_group(mol, group, connected_atoms, connected_group_atoms, bond_type):
    if connected_atoms==[] or connected_group_atoms==[]:
        return None
    
    # 定义键类型映射
    bond_order_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE
    }
    
    if bond_type not in bond_order_map and bond_type not in {4, 5, 6}:
        raise ValueError(f"Invalid bond type: {bond_type}")
    
    # 合并原始分子和基团，并转换为 RWMol 进行编辑
    combined_mol = Chem.CombineMols(mol, group)
    em = Chem.RWMol(combined_mol)

    offset = mol.GetNumAtoms()  # 基团的原子索引偏移量

    if bond_type == 4:  # 合并原子
        # 选择一个原子进行合并，假设连接列表中只有一个元素
        atom_to_merge = connected_atoms[0]  # 从原分子中选取要合并的原子
        group_atom_to_merge = connected_group_atoms[0]  # 从基团中选取要合并的原子

        # 获取要移除的原子
        group_atom = em.GetAtomWithIdx(group_atom_to_merge + offset)
        # 获取要移除的原子连接的所有键
        atom_bonds = group_atom.GetBonds()
        # 记录移除原子连接的所有其他原子及其键类型
        other_atoms_and_bonds = [(bond.GetOtherAtom(group_atom).GetIdx(), bond.GetBondType()) for bond in atom_bonds]
        
        # 重新连接原子，保留原有的键类型
        for other_atom, bond_type in other_atoms_and_bonds:
            em.AddBond(atom_to_merge, other_atom, order=bond_type)
        
        # 移除其中一个原子（选择基团中的原子进行移除）
        em.RemoveAtom(group_atom_to_merge + offset)
        
    elif bond_type == 5 or bond_type == 6:  # 合并原子
        # 获取分子1中给定键的原子
        bond1 = mol.GetBondWithIdx(connected_atoms[0])
        atom1a = bond1.GetBeginAtom()
        atom1b = bond1.GetEndAtom()

        # 获取分子2中给定键的原子
        bond2 = group.GetBondWithIdx(connected_group_atoms[0])
        atom2a = bond2.GetBeginAtom()
        atom2b = bond2.GetEndAtom()
        if bond_type == 5:
            connected_atoms_a = [atom for atom in atom2a.GetNeighbors() if atom.GetIdx() != atom2b.GetIdx()]
            for atom in connected_atoms_a:
                bond_type_a = em.GetBondBetweenAtoms(atom2a.GetIdx() + offset, atom.GetIdx() + offset).GetBondType()
                em.AddBond(atom.GetIdx() + offset, atom1a.GetIdx(), bond_type_a)
            
            connected_atoms_b = [atom for atom in atom2b.GetNeighbors() if atom.GetIdx() != atom2a.GetIdx()]
            for atom in connected_atoms_b:
                bond_type_b = em.GetBondBetweenAtoms(atom2b.GetIdx() + offset, atom.GetIdx() + offset).GetBondType()
                em.AddBond(atom.GetIdx() + offset, atom1b.GetIdx(), bond_type_b)
        else:
            connected_atoms_a = [atom for atom in atom2a.GetNeighbors() if atom.GetIdx() != atom2b.GetIdx()]
            for atom in connected_atoms_a:
                bond_type_a = em.GetBondBetweenAtoms(atom2a.GetIdx() + offset, atom.GetIdx() + offset).GetBondType()
                em.AddBond(atom.GetIdx() + offset, atom1b.GetIdx(), bond_type_a)
            
            connected_atoms_b = [atom for atom in atom2b.GetNeighbors() if atom.GetIdx() != atom2a.GetIdx()]
            for atom in connected_atoms_b:
                bond_type_b = em.GetBondBetweenAtoms(atom2b.GetIdx() + offset, atom.GetIdx() + offset).GetBondType()
                em.AddBond(atom.GetIdx() + offset, atom1a.GetIdx(), bond_type_b)
                
        # 获取两个原子的索引并加上偏移量
        idx1 = atom2a.GetIdx() + offset
        idx2 = atom2b.GetIdx() + offset

        # 按逆序删除原子
        if idx1 > idx2:
            em.RemoveAtom(idx1)
            em.RemoveAtom(idx2)
        else:
            em.RemoveAtom(idx2)
            em.RemoveAtom(idx1)

    else:
        # 添加正常的键
        bond_order = bond_order_map[bond_type]
        for atom_idx, group_atom_idx in zip(connected_atoms, connected_group_atoms):
            em.AddBond(atom_idx, group_atom_idx + offset, order=bond_order)

    new_mol = em.GetMol()

    try:
        Chem.SanitizeMol(new_mol)
        return new_mol
    except Exception:
        # 获取原始 SMILES
        original_smiles = Chem.MolToSmiles(new_mol)
        # 编译正则表达式以提高性能
        bracket_pattern = re.compile(r'\[([^\]]+)\]')
        bracket_contents = bracket_pattern.findall(original_smiles)

        if not bracket_contents:
            return new_mol

        # 尝试单个部分的修正
        for idx, content in enumerate(bracket_contents, start=1):
            # 查找所有 H 原子及其数量
            h_matches = re.findall(r'H(\d*)', content, re.IGNORECASE)
            total_h = sum(int(count) if count else 1 for count in h_matches)
            if total_h == 0:
                continue  # 当前部分没有 H 原子，无需处理

            # 逐步减少 H 的数量
            for reduce_h in range(1, total_h + 1):
                remaining_h = total_h - reduce_h
                if remaining_h > 1:
                    new_h = f'H{remaining_h}'
                elif remaining_h == 1:
                    new_h = 'H'
                else:
                    new_h = ''

                # 修改内容中的 H 数量
                modified_content = re.sub(r'H\d*', new_h, content, flags=re.IGNORECASE)
                # 构造替换后的 SMILES
                replacement_options = [f'[{modified_content}]', modified_content] if remaining_h == 0 else [f'[{modified_content}]']
                        # 尝试替换并生成分子
                for replacement in replacement_options:
                    new_smiles = replace_nth(bracket_pattern, replacement, original_smiles, idx)
                    fixed_mol = Chem.MolFromSmiles(new_smiles)
                
                    if not fixed_mol:
                        continue  # SMILES 无效，尝试下一个修改
                    try:
                        Chem.SanitizeMol(fixed_mol)
                        return fixed_mol  # 成功修正并消毒分子
                    except Exception:
                        continue  # 修正失败，尝试下一个修改

        # 尝试同时从两个不同的部分各移除一个 H
        if len(bracket_contents) >= 2:
            # 生成所有两个不同部分的组合
            for (idx1, content1), (idx2, content2) in itertools.combinations(enumerate(bracket_contents, start=1), 2):
                # 查找第一个部分的 H 数量
                h_matches1 = re.findall(r'H(\d*)', content1, re.IGNORECASE)
                total_h1 = sum(int(count) if count else 1 for count in h_matches1)
                if total_h1 == 0:
                    continue  # 第一个部分没有 H 原子，无需处理

                # 查找第二个部分的 H 数量
                h_matches2 = re.findall(r'H(\d*)', content2, re.IGNORECASE)
                total_h2 = sum(int(count) if count else 1 for count in h_matches2)
                if total_h2 == 0:
                    continue  # 第二个部分没有 H 原子，无需处理

                # 只移除一个 H 从每个部分
                remaining_h1 = total_h1 - 1
                remaining_h2 = total_h2 - 1

                # 构造新的 H 表达式
                new_h1 = f'H{remaining_h1}' if remaining_h1 > 1 else ('H' if remaining_h1 == 1 else '')
                new_h2 = f'H{remaining_h2}' if remaining_h2 > 1 else ('H' if remaining_h2 == 1 else '')

                # 修改内容中的 H 数量
                modified_content1 = re.sub(r'H\d*', new_h1, content1, flags=re.IGNORECASE)
                modified_content2 = re.sub(r'H\d*', new_h2, content2, flags=re.IGNORECASE)
                
                # 构造替换后的 SMILES
                replacement1_options = [f'[{modified_content1}]', modified_content1] if remaining_h1 == 0 else [f'[{modified_content1}]']
                replacement2_options = [f'[{modified_content2}]', modified_content2] if remaining_h2 == 0 else [f'[{modified_content2}]']

                # 为了避免索引变化，按降序替换
                if idx1 > idx2:
                    first_idx, first_repl = idx1, replacement1_options
                    second_idx, second_repl = idx2, replacement2_options
                else:
                    first_idx, first_repl = idx2, replacement2_options
                    second_idx, second_repl = idx1, replacement1_options

                # 首先替换索引较大的部分
                for replacement1 in first_repl:
                    temp_smiles = replace_nth(bracket_pattern, replacement1, original_smiles, first_idx)
                    # 然后替换索引较小的部分
                    for replacement2 in second_repl:
                        new_smiles = replace_nth(bracket_pattern, replacement2, temp_smiles, second_idx)

                        # 尝试生成并消毒新的分子
                        fixed_mol = Chem.MolFromSmiles(new_smiles)
                        if not fixed_mol:
                            continue  # SMILES 无效，尝试下一个修改

                        try:
                            Chem.SanitizeMol(fixed_mol)
                            return fixed_mol  # 成功修正并消毒分子
                        except Exception:
                            continue  # 修正失败，尝试下一个修改

        # 如果所有修正尝试均失败，返回原始分子
        return new_mol


def replace_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    # 去除立体结构和手性
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetStereo(Chem.BondStereo.STEREONONE)

    end_smiles = Chem.MolToSmiles(mol)
    end_smiles = end_smiles.replace("[SiH2]", "[Si]")
    end_smiles = end_smiles.replace("[SH3+]", "[S+]")
    end_smiles = end_smiles.replace("[SeH2]", "[Se]")
    end_smiles = end_smiles.replace("[CH3]", "C").replace("[CH2]", "C").replace("[CH]", "C").replace("[C]", "C")
    end_smiles = end_smiles.replace("[NH4]", "N").replace("[NH3]", "N").replace("[NH2]", "N").replace("[NH]", "N").replace("[N]", "N")
    end_smiles = end_smiles.replace("[SH5]", "S").replace("[SH4]", "S").replace("[SH3]", "S").replace("[SH2]", "S").replace("[SH]", "S").replace("[S]", "S")
    smiles = end_smiles.replace("[PH5]", "P").replace("[PH4]", "P").replace("[PH3]", "P").replace("[PH2]", "P").replace("[PH]", "P").replace("[P]", "P")
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    smiles = Chem.MolToSmiles(mol)
    return smiles


def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        new_smiles = Chem.MolToSmiles(mol)
        return True
    except:
        return False