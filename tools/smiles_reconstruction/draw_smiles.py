from rdkit import Chem
from rdkit.Chem import Draw

# SMILES字符串
smiles = 'O=C(NCc1cccc(F)c1F)c1ccccc1OC1CC1'

# 从SMILES字符串创建分子
mol = Chem.MolFromSmiles(smiles)

# 创建一个SVG drawer
drawer = Draw.MolDraw2DSVG(300, 300)
# 设置原子标签的字体大小
drawer.drawOptions().atomLabelFontSize = 500
# 设置背景为透明
drawer.drawOptions().clearBackground = False

# 准备并绘制分子
drawer.DrawMolecule(mol)

# 结束绘制
drawer.FinishDrawing()

# 获取结果
svg = drawer.GetDrawingText()
# 将SVG保存到文件
with open('molecule1.svg', 'w') as f:
    f.write(svg)
