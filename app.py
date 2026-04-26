import streamlit as st
import pandas as pd
import warnings
import re
from pymatgen.io.vasp import Incar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet

# 屏蔽 Pymatgen 的冗余终端警告
warnings.filterwarnings("ignore", message=".*Too few KPOINTS.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

# ==========================================
# 页面基础配置
# ==========================================
st.set_page_config(page_title="VASP 智能诊断系统", layout="wide", page_icon="🔬")

def parse_vasp_bool(val):
    if isinstance(val, bool): return val
    if isinstance(val, str):
        return val.strip().upper() in [".TRUE.", "TRUE", "T", "1"]
    if isinstance(val, int):
        return val == 1
    return False

# ==========================================
# 元素磁性与 DFT+U 知识库 (来自 parameters.py)
# ==========================================
ELEMENT_MAGNETIC_MOMENTS = {
    'H': 0, 'He': 0, 'Li': 0, 'Be': 0, 'B': 0, 'C': 0, 'N': 0, 'O': 0, 'F': 0, 'Ne': 0,
    'Na': 0, 'Mg': 0, 'Al': 0, 'Si': 0, 'P': 0, 'S': 0, 'Cl': 0, 'Ar': 0,
    'K': 0, 'Ca': 0, 'Sc': 0, 'Ti': 1, 'V': 3, 'Cr': 5, 'Mn': 5, 'Fe': 4, 'Co': 3, 'Ni': 2, 'Cu': 1,
    'Zn': 0, 'Ga': 0, 'Ge': 0, 'As': 0, 'Se': 0, 'Br': 0, 'Kr': 0,
    'Rb': 0, 'Sr': 0, 'Y': 1, 'Zr': 1, 'Nb': 1, 'Mo': 1, 'Tc': 1, 'Ru': 1, 'Rh': 1, 'Pd': 0, 'Ag': 1,
    'Cd': 0, 'In': 0, 'Sn': 0, 'Sb': 0, 'Te': 0, 'I': 0, 'Xe': 0,
    'Cs': 0, 'Ba': 0, 'La': 3, 'Ce': 1, 'Pr': 3, 'Nd': 3, 'Pm': 4, 'Sm': 5, 'Eu': 7, 'Gd': 7,
    'Tb': 6, 'Dy': 5, 'Ho': 4, 'Er': 3, 'Tm': 2, 'Yb': 0, 'Lu': 0,
    'Hf': 1, 'Ta': 1, 'W': 1, 'Re': 1, 'Os': 1, 'Ir': 1, 'Pt': 0, 'Au': 0, 'Hg': 0,
    'Tl': 0, 'Pb': 0, 'Bi': 0, 'Po': 0, 'At': 0, 'Rn': 0,
    'U': 3, 'Np': 4, 'Pu': 6, 'Am': 6
}

DFT_U_VALUES = {
    'Ti': {'d': 3.5, 'f': 0, 'notes': 'TiO2等氧化物推荐值'},
    'V': {'d': 3.5, 'f': 0, 'notes': '钒氧化物推荐值'},
    'Cr': {'d': 3.5, 'f': 0, 'notes': 'Cr2O3: ~4.0 eV'},
    'Mn': {'d': 3.5, 'f': 0, 'notes': 'MnO: ~3.5 eV; 金属间化合物: ~2.0 eV'},
    'Fe': {'d': 3.5, 'f': 0, 'notes': 'FeO: ~4.0 eV; Fe2O3: ~4.5 eV'},
    'Co': {'d': 3.5, 'f': 0, 'notes': 'CoO: ~3.5 eV'},
    'Ni': {'d': 4.0, 'f': 0, 'notes': 'NiO: ~4.0-6.0 eV'},
    'Cu': {'d': 4.0, 'f': 0, 'notes': 'Cu2O: ~5.0 eV'},
    'Zn': {'d': 0, 'f': 0}, 'Y': {'d': 0, 'f': 0}, 'Zr': {'d': 0, 'f': 0},
    'Nb': {'d': 0, 'f': 0}, 'Mo': {'d': 0, 'f': 0}, 'Tc': {'d': 0, 'f': 0},
    'Ru': {'d': 0, 'f': 0}, 'Rh': {'d': 0, 'f': 0}, 'Pd': {'d': 0, 'f': 0},
    'Ag': {'d': 0, 'f': 0}, 'Hf': {'d': 0, 'f': 0}, 'Ta': {'d': 0, 'f': 0},
    'W': {'d': 0, 'f': 0}, 'Re': {'d': 0, 'f': 0}, 'Os': {'d': 0, 'f': 0},
    'Ir': {'d': 0, 'f': 0}, 'Pt': {'d': 0, 'f': 0}, 'Au': {'d': 0, 'f': 0},
    'La': {'d': 0, 'f': 6}, 'Ce': {'d': 0, 'f': 5}, 'Pr': {'d': 0, 'f': 5},
    'Nd': {'d': 0, 'f': 5}, 'Pm': {'d': 0, 'f': 5}, 'Sm': {'d': 0, 'f': 5},
    'Eu': {'d': 0, 'f': 6}, 'Gd': {'d': 0, 'f': 6}, 'Tb': {'d': 0, 'f': 5},
    'Dy': {'d': 0, 'f': 5}, 'Ho': {'d': 0, 'f': 5}, 'Er': {'d': 0, 'f': 5},
    'Tm': {'d': 0, 'f': 5}, 'Yb': {'d': 0, 'f': 4}, 'Lu': {'d': 0, 'f': 0},
    'U': {'d': 0, 'f': 4}, 'Np': {'d': 0, 'f': 4}, 'Pu': {'d': 0, 'f': 4}, 'Am': {'d': 0, 'f': 4}
}

# ==========================================
# 全量无删减 VASP 参数百科 (1:1 复制自 parameters.py)
# ==========================================
INCAR_PARAMS_RAW = {
    'SYSTEM': {'category': '基础设置', 'chinese_name': '系统名称', 'description': '计算的系统名称或注释', 'physical_meaning': '用于标识计算的字符串，不影响计算结果，但有助于文件管理', 'recommendation': '建议设置，描述计算内容如 "Fe2O3 static calculation"'},
    'ISTART': {'category': '基础设置', 'chinese_name': '波函数初始化', 'description': '波函数初始化选项', 'physical_meaning': '控制是否从磁盘读取初始波函数。0=从头开始，1=读取CHGCAR继续，2=使用WAVECAR，3=仅读取K点', 'recommendation': '首次计算用0；续算用1；需要继续SCF用2', 'warnings':['续算时需确保CHGCAR/WAVECAR存在']},
    'ICHARG': {'category': '基础设置', 'chinese_name': '电荷密度初始化', 'description': '电荷密度初始化方式', 'physical_meaning': '控制如何初始化电荷密度。0=从原子叠加计算，1=读取CHGCAR，2=从原子密度插值，11=从CHGCAR扣除', 'recommendation': '自洽计算用0；读取电荷用1；能带计算用11或12', 'warnings':['能带/DOS计算常用ICHARG=11跳过自洽']},
    'ENCUT': {'category': '基础设置', 'chinese_name': '平面波截断能', 'description': '平面波截断能 (单位: eV)', 'physical_meaning': '平面波基组的动能截断值。决定了计算精度：值越大精度越高', 'recommendation': '通常设为最大ENMAX的1.0-1.3倍；高精度用1.3倍', 'warnings': ['ENCUT过低会导致结果不可靠；过高增加计算成本']},
    'PREC': {'category': '基础设置', 'chinese_name': '计算精度', 'description': '计算精度控制', 'physical_meaning': '设置计算的基本精度级别，影响FFT网格、基底等', 'recommendation': '精度要求高用Accurate；常规用Normal；快速测试用Low', 'warnings': ['高精度计算建议用PREC=Accurate']},
    'EDIFF': {'category': '电子求解器', 'chinese_name': '电子收敛标准', 'description': '电子自洽收敛标准 (单位: eV)', 'physical_meaning': '相邻两次电子步之间总能量差的阈值。低于此值认为收敛', 'recommendation': '常规计算1E-5；高精度1E-6；测试可用1E-4', 'warnings':['金属体系收敛较难，可能需要更宽松的标准']},
    'EDIFFG': {'category': '电子求解器', 'chinese_name': '离子收敛标准', 'description': '离子弛豫收敛标准', 'physical_meaning': '几何优化的收敛判据。正值=能量收敛标准，负值=力收敛标准(绝对值)', 'recommendation': '能量收敛: 1E-5；力收敛: -1E-2 到 -1E-3 (eV/Å)', 'warnings': ['力收敛通常比能量收敛更严格']},
    'IALGO': {'category': '电子求解器', 'chinese_name': '电子算法', 'description': '电子结构优化算法', 'physical_meaning': '选择用于对角化和子空间旋转的算法。', 'recommendation': '38=Davidson(通用)，48=RMM-DIIS(快速但不稳定)', 'warnings':['RMM-DIIS对某些体系可能不收敛']},
    'ISMEAR': {'category': '电子求解器', 'chinese_name': '占据数展宽', 'description': 'occupancy smearing 方法', 'physical_meaning': '如何处理能带占据的smearing。影响金属的收敛性', 'recommendation': '金属用1或2，绝缘体/半导体用-5或0', 'warnings':['金属体系或弛豫时绝对不能用ISMEAR=-5']},
    'SIGMA': {'category': '电子求解器', 'chinese_name': '展宽宽度', 'description': 'Smearing 宽度 (单位: eV)', 'physical_meaning': 'Gaussian或MP smearing的展宽参数。影响收敛速度和精度', 'recommendation': '金属: 0.05-0.2；绝缘体: 0.01-0.05', 'warnings': ['SIGMA过小收敛慢，过大引入误差']},
    'ISPIN': {'category': '磁性设置', 'chinese_name': '自旋极化', 'description': '自旋极化开关', 'physical_meaning': '控制是否进行自旋极化计算。1=不考虑自旋，2=考虑自旋上下', 'recommendation': '磁性材料必须设2', 'warnings': ['磁性材料不设ISPIN=2会丢失磁性信息']},
    'MAGMOM': {'category': '磁性设置', 'chinese_name': '原子磁矩', 'description': '初始原子磁矩设置 (单位: μB)', 'physical_meaning': '每个原子的初始自旋磁矩。正值=自旋向上，负值=自旋向下', 'recommendation': '按元素设置：Fe=4, Co=3, Ni=2, Mn=5', 'warnings':['初始磁矩过大可能导致磁矩锁定在亚稳态']},
    'LSORBIT': {'category': '磁性设置', 'chinese_name': '自旋轨道耦合', 'description': '自旋轨道耦合开关', 'physical_meaning': '是否包含相对论自旋轨道耦合(SOC)效应。重元素必须考虑', 'recommendation': '重元素磁性材料、拓扑绝缘体等需要设为.TRUE.', 'warnings': ['开启SOC后计算量大增']},
    'ICHIBERN': {'category': '磁性设置', 'chinese_name': '磁化方向', 'description': '初始磁密方向设置', 'physical_meaning': '非共线磁性中，初始磁化方向的空间分布', 'recommendation': '通常用1', 'warnings':[]},
    'LDAU': {'category': 'DFT+U', 'chinese_name': 'DFT+U开关', 'description': '是否启用DFT+U', 'physical_meaning': '对强关联d/f电子添加Hubbard U校正，处理电子定域化问题', 'recommendation': '过渡金属氧化物、稀土化合物等强关联体系设为.TRUE.', 'warnings': ['U值不当会导致结果错误']},
    'LDAUTYPE': {'category': 'DFT+U', 'chinese_name': 'DFT+U方法', 'description': 'DFT+U方法类型', 'physical_meaning': '选择DFT+U的具体实现方法', 'recommendation': '2=Dudarev(最常用，只需U值)', 'warnings':['Dudarev方法只与U-J有关']},
    'LDAUL': {'category': 'DFT+U', 'chinese_name': 'U值作用轨道', 'description': '每个元素应用U的角动量通道', 'physical_meaning': '对哪些角动量加U。2=d轨道，3=f轨道，-1=不加U', 'recommendation': '过渡金属氧化物通常用2(d轨道)；稀土用3(f轨道)', 'warnings': ['需与元素顺序对应']},
    'LDAUU': {'category': 'DFT+U', 'chinese_name': 'U值', 'description': '每个元素的U值 (单位: eV)', 'physical_meaning': 'Hubbard U参数，描述电子-电子相互作用强度', 'recommendation': 'Fe/Co/Ni/Mn等3d金属: 3-5 eV', 'warnings':['U值需参考文献']},
    'LDAUJ': {'category': 'DFT+U', 'chinese_name': 'J值', 'description': '每个元素的J值 (单位: eV)', 'physical_meaning': 'Hund交换参数J。对于Dudarev方法通常设为0', 'recommendation': 'Dudarev方法设为0', 'warnings':[]},
    'LDAUPRINT': {'category': 'DFT+U', 'chinese_name': 'DFT+U输出', 'description': 'DFT+U输出控制', 'physical_meaning': '控制是否输出DFT+U相关的详细信息', 'recommendation': '调试用1或2；正常计算用0', 'warnings':[]},
    'GGA': {'category': '交换关联泛函', 'chinese_name': 'GGA泛函', 'description': 'GGA泛函类型', 'physical_meaning': '选择广义梯度近似(GGA)的具体形式', 'recommendation': 'PE=PBE(最常用)；PS=PW91', 'warnings':[]},
    'METAGGA': {'category': '交换关联泛函', 'chinese_name': 'Meta-GGA泛函', 'description': 'meta-GGA泛函', 'physical_meaning': '使用包含动能密度的meta-GGA泛函', 'recommendation': 'SCAN(强相关体系)', 'warnings':[]},
    'LHFCALC': {'category': '交换关联泛函', 'chinese_name': 'HF混合', 'description': 'Hartree-Fock混合开关', 'physical_meaning': '是否混合Hartree-Fock交换。HSE06等需要', 'recommendation': 'HSE06计算设为.TRUE.', 'warnings': ['杂化泛函计算量显著增加']},
    'AEXX': {'category': '交换关联泛函', 'chinese_name': 'HF交换比例', 'description': 'HF交换比例', 'physical_meaning': 'Hartley-Fock交换在杂化泛函中的比例', 'recommendation': 'HSE06: 0.25', 'warnings':[]},
    'HFSCREEN': {'category': '交换关联泛函', 'chinese_name': 'HF屏蔽参数', 'description': 'HF屏蔽参数 (HSE专用)', 'physical_meaning': 'HSE短程/长程交换的分离参数', 'recommendation': 'HSE06: 0.207', 'warnings':[]},
    'IBRION': {'category': '几何优化', 'chinese_name': '离子优化算法', 'description': '离子弛豫/优化算法', 'physical_meaning': '选择原子位置优化的算法', 'recommendation': '-1=固定；2=共轭梯度(常用)', 'warnings': ['IBRION=2需设置ISIF']},
    'ISIF': {'category': '几何优化', 'chinese_name': '优化自由度', 'description': '优化自由度控制', 'physical_meaning': '控制哪些自由度允许改变', 'recommendation': '2=仅原子位置；3=原子+体积', 'warnings':['ISIF=3不能用于二维表面材料']},
    'NSW': {'category': '几何优化', 'chinese_name': '最大离子步数', 'description': '最大离子步数', 'physical_meaning': '离子弛豫或分子动力学模拟的最大步数', 'recommendation': '几何优化: 100-300', 'warnings':['NSW=0为静态计算']},
    'ISYM': {'category': '几何优化', 'chinese_name': '对称性', 'description': '对称性开关', 'physical_meaning': '是否利用晶体对称性减少计算量', 'recommendation': '默认1=使用；0=不使用', 'warnings':['SOC计算需关闭对称性(ISYM=-1)']},
    'LWAVE': {'category': '输出控制', 'chinese_name': '波函数输出', 'description': '是否输出波函数', 'physical_meaning': '控制WAVECAR文件的写入', 'recommendation': '续算用.TRUE.', 'warnings':[]},
    'LCHARG': {'category': '输出控制', 'chinese_name': '电荷密度输出', 'description': '是否输出电荷密度', 'physical_meaning': '控制CHGCAR文件的写入', 'recommendation': '能带/DOS前需输出.TRUE.', 'warnings':[]},
    'LVTOT': {'category': '输出控制', 'chinese_name': '静电势输出', 'description': '是否输出静电势', 'physical_meaning': '输出LOCPOT文件', 'recommendation': '分析功函数时设为.TRUE.', 'warnings':[]},
    'NELECT': {'category': '输出控制', 'chinese_name': '电子总数', 'description': '电子总数', 'physical_meaning': '强制设置体系的总电子数', 'recommendation': '缺陷加电荷计算时手动设置', 'warnings': ['设错会导致错误结果']},
    'SMASS': {'category': '分子动力学', 'chinese_name': '热浴参数', 'description': 'Nose-Hoover chain参数', 'physical_meaning': '控制分子动力学中的热浴设置', 'recommendation': '-3=NVT；-1=NVE', 'warnings':[]},
    'TEBEG': {'category': '分子动力学', 'chinese_name': '初始温度', 'description': 'MD初始温度 (K)', 'physical_meaning': '分子动力学模拟的初始温度', 'recommendation': '300 (室温)', 'warnings':[]},
    'TEEND': {'category': '分子动力学', 'chinese_name': '结束温度', 'description': 'MD结束温度 (K)', 'physical_meaning': '如果TEEND≠TEBEG，进行温度斜坡MD', 'recommendation': '恒温MD时设为与TEBEG相同', 'warnings':[]},
    'IMAGES': {'category': 'NEB计算', 'chinese_name': '中间图像数', 'description': 'NEB中间图像数', 'physical_meaning': '初态和终态之间的中间结构数', 'recommendation': '简单反应用3-5', 'warnings': ['需配合IBRION=3使用']},
    'NELM': {'category': '高级设置', 'chinese_name': '最大电子迭代', 'description': '最大电子自洽迭代次数', 'physical_meaning': '电子步SCF循环的最大次数', 'recommendation': '常规60；难收敛体系200+', 'warnings':[]},
    'NELMIN': {'category': '高级设置', 'chinese_name': '最小电子迭代', 'description': '最小电子迭代次数', 'physical_meaning': '即使已收敛，也最少迭代的步数', 'recommendation': '通常2-6', 'warnings':[]},
    'NPAR': {'category': '高级设置', 'chinese_name': '并行参数', 'description': '并行化参数', 'physical_meaning': '控制k点和频带并行', 'recommendation': '1或NCORE的约数', 'warnings':[]},
    'NCORE': {'category': '高级设置', 'chinese_name': '并行能带数', 'description': '每组-core处理的能带数', 'physical_meaning': '控制计算并行度', 'recommendation': '1-16，根据核数调整', 'warnings': ['与NPAR二选一']},
    'AMIX': {'category': '电荷混合', 'chinese_name': '电荷混合参数', 'description': '电荷密度混合参数', 'physical_meaning': '控制SCF循环中电荷密度的混合比例', 'recommendation': '难收敛用0.1-0.2', 'warnings': ['过大可能震荡']},
    'BMIX': {'category': '电荷混合', 'chinese_name': 'Kerker衰减长度', 'description': '电荷密度混合的Kerker衰减长度', 'physical_meaning': '在倒空间中使用衰减的长程混合', 'recommendation': '通常保持默认', 'warnings':[]},
    'AMIX_MAG': {'category': '电荷混合', 'chinese_name': '磁性混合参数', 'description': '磁性体系电荷密度混合参数', 'physical_meaning': '单独控制自旋通道的混合比例', 'recommendation': '磁性体系难收敛可增至1.6-2.0', 'warnings':[]},
    'BMIX_MAG': {'category': '电荷混合', 'chinese_name': '磁性Kerker衰减', 'description': '磁性体系Kerker衰减长度', 'physical_meaning': '自旋通道的衰减长度', 'recommendation': '通常保持默认', 'warnings':[]},
    'MAXMIX': {'category': '电荷混合', 'chinese_name': 'Broyden迭代历史', 'description': '混合的最大迭代历史', 'physical_meaning': '储存历史电荷密度用于更智能的混合', 'recommendation': '难收敛体系可增大到40', 'warnings':[]},
    'LREAL': {'category': '基础设置', 'chinese_name': '实空间投影', 'description': '实空间投影开关', 'physical_meaning': '选择在实空间还是倒空间进行局域投影', 'recommendation': '大体系可用Auto；高精度须False', 'warnings':[]},
    'VOSKOWN': {'category': '交换关联泛函', 'chinese_name': 'VWN插值', 'description': 'VWN 插值开关', 'physical_meaning': 'LDA中使用VWN公式插值', 'recommendation': '使用LDA时建议设为1', 'warnings':[]},
    'NWRITE': {'category': '输出控制', 'chinese_name': 'OUTCAR写入频率', 'description': 'OUTCAR写入频率', 'physical_meaning': '控制写入OUTCAR的信息量', 'recommendation': '2=详细(推荐)', 'warnings':[]},
    'INIWAV': {'category': '基础设置', 'chinese_name': '初始波函数', 'description': '初始波函数生成方式', 'physical_meaning': '选择如何生成初始波函数', 'recommendation': '1=原子叠加(默认)', 'warnings':[]},
    'ADDGRID': {'category': '高级设置', 'chinese_name': '额外FFT网格', 'description': '添加额外FFT网格', 'physical_meaning': '在计算电荷密度时使用更密的FFT网格', 'recommendation': '高精度计算建议.TRUE.', 'warnings':[]},
    'LSCALAPACK': {'category': '高级设置', 'chinese_name': 'ScaLAPACK并行', 'description': 'ScaLAPACK并行对角化开关', 'physical_meaning': '使用ScaLAPACK库', 'recommendation': '大体系可显式开启', 'warnings':[]},
    'POTIM': {'category': '分子动力学', 'chinese_name': '时间步长', 'description': '离子时间步长', 'physical_meaning': '弛豫移动缩放或MD时间步长', 'recommendation': '弛豫炸裂时减小至0.1', 'warnings':[]},
    'RWIGS': {'category': '基础设置', 'chinese_name': 'Wigner-Seitz半径', 'description': '原子Wigner-Seitz半径 (Ang)', 'physical_meaning': '每个元素的原子半径，用于计算Bader电荷、投影DOS等', 'recommendation': '通常设为共价半径的50-70%', 'warnings':['算DOS时如果不设此项，且LORBIT未配置好，投影会出错']},
    'RIMPODATA': {'category': '基础设置', 'chinese_name': '离子半径', 'description': '离子半径参数', 'physical_meaning': '用于某些分析工具', 'recommendation': '通常保持默认', 'warnings':[]},
    'NBLOCK': {'category': '输出控制', 'chinese_name': '写入间隔', 'description': 'CHGCAR/ENERGY写入间隔', 'physical_meaning': '每隔NBLOCK步写入一次', 'recommendation': 'MD中可增大', 'warnings':[]},
    'KBLOCK': {'category': '输出控制', 'chinese_name': '波函数块写入', 'description': '波函数写入的块大小', 'physical_meaning': 'NBLOCK*KBLOCK步后写入WAVECAR', 'recommendation': 'MD中可设较大值', 'warnings':[]},
    'LELF': {'category': '输出控制', 'chinese_name': '电子局域化函数', 'description': '计算电子局域化函数', 'physical_meaning': '输出ELF文件用于分析键', 'recommendation': '分析化学键时设.TRUE.', 'warnings':[]},
    'LVHAR': {'category': '输出控制', 'chinese_name': '静电势输出', 'description': 'Hartree势输出', 'physical_meaning': '输出LOCPOT包含静电势', 'recommendation': '通常.FALSE.', 'warnings':[]},
    'LORBIT': {'category': 'DOS计算', 'chinese_name': '局域态密度', 'description': '局域态密度输出控制', 'physical_meaning': '控制输出各轨道的分波态密度 PROCAR', 'recommendation': '算能带/DOS强烈推荐11', 'warnings':[]},
    'NEDOS': {'category': 'DOS计算', 'chinese_name': 'DOS能量点数', 'description': 'DOS计算的能量点数', 'physical_meaning': '决定态密度曲线的平滑度', 'recommendation': '高精度用1000-3000', 'warnings':[]},
    'EMAX': {'category': 'DOS计算', 'chinese_name': 'DOS能量范围', 'description': 'DOS计算的能量范围', 'physical_meaning': '设置能量网格的上限', 'recommendation': '自动设置', 'warnings':[]},
    'EMIN': {'category': 'DOS计算', 'chinese_name': 'DOS最小能量', 'description': '能量下限', 'physical_meaning': '能量网格下限', 'recommendation': '自动设置', 'warnings':[]},
    'WEIMIN': {'category': '高级设置', 'chinese_name': '权重最小值', 'description': '权重最小值', 'physical_meaning': '迭代子空间最小权重，防数值问题', 'recommendation': '难收敛可减小到0.0001', 'warnings':[]},
    'EBREAK': {'category': '高级设置', 'chinese_name': '电子收敛阈值', 'description': '电子步收敛判断', 'physical_meaning': '判断电子自洽收敛能量', 'recommendation': '通常自动', 'warnings':[]},
    'SYMPREC': {'category': '基础设置', 'chinese_name': '对称性精度', 'description': '对称性识别精度', 'physical_meaning': '判断原子等效容差', 'recommendation': '晶格精度低时改1E-6', 'warnings':[]},
    'SPRING': {'category': 'NEB计算', 'chinese_name': '弹簧力常数', 'description': 'NEB弹簧力常数', 'physical_meaning': '控制图像弹簧力强度', 'recommendation': '-5', 'warnings':[]},
    'LCLIMB': {'category': 'NEB计算', 'chinese_name': '爬坡', 'description': 'NEB爬坡开关', 'physical_meaning': '启用CI-NEB方法', 'recommendation': '标准NEB建议.TRUE.', 'warnings':[]},
    'ALGO': {'category': '基础设置', 'chinese_name': '宏观算法', 'description': '算法', 'physical_meaning': 'Normal(稳), Fast(快), Damped(杂化)', 'recommendation': '默认', 'warnings':[]},
    'NBANDS': {'category': '能带数', 'description': '包含的能带数量', 'physical_meaning': '总能带', 'recommendation': '光学计算需手动增加', 'warnings':[]},
    'KSPACING': {'category': 'K点设置', 'chinese_name': 'k点间距', 'description': '最大间距', 'physical_meaning': '自动生成K点网格间距', 'recommendation': '半导体0.2; 金属0.15', 'warnings':[]},
    'KGAMMA': {'category': 'K点设置', 'chinese_name': 'Gamma点', 'description': '是否居中Gamma', 'physical_meaning': 'Gamma居中网格', 'recommendation': '.TRUE.', 'warnings':[]},
    'NKRED': {'category': 'K点设置', 'chinese_name': 'k点缩减', 'description': 'k点缩减', 'physical_meaning': '减少计算量', 'recommendation': '1', 'warnings':[]},
    'NLSPLINE': {'category': 'K点设置', 'chinese_name': 'k点插值', 'description': '样条插值', 'physical_meaning': '高精度', 'recommendation': '.FALSE.', 'warnings':[]},
    'IVDW': {'category': '范德华力', 'chinese_name': '范德华校正', 'description': '色散校正', 'physical_meaning': '处理弱相互作用', 'recommendation': '11=D3; 12=D3(BJ)', 'warnings': ['层状材料必开']},
    'VDW_S6': {'category': '范德华力', 'chinese_name': 'D3比例因子', 'description': 'D3缩放', 'physical_meaning': 'D3缩放', 'recommendation': '默认', 'warnings':[]},
    'VDW_SR': {'category': '范德华力', 'chinese_name': 'D3短程', 'description': 'D3短程', 'physical_meaning': '短程缩放', 'recommendation': '默认', 'warnings':[]},
    'VDW_A1': {'category': '范德华力', 'chinese_name': 'D3_A1', 'description': 'D3_A1', 'physical_meaning': '参数', 'recommendation': '默认', 'warnings':[]},
    'VDW_A2': {'category': '范德华力', 'chinese_name': 'D3_A2', 'description': 'D3_A2', 'physical_meaning': '参数', 'recommendation': '默认', 'warnings':[]},
    'VDW_RADIUS': {'category': '范德华力', 'chinese_name': 'vdW半径', 'description': '截断半径', 'physical_meaning': '截断', 'recommendation': '默认', 'warnings':[]},
    'LUSE_VDW': {'category': '范德华力', 'chinese_name': '使用vdW', 'description': 'MBD方法', 'physical_meaning': 'MBD', 'recommendation': '.FALSE.', 'warnings':[]},
    'ENAUG': {'category': '基础设置', 'chinese_name': '增强截断能', 'description': 'PAW增强', 'physical_meaning': 'PAW截断', 'recommendation': '1.5-2倍ENCUT', 'warnings':[]},
    'ENCUTFOCK': {'category': '基础设置', 'chinese_name': 'FOCK截断能', 'description': '精确交换截断', 'physical_meaning': '精确交换截断', 'recommendation': '同ENCUT', 'warnings':[]},
    'ROPT': {'category': '基础设置', 'chinese_name': '投影精度', 'description': '投影参数', 'physical_meaning': '投影参数', 'recommendation': '-1E-3', 'warnings':[]},
    'LASPH': {'category': 'DFT+U', 'chinese_name': '非球面校正', 'description': '非球面校正', 'physical_meaning': '非球面电荷', 'recommendation': '加U和SOC建议开启', 'warnings':[]},
    'LMAXFOCK': {'category': 'DFT+U', 'chinese_name': 'Fock最大l', 'description': '角动量', 'physical_meaning': '角动量', 'recommendation': '0', 'warnings':[]},
    'LMAXMIX': {'category': 'DFT+U', 'chinese_name': '混合最大l', 'description': '混合最大角动量', 'physical_meaning': '电荷密度混合中包含的最高角动量', 'recommendation': '含d电子体系必设4，含f设6', 'warnings':['不设置会导致加U和磁性体系极难收敛！']},
    'MDALGO': {'category': '分子动力学', 'chinese_name': 'MD算法', 'description': 'MD积分', 'physical_meaning': 'MD积分', 'recommendation': '0=Verlet', 'warnings':[]},
    'LANGEVIN_GAMMA': {'category': '分子动力学', 'chinese_name': 'Langevin阻尼', 'description': '阻尼', 'physical_meaning': '阻尼', 'recommendation': '默认', 'warnings':[]},
    'PSTRESS': {'category': '分子动力学', 'chinese_name': '静水压', 'description': '静水压', 'physical_meaning': '外部压力', 'recommendation': '0.0', 'warnings':[]},
    'PMASS': {'category': '分子动力学', 'chinese_name': '离子赝质量', 'description': '质量', 'physical_meaning': '质量', 'recommendation': '0.0', 'warnings':[]},
    'LEPSILON': {'category': '光学计算', 'chinese_name': '高频介电', 'description': '介电常数', 'physical_meaning': '介电常数', 'recommendation': '光学开启', 'warnings':[]},
    'LOPTICS': {'chinese_name': '光学计算', 'description': '光学性质', 'physical_meaning': '光学性质', 'recommendation': '光学计算设.TRUE.', 'warnings':[]},
    'CSHIFT': {'chinese_name': '复位移', 'description': '复位移', 'physical_meaning': '展宽', 'recommendation': '0.1', 'warnings':[]},
    'CLL': {'chinese_name': 'CL规范', 'description': '规范', 'physical_meaning': '规范', 'recommendation': '0', 'warnings':[]},
    'ICORELEVEL': {'chinese_name': '芯能级', 'description': '芯能级', 'physical_meaning': '处理', 'recommendation': '芯空穴设1', 'warnings':[]},
    'ENCUTGW': {'chinese_name': 'GW截断能', 'description': '截断', 'physical_meaning': '截断', 'recommendation': '同ENCUT', 'warnings':[]},
    'NOMEGA': {'chinese_name': '频率点数', 'description': '频率点', 'physical_meaning': '点数', 'recommendation': '50', 'warnings':[]},
    'OMEGAMAX': {'chinese_name': '最大频率', 'description': '最大频率', 'physical_meaning': '最大', 'recommendation': '自动', 'warnings':[]},
    'LWANNIER90': {'chinese_name': 'Wannier90', 'description': '接口', 'physical_meaning': '接口', 'recommendation': '需要开启', 'warnings':[]},
    'LWANNIER90_RUN': {'chinese_name': 'Wannier运行', 'description': '运行', 'physical_meaning': '运行', 'recommendation': '.TRUE.', 'warnings':[]},
    'LNONCOLLINEAR': {'chinese_name': '非共线磁性', 'description': '非共线', 'physical_meaning': '结构', 'recommendation': '复杂磁性设.TRUE.', 'warnings':[]},
    'SAXIS': {'chinese_name': '自旋轴', 'description': '方向', 'physical_meaning': '方向', 'recommendation': '0 0 1', 'warnings':[]},
    'ICHIBARE': {'chinese_name': '手征密度', 'description': '密度', 'physical_meaning': '密度', 'recommendation': '1', 'warnings':[]},
    'IDIPOL': {'chinese_name': '偶极校正', 'description': '校正', 'physical_meaning': '校正', 'recommendation': '不对称表面设3', 'warnings': ['大真空面必须开启']},
    'LORBMOM': {'chinese_name': '轨道矩', 'description': '磁矩', 'physical_meaning': '输出', 'recommendation': 'SOC计算1', 'warnings':[]},
    'NUPDOWN': {'chinese_name': '自旋差', 'description': '自旋差', 'physical_meaning': '固定', 'recommendation': '固定总磁矩用', 'warnings':[]},
    'LCALCPOL': {'chinese_name': '极化输出', 'description': '输出', 'physical_meaning': '极化', 'recommendation': '铁电开启', 'warnings':[]},
    'LBERRY': {'chinese_name': 'Berry相', 'description': '相计算', 'physical_meaning': '相', 'recommendation': '极化开启', 'warnings':[]},
    'I_CONSTRAINED_M': {'chinese_name': '磁矩约束', 'description': '约束', 'physical_meaning': '约束', 'recommendation': '1', 'warnings':[]},
    'CONSTRAINED_M': {'chinese_name': '约束强度', 'description': '强度', 'physical_meaning': '强度', 'recommendation': '10', 'warnings':[]},
    'LAMBDA': {'chinese_name': '拉格朗日', 'description': '乘子', 'physical_meaning': '乘子', 'recommendation': '默认', 'warnings':[]},
    'AGGAX': {'chinese_name': 'GGA交换', 'description': '比例', 'physical_meaning': '比例', 'recommendation': '1.0', 'warnings':[]},
    'PHON_NSTRUCT': {'chinese_name': '声子结构数', 'description': '结构数', 'physical_meaning': '结构数', 'recommendation': '-1', 'warnings':[]},
    'IMIX': {'chinese_name': '混合方式', 'description': '方式', 'physical_meaning': '方式', 'recommendation': '金属4', 'warnings':[]},
    'NELMDL': {'chinese_name': '延迟开始', 'description': '延迟', 'physical_meaning': '延迟', 'recommendation': '难收敛-5', 'warnings':[]},
    'EFIELD': {'chinese_name': '电场', 'description': '外电场', 'physical_meaning': '电场', 'recommendation': '默认', 'warnings':[]},
    'EFIELD_PEAD': {'chinese_name': 'PEAD场', 'description': '场', 'physical_meaning': '场', 'recommendation': '默认', 'warnings':[]},
    'FERWE': {'chinese_name': 'Fermi面权重', 'description': '权重', 'physical_meaning': '权重', 'recommendation': '默认', 'warnings':[]},
    'MAXMEM': {'chinese_name': '最大内存', 'description': '内存', 'physical_meaning': '内存', 'recommendation': '按需', 'warnings':[]},
    'NSIM': {'chinese_name': '同时迭代', 'description': '迭代', 'physical_meaning': '迭代', 'recommendation': '4', 'warnings':[]},
    'LASYNC': {'chinese_name': '异步IO', 'description': 'IO', 'physical_meaning': 'IO', 'recommendation': '大超胞开启', 'warnings':[]},
    'GGA_COMPAT': {'chinese_name': 'GGA兼容', 'description': '兼容', 'physical_meaning': '兼容', 'recommendation': '.TRUE.', 'warnings':[]},
    'PRECFOCK': {'chinese_name': 'FOCK精度', 'description': '精度', 'physical_meaning': '精度', 'recommendation': 'Accurate', 'warnings':[]},
    'ENCUTLF': {'chinese_name': 'LF截断', 'description': '截断', 'physical_meaning': '截断', 'recommendation': '默认', 'warnings':[]},
    'DARWINR': {'chinese_name': 'Darwin标量', 'description': '标量', 'physical_meaning': '标量', 'recommendation': '默认', 'warnings':[]},
    'DARWINV': {'chinese_name': 'Darwin矢量', 'description': '矢量', 'physical_meaning': '矢量', 'recommendation': 'SOC开启', 'warnings':[]},
    'LSOL': {'chinese_name': '溶剂化', 'description': '溶剂', 'physical_meaning': '模型', 'recommendation': '溶液反应', 'warnings':['需编译VASPsol']},
    'LADDER': {'chinese_name': '能带输出', 'description': '能带', 'physical_meaning': '能带', 'recommendation': '.FALSE.', 'warnings':[]},
    'LAECHG': {'chinese_name': '全电荷密度', 'description': '全电荷', 'physical_meaning': '电荷', 'recommendation': '.FALSE.', 'warnings':[]},
    'LPARD': {'chinese_name': '投影态密度', 'description': '投影DOS', 'physical_meaning': 'PDOS', 'recommendation': '.FALSE.', 'warnings':[]},
    'NBMOD': {'chinese_name': '能带模式', 'description': '模式', 'physical_meaning': '模式', 'recommendation': '-1', 'warnings':[]},
    'IBAND': {'chinese_name': '能带索引', 'description': '索引', 'physical_meaning': '索引', 'recommendation': '默认', 'warnings':[]},
    'EINT': {'chinese_name': '能量范围', 'description': '积分', 'physical_meaning': '积分', 'recommendation': '默认', 'warnings':[]},
    'DIPOL': {'chinese_name': '偶极中心', 'description': '中心', 'physical_meaning': '中心', 'recommendation': '0.5 0.5 0.5', 'warnings':[]},
    'AMIN': {'chinese_name': '最小混合', 'description': '混合', 'physical_meaning': '混合', 'recommendation': '0.01', 'warnings':[]},
    'LMODELHF': {'chinese_name': '模型HF', 'description': '模型', 'physical_meaning': '模型', 'recommendation': '默认', 'warnings':[]},
    'HFLMAX': {'chinese_name': 'HF最大l', 'description': 'l', 'physical_meaning': 'l', 'recommendation': '-1', 'warnings':[]},
    'HFRCUT': {'chinese_name': 'HF截断', 'description': '截断', 'physical_meaning': '截断', 'recommendation': '默认', 'warnings':[]},
    'LRHFCALC': {'chinese_name': '相对论HF', 'description': 'HF', 'physical_meaning': 'HF', 'recommendation': '默认', 'warnings':[]},
    'LHFONE': {'chinese_name': '单中心HF', 'description': '单中心', 'physical_meaning': '中心', 'recommendation': '默认', 'warnings':[]},
    'HFSCREENC': {'chinese_name': '屏蔽类型', 'description': '类型', 'physical_meaning': '类型', 'recommendation': '默认', 'warnings':[]},
    'CMBJ': {'chinese_name': 'MBJ势', 'description': '势', 'physical_meaning': '势', 'recommendation': '校正带隙', 'warnings':[]},
    'CMBJA': {'chinese_name': 'MBJ参数A', 'description': 'A', 'physical_meaning': 'A', 'recommendation': '0.0', 'warnings':[]},
    'CMBJB': {'chinese_name': 'MBJ参数B', 'description': 'B', 'physical_meaning': 'B', 'recommendation': '1.0', 'warnings':[]},
    'LNICSALL': {'chinese_name': 'NMR位移', 'description': '位移', 'physical_meaning': '位移', 'recommendation': 'NMR计算', 'warnings':[]},
    'LCHIMAG': {'chinese_name': '化学位移', 'description': '化学', 'physical_meaning': '化学', 'recommendation': 'NMR计算', 'warnings':[]},
    'LDOWNSAMPLE': {'chinese_name': '降采样', 'description': '采样', 'physical_meaning': '降采', 'recommendation': '默认', 'warnings':[]},
    'ANDERSEN_PROB': {'chinese_name': 'Andersen概率', 'description': '概率', 'physical_meaning': '概率', 'recommendation': '0', 'warnings':[]},
    'HILLS_BIN': {'chinese_name': 'Hills采样', 'description': '采样', 'physical_meaning': '采样', 'recommendation': '-1', 'warnings':[]},
    'HILLS_H': {'chinese_name': 'Hills高度', 'description': '高度', 'physical_meaning': '高度', 'recommendation': '0.01', 'warnings':[]},
    'HILLS_W': {'chinese_name': 'Hills宽度', 'description': '宽度', 'physical_meaning': '宽度', 'recommendation': '0.05', 'warnings':[]},
    'APACO': {'chinese_name': '层间距', 'description': '间距', 'physical_meaning': '间距', 'recommendation': '默认', 'warnings':[]},
    'NPACO': {'chinese_name': 'PACO点数', 'description': '点数', 'physical_meaning': '点数', 'recommendation': '256', 'warnings':[]},
    'TIME': {'chinese_name': '时间参数', 'description': '时间', 'physical_meaning': '时间', 'recommendation': '自动', 'warnings':[]},
    'STEP_MAX': {'chinese_name': '最大步长', 'description': '步长', 'physical_meaning': '步长', 'recommendation': '自动', 'warnings':[]},
    'STEP_SIZE': {'chinese_name': '步长', 'description': '步', 'physical_meaning': '步', 'recommendation': '自动', 'warnings':[]},
    'MINROT': {'chinese_name': '最小旋转', 'description': '旋转', 'physical_meaning': '旋转', 'recommendation': '0.0', 'warnings':[]},
    'MIXFIRST': {'chinese_name': '先混合', 'description': '先混', 'physical_meaning': '先混', 'recommendation': '难收敛开启', 'warnings':[]},
    'ANORTH': {'chinese_name': '非正交盒', 'description': '非正交', 'physical_meaning': '非正交', 'recommendation': '0.0', 'warnings':[]},
    'LATTICE_CONSTRAINTS': {'chinese_name': '晶格约束', 'description': '约束', 'physical_meaning': '约束', 'recommendation': '默认', 'warnings':[]},
    'QSPIRAL': {'chinese_name': '螺旋q矢量', 'description': '螺旋', 'physical_meaning': '螺旋', 'recommendation': '0', 'warnings':[]},
    'LANGEVIN_GAMMA_L': {'chinese_name': '晶格阻尼', 'description': '阻尼', 'physical_meaning': '阻尼', 'recommendation': '1.0', 'warnings':[]},
    'SCSRAD': {'chinese_name': 'SCS半径', 'description': '半径', 'physical_meaning': '半径', 'recommendation': '0.0', 'warnings':[]},
    'TSUBSYS': {'chinese_name': '热浴', 'description': '子系', 'physical_meaning': '子系', 'recommendation': '1 1', 'warnings':[]},
    'VCUTOFF': {'chinese_name': '截断速度', 'description': '截断', 'physical_meaning': '截断', 'recommendation': '0.0', 'warnings':[]},
    'OFIELD_A': {'chinese_name': '有序场A', 'description': 'A', 'physical_meaning': 'A', 'recommendation': '0.0', 'warnings':[]},
    'OFIELD_KAPPA': {'chinese_name': '有序场kappa', 'description': 'kappa', 'physical_meaning': 'kappa', 'recommendation': '0.0', 'warnings':[]},
    'OFIELD_Q6_FAR': {'chinese_name': 'Q6远场', 'description': 'Q6', 'physical_meaning': 'Q6', 'recommendation': '0.0', 'warnings':[]},
    'OFIELD_Q6_NEAR': {'chinese_name': 'Q6近场', 'description': '近场', 'physical_meaning': '近', 'recommendation': '0.0', 'warnings':[]},
    'LEFG': {'chinese_name': 'EFG', 'description': 'EFG', 'physical_meaning': 'EFG', 'recommendation': 'NMR计算', 'warnings':[]},
    'QUAD_EFG': {'chinese_name': '四极矩', 'description': '四极', 'physical_meaning': '四极', 'recommendation': '默认', 'warnings':[]},
    'RANDOM_SEED': {'chinese_name': '随机种子', 'description': '种子', 'physical_meaning': '种子', 'recommendation': '自动', 'warnings':[]},
    'PARAM1': {'chinese_name': '参数1', 'description': '1', 'physical_meaning': '1', 'recommendation': '0.0', 'warnings':[]},
    'PARAM2': {'chinese_name': '参数2', 'description': '2', 'physical_meaning': '2', 'recommendation': '0.0', 'warnings':[]},
    'LGAUGE': {'chinese_name': '规范固定', 'description': '规范', 'physical_meaning': '规范', 'recommendation': '默认', 'warnings':[]},
    'LRPAFORCE': {'chinese_name': 'RPA力', 'description': 'RPA', 'physical_meaning': 'RPA', 'recommendation': 'RPA计算', 'warnings':[]},
    'LFXC': {'chinese_name': 'FXC', 'description': 'FXC', 'physical_meaning': 'FXC', 'recommendation': '默认', 'warnings':[]},
    'LTCTE': {'chinese_name': 'TCTE', 'description': 'TCTE', 'physical_meaning': 'TCTE', 'recommendation': '默认', 'warnings':[]},
    'LTETE': {'chinese_name': 'TETE', 'description': 'TETE', 'physical_meaning': 'TETE', 'recommendation': '默认', 'warnings':[]},
    'LTRIPLET': {'chinese_name': '三态', 'description': '三态', 'physical_meaning': '三态', 'recommendation': '激发态', 'warnings':[]},
    'LUSEW': {'chinese_name': 'USEW', 'description': 'USEW', 'physical_meaning': 'USEW', 'recommendation': '默认', 'warnings':[]},
    'NUCIND': {'chinese_name': '核独立', 'description': '核', 'physical_meaning': '核', 'recommendation': '默认', 'warnings':[]},
    'NTAUPAR': {'chinese_name': '时间并行', 'description': '并行', 'physical_meaning': '并行', 'recommendation': '1', 'warnings':[]},
    'NTARGET_STATES': {'chinese_name': '目标态', 'description': '目标', 'physical_meaning': '目标', 'recommendation': '0', 'warnings':[]},
    'LOCPROJ': {'chinese_name': '局域投影', 'description': '投影', 'physical_meaning': '投影', 'recommendation': '0', 'warnings':[]},
    'POMASS': {'chinese_name': '离子质量', 'description': '质量', 'physical_meaning': '质量', 'recommendation': '自动', 'warnings':[]},
    'PROUTINE': {'chinese_name': '打印程序', 'description': '打印', 'physical_meaning': '打印', 'recommendation': '0', 'warnings':[]},
    'PTHRESHOLD': {'chinese_name': '打印阈值', 'description': '阈值', 'physical_meaning': '阈值', 'recommendation': '1E-4', 'warnings':[]},
    'LMUSIC': {'chinese_name': 'MUSIC', 'description': '接口', 'physical_meaning': '接口', 'recommendation': '默认', 'warnings':[]},
    'SHIFTRED': {'chinese_name': '偏移缩减', 'description': '缩减', 'physical_meaning': '缩减', 'recommendation': '默认', 'warnings':[]},
    'NKREDX': {'chinese_name': 'X向K缩减', 'description': 'X缩减', 'physical_meaning': '缩减', 'recommendation': '1', 'warnings':[]},
    'NKREDY': {'chinese_name': 'Y向K缩减', 'description': 'Y缩减', 'physical_meaning': '缩减', 'recommendation': '1', 'warnings':[]},
    'NKREDZ': {'chinese_name': 'Z向K缩减', 'description': 'Z缩减', 'physical_meaning': '缩减', 'recommendation': '1', 'warnings':[]},
    'KPOINT_BSE': {'chinese_name': 'BSE k点', 'description': 'BSE', 'physical_meaning': 'BSE', 'recommendation': '0', 'warnings':[]},
    'KPUSE': {'chinese_name': '使用k点', 'description': '使用k', 'physical_meaning': 'k', 'recommendation': '0', 'warnings':[]},
    'EVENONLY': {'chinese_name': '偶k点', 'description': '偶数', 'physical_meaning': '偶数', 'recommendation': '.FALSE.', 'warnings':[]},
    'EVENONLYGW': {'chinese_name': 'GW偶k点', 'description': '偶', 'physical_meaning': '偶', 'recommendation': '.FALSE.', 'warnings':[]},
    'ODDONLY': {'chinese_name': '奇k点', 'description': '奇数', 'physical_meaning': '奇数', 'recommendation': '.FALSE.', 'warnings':[]},
    'ODDONLYGW': {'chinese_name': 'GW奇k点', 'description': '奇', 'physical_meaning': '奇', 'recommendation': '.FALSE.', 'warnings':[]},
    'NBANDSGW': {'chinese_name': 'GW能带数', 'description': '数量', 'physical_meaning': '数量', 'recommendation': '自动', 'warnings':[]},
    'NBANDSO': {'chinese_name': '占据能带数', 'description': '占据', 'physical_meaning': '占据', 'recommendation': '自动', 'warnings':[]},
    'NBANDSV': {'chinese_name': '虚能带数', 'description': '虚', 'physical_meaning': '虚', 'recommendation': '自动', 'warnings':[]},
    'NOMEGAPAR': {'chinese_name': '频率并行', 'description': '频率', 'physical_meaning': '频率', 'recommendation': '1', 'warnings':[]},
    'NOMEGAR': {'chinese_name': '实频率点', 'description': '采样', 'physical_meaning': '采样', 'recommendation': '0', 'warnings':[]},
    'OMEGAMIN': {'chinese_name': '最小频率', 'description': '最小', 'physical_meaning': '最小', 'recommendation': '-1.0', 'warnings':[]},
    'OMEGATL': {'chinese_name': '频率尾参数', 'description': '尾巴', 'physical_meaning': '尾巴', 'recommendation': '0.0', 'warnings':[]},
    'SELFENERGY': {'chinese_name': '自能计算', 'description': 'GW', 'physical_meaning': 'GW', 'recommendation': 'GW计算开启', 'warnings':[]},
    'LFERMIGW': {'chinese_name': 'Fermi更新', 'description': '更新', 'physical_meaning': '更新', 'recommendation': 'GW迭代开启', 'warnings':[]},
    'LSINGLES': {'chinese_name': '单粒子', 'description': '近似', 'physical_meaning': '近似', 'recommendation': '默认', 'warnings':[]},
    'ALDA': {'chinese_name': 'ALDA校正', 'description': '校正', 'physical_meaning': '校正', 'recommendation': '默认', 'warnings':[]},
    'ENCUTGWSOFT': {'chinese_name': 'GW软截断', 'description': '软', 'physical_meaning': '软', 'recommendation': '自动', 'warnings':[]},
    'ENINI': {'chinese_name': '初始能量', 'description': '初始', 'physical_meaning': '初始', 'recommendation': '自动', 'warnings':[]},
    'PHON_LBOSE': {'chinese_name': '声子展宽', 'description': '展宽', 'physical_meaning': '展宽', 'recommendation': '默认', 'warnings':[]},
    'PHON_LMC': {'chinese_name': '声子MC', 'description': 'MC', 'physical_meaning': 'MC', 'recommendation': '默认', 'warnings':[]},
    'PHON_NTLIST': {'chinese_name': '声子点', 'description': '点', 'physical_meaning': '点', 'recommendation': '0', 'warnings':[]},
    'PHON_TLIST': {'chinese_name': '声子温度', 'description': '温度', 'physical_meaning': '温度', 'recommendation': '0.0', 'warnings':[]},
    'WANPROJ': {'chinese_name': 'Wannier投影', 'description': '投影', 'physical_meaning': '投影', 'recommendation': 'Wannier计算开启', 'warnings':[]},
    'LWRITE_MMN_AMN': {'chinese_name': '写MMN/AMN', 'description': '重叠', 'physical_meaning': '重叠', 'recommendation': '默认', 'warnings':[]},
    'LWRITE_UNK': {'chinese_name': '写UNK', 'description': '写波', 'physical_meaning': '写波', 'recommendation': '默认', 'warnings':[]},
    'LWRITE_WANPROJ': {'chinese_name': '写投影', 'description': '写投', 'physical_meaning': '写投', 'recommendation': '默认', 'warnings':[]},
    'CH_LSPEC': {'chinese_name': '芯空穴谱', 'description': '谱', 'physical_meaning': '谱', 'recommendation': '芯空穴计算', 'warnings':[]},
    'CH_NEDOS': {'chinese_name': '空穴DOS点', 'description': '点数', 'physical_meaning': '点数', 'recommendation': '0', 'warnings':[]},
    'CH_SIGMA': {'chinese_name': '空穴展宽', 'description': '展宽', 'physical_meaning': '展宽', 'recommendation': '0.1', 'warnings':[]},
    'CLN': {'chinese_name': 'CL规范', 'description': '规范', 'physical_meaning': '规范', 'recommendation': '0', 'warnings':[]},
    'CLNT': {'chinese_name': 'CL类型', 'description': '类型', 'physical_meaning': '类型', 'recommendation': '0', 'warnings':[]},
    'CLZ': {'chinese_name': 'CL_Z', 'description': 'Z', 'physical_meaning': 'Z', 'recommendation': '0.0', 'warnings':[]},
    'IEPSILON': {'chinese_name': '介电索引', 'description': '索引', 'physical_meaning': '索引', 'recommendation': '1', 'warnings':[]},
    'IGPAR': {'chinese_name': '光学方向', 'description': '方向', 'physical_meaning': '方向', 'recommendation': '0', 'warnings':[]},
    'IPEAD': {'chinese_name': 'PEAD', 'description': 'PEAD', 'physical_meaning': 'PEAD', 'recommendation': '0', 'warnings':[]},
    'LORBITALREAL': {'chinese_name': '实空间轨道', 'description': '轨道', 'physical_meaning': '轨道', 'recommendation': '默认', 'warnings':[]},
    'NMAXFOCKAE': {'chinese_name': 'AE最大索引', 'description': 'AE', 'physical_meaning': 'AE', 'recommendation': '0', 'warnings':[]},
    'AGGAC': {'chinese_name': 'GGA相关', 'description': '相关', 'physical_meaning': '相关', 'recommendation': '0.0', 'warnings':[]},
    'ALDAC': {'chinese_name': 'LDA相关', 'description': '相关', 'physical_meaning': '相关', 'recommendation': '0.0', 'warnings':[]},
    'LMIXTAU': {'chinese_name': '自旋混合', 'description': '常数', 'physical_meaning': '常数', 'recommendation': '默认', 'warnings':[]},
    'LNABLA': {'chinese_name': '梯度输出', 'description': '梯度', 'physical_meaning': '梯度', 'recommendation': '默认', 'warnings':[]},
    'MAGPOS': {'chinese_name': '磁矩位置', 'description': '位置', 'physical_meaning': '位置', 'recommendation': '默认', 'warnings':[]},
    'ORBITALMAG': {'chinese_name': '轨道磁性', 'description': '轨道', 'physical_meaning': '轨道', 'recommendation': 'SOC计算开启', 'warnings':[]},
    'MAGDIPOLOUT': {'chinese_name': '磁偶极输出', 'description': '偶极', 'physical_meaning': '偶极', 'recommendation': '默认', 'warnings':[]},
    'ISPIND': {'chinese_name': '分立自旋', 'description': '分立', 'physical_meaning': '分立', 'recommendation': '1', 'warnings':[]},
    'ICALCEPS': {'chinese_name': '介电开关', 'description': '开关', 'physical_meaning': '开关', 'recommendation': '介电计算开启', 'warnings':[]},
    'FINDIFF': {'chinese_name': '有限差分', 'description': '差分', 'physical_meaning': '差分', 'recommendation': '0', 'warnings':[]},
    'DQ': {'chinese_name': '位移增量', 'description': '增量', 'physical_meaning': '增量', 'recommendation': '0.005', 'warnings':[]},
    'DEPER': {'chinese_name': '能量步长', 'description': '步长', 'physical_meaning': '步长', 'recommendation': '0.0', 'warnings':[]},
    'DIMER_DIST': {'chinese_name': '二聚体距离', 'description': '距离', 'physical_meaning': '距离', 'recommendation': '0.01', 'warnings':[]},
    'IWAVPR': {'chinese_name': '波函数处理', 'description': '处理', 'physical_meaning': '处理', 'recommendation': '1', 'warnings':[]},
    'LCOMPAT': {'chinese_name': '兼容性', 'description': '兼容', 'physical_meaning': '兼容', 'recommendation': '默认', 'warnings':[]},
    'LCORR': {'chinese_name': '电荷校正', 'description': '平均', 'physical_meaning': '平均', 'recommendation': '默认', 'warnings':[]},
    'LDIAG': {'chinese_name': '对角化', 'description': '对角', 'physical_meaning': '对角', 'recommendation': '默认', 'warnings':[]},
    'LDIPOL': {'chinese_name': '偶极校正', 'description': '极性面校正', 'physical_meaning': '偶极矩消除', 'recommendation': '配合IDIPOL使用', 'warnings':[]},
    'LLRAUG': {'chinese_name': 'LR_AUG', 'description': '平滑', 'physical_meaning': '平滑', 'recommendation': '默认', 'warnings':[]},
    'LSYMGRAD': {'chinese_name': '对称梯度', 'description': '加速', 'physical_meaning': '加速', 'recommendation': '默认', 'warnings':[]},
    'VALUE_MAX': {'chinese_name': '最大值', 'description': '约束', 'physical_meaning': '约束', 'recommendation': '0.0', 'warnings':[]},
    'VALUE_MIN': {'chinese_name': '最小值', 'description': '约束', 'physical_meaning': '约束', 'recommendation': '0.0', 'warnings':[]},
    'ENMAX': {'chinese_name': '最大ENMAX', 'description': 'POTCAR', 'physical_meaning': 'POTCAR', 'recommendation': '自动', 'warnings':[]},
    'ENMIN': {'chinese_name': '最小ENMIN', 'description': 'POTCAR', 'physical_meaning': 'POTCAR', 'recommendation': '自动', 'warnings':[]},
    'ENAVG': {'chinese_name': '平均截断能', 'description': 'POTCAR', 'physical_meaning': 'POTCAR', 'recommendation': '自动', 'warnings':[]},
    'PFLAT': {'chinese_name': 'PFLAT', 'description': 'PFLAT', 'physical_meaning': 'PFLAT', 'recommendation': '默认', 'warnings':[]},
    'PSUBSYS': {'chinese_name': '参数子系统', 'description': '子系', 'physical_meaning': '子系', 'recommendation': '1', 'warnings':[]},
    'QMAXFOCKAE': {'chinese_name': 'QMAX_AE', 'description': 'QMAX', 'physical_meaning': 'QMAX', 'recommendation': '0.0', 'warnings':[]},
    'ZVAL': {'chinese_name': 'ZVAL', 'description': '价电子', 'physical_meaning': '价电', 'recommendation': '自动', 'warnings':[]},
    'NBLK': {'chinese_name': '输出块大小', 'description': '块', 'physical_meaning': '块', 'recommendation': '-1', 'warnings':[]},
    'NCRPA_BANDS': {'chinese_name': 'CRPA能带', 'description': 'CRPA', 'physical_meaning': 'CRPA', 'recommendation': '0', 'warnings':[]},
    'NPPSTR': {'chinese_name': '投影方向', 'description': '方向', 'physical_meaning': '方向', 'recommendation': '0', 'warnings':[]},
    'NBSEEIG': {'chinese_name': 'BSE本征值', 'description': '本征', 'physical_meaning': '本征', 'recommendation': '0', 'warnings':[]},
    'PLEVEL': {'chinese_name': '打印级别', 'description': '级别', 'physical_meaning': '级别', 'recommendation': '0', 'warnings':[]},
    'INIMIX': {'chinese_name': '初始混合', 'description': '初始', 'physical_meaning': '初始', 'recommendation': '1', 'warnings':[]},
    'MIXPRE': {'chinese_name': '混合预处理', 'description': '预处理', 'physical_meaning': '预处理', 'recommendation': '0', 'warnings':[]},
    'NFREE': {'chinese_name': '有限差分步数', 'description': '步数', 'physical_meaning': '步数', 'recommendation': '0', 'warnings':[]},
    'NDAV': {'chinese_name': 'Davidson迭代', 'description': '迭代', 'physical_meaning': '迭代', 'recommendation': '30', 'warnings':[]},
    'INCREM': {'chinese_name': '增量参数', 'description': '增量', 'physical_meaning': '增量', 'recommendation': '0.015', 'warnings':[]},
    'ANTIRES': {'chinese_name': '反共振计算', 'description': '开关', 'physical_meaning': '开关', 'recommendation': '0', 'warnings':[]},
    'HITOLER': {'chinese_name': '高精度容差', 'description': '容差', 'physical_meaning': '容差', 'recommendation': '1E-5', 'warnings':[]},
    'SHAKEMAXITER': {'chinese_name': 'Shake迭代', 'description': '迭代', 'physical_meaning': '迭代', 'recommendation': '50', 'warnings':[]},
    'SHAKETOL': {'chinese_name': 'Shake容差', 'description': '容差', 'physical_meaning': '容差', 'recommendation': '1E-5', 'warnings':[]},
    'EPSILON': {'chinese_name': '介电常数', 'description': '背景', 'physical_meaning': '背景', 'recommendation': '1.0', 'warnings':[]},
    'SMEARINGS': {'chinese_name': 'Smearing列表', 'description': '列表', 'physical_meaning': '列表', 'recommendation': '0.0', 'warnings':[]},
    'LGauss': {'chinese_name': '高斯展宽', 'description': '开关', 'physical_meaning': '开关', 'recommendation': '默认', 'warnings':[]},
    'LVDWEXPANSION': {'chinese_name': 'vdW展开', 'description': '展开', 'physical_meaning': '展开', 'recommendation': '默认', 'warnings':[]},
    'LVDW_EWALD': {'chinese_name': 'vdW Ewald', 'description': '求和', 'physical_meaning': '求和', 'recommendation': '默认', 'warnings':[]},
    'VDW_C6': {'chinese_name': 'C6系数', 'description': '原子C6', 'physical_meaning': '原子C6', 'recommendation': '自动', 'warnings':[]},
    'VDW_R0': {'chinese_name': 'R0半径', 'description': '半径', 'physical_meaning': '半径', 'recommendation': '自动', 'warnings':[]},
    'VDW_CNRADIUS': {'chinese_name': '截断半径', 'description': '距离', 'physical_meaning': '距离', 'recommendation': '0.0', 'warnings':[]},
    'VDW_D': {'chinese_name': 'D参数', 'description': 'damping', 'physical_meaning': 'damping', 'recommendation': '0.0', 'warnings':[]},
    'VDW_S8': {'chinese_name': 'S8参数', 'description': '缩放', 'physical_meaning': '缩放', 'recommendation': '0.0', 'warnings':[]},
    'ZAB_VDW': {'chinese_name': 'vdW半径', 'description': 'Hutson', 'physical_meaning': 'Hutson', 'recommendation': '0.0', 'warnings':[]},
    'TAU': {'chinese_name': '温度耦合', 'description': '常数', 'physical_meaning': '常数', 'recommendation': '自动', 'warnings':[]},
    'LTEEPS': {'chinese_name': 'EEPS', 'description': '总能', 'physical_meaning': '总能', 'recommendation': '默认', 'warnings':[]},
    'LTHOMAS': {'chinese_name': 'Thomas', 'description': '屏蔽', 'physical_meaning': '屏蔽', 'recommendation': '默认', 'warnings':[]},
    'LFXCEPS': {'chinese_name': 'FXC_EPS', 'description': '介电', 'physical_meaning': '介电', 'recommendation': '默认', 'warnings':[]},
    'LFXHEG': {'chinese_name': 'FXC_HEG', 'description': '均匀电子', 'physical_meaning': '均匀电子', 'recommendation': '默认', 'warnings':[]},
    'LMAGBLOCH': {'chinese_name': '磁性Bloch变换', 'description': '开关', 'physical_meaning': '开关', 'recommendation': '默认', 'warnings':[]},
    'LBLUEOUT': {'chinese_name': 'Bloch校正输出', 'description': '输出', 'physical_meaning': '输出', 'recommendation': '默认', 'warnings':[]},
    'LBONE': {'chinese_name': 'BondOrder输出', 'description': '输出', 'physical_meaning': '输出', 'recommendation': '默认', 'warnings':[]},
    'LCALCEPS': {'chinese_name': '介电常数输出', 'description': '输出', 'physical_meaning': '输出', 'recommendation': '介电计算开启', 'warnings':[]},
    'LHARTREE': {'chinese_name': 'Hartree势输出', 'description': '输出', 'physical_meaning': '输出', 'recommendation': '默认', 'warnings':[]},
    'LHYPERFINE': {'chinese_name': '超精细输出', 'description': '精细', 'physical_meaning': '精细', 'recommendation': '默认', 'warnings':[]},
    'LPEAD': {'chinese_name': 'PEAD输出', 'description': '分析', 'physical_meaning': '分析', 'recommendation': '默认', 'warnings':[]},
    'LPLANE': {'chinese_name': '平面波输出', 'description': '系数', 'physical_meaning': '系数', 'recommendation': '默认', 'warnings':[]},
    'LRPA': {'chinese_name': 'RPA输出', 'description': '相关', 'physical_meaning': '相关', 'recommendation': 'RPA开启', 'warnings':[]},
    'LSCAAWARE': {'chinese_name': 'SCA启用', 'description': '电荷', 'physical_meaning': '电荷', 'recommendation': '默认', 'warnings':[]},
    'LSCALU': {'chinese_name': 'LU分解输出', 'description': '分解', 'physical_meaning': '分解', 'recommendation': '默认', 'warnings':[]},
    'LSCSGRAD': {'chinese_name': 'SCS梯度输出', 'description': '梯度', 'physical_meaning': '梯度', 'recommendation': '默认', 'warnings':[]},
    'LSELFENERGY': {'chinese_name': '自能输出', 'description': '计算', 'physical_meaning': '计算', 'recommendation': '默认', 'warnings':[]},
    'LSEPB': {'chinese_name': '分离带输出', 'description': '能带', 'physical_meaning': '能带', 'recommendation': '默认', 'warnings':[]},
    'LSEPK': {'chinese_name': '分离k点输出', 'description': 'k点', 'physical_meaning': 'k点', 'recommendation': '默认', 'warnings':[]},
    'LSPECTRAL': {'chinese_name': '谱函数输出', 'description': '开关', 'physical_meaning': '开关', 'recommendation': '默认', 'warnings':[]},
    'LSPECTRALGW': {'chinese_name': 'GW谱函数', 'description': '开关', 'physical_meaning': '开关', 'recommendation': '默认', 'warnings':[]},
    'LSPIRAL': {'chinese_name': '螺旋输出', 'description': '结构', 'physical_meaning': '结构', 'recommendation': '默认', 'warnings':[]},
    'LSUBROT': {'chinese_name': '子旋转输出', 'description': '旋转', 'physical_meaning': '旋转', 'recommendation': '默认', 'warnings':[]},
    'LZEROZ': {'chinese_name': 'Z方向零点', 'description': '能量', 'physical_meaning': '能量', 'recommendation': '默认', 'warnings':[]},
    'ISEARCH': {'chinese_name': '原子位置搜索', 'description': '搜索算法', 'physical_meaning': '搜索', 'recommendation': '0', 'warnings':[]},
    'LFOCKAEDFT': {'chinese_name': 'HSE精确交换', 'description': '计算精确交换', 'physical_meaning': '交换', 'recommendation': '默认', 'warnings':[]},
    'LKPROJ': {'chinese_name': 'Wannier投影', 'description': '基组', 'physical_meaning': '基组', 'recommendation': '默认', 'warnings':[]},
    'LMAXFOCKAE': {'chinese_name': 'Fock算符最大L', 'description': '角动量', 'physical_meaning': '角动量', 'recommendation': '0', 'warnings':[]},
    'LMAXPAW': {'chinese_name': 'PAW投影最大L', 'description': '角动量', 'physical_meaning': '角动量', 'recommendation': '-1', 'warnings':[]},
    'LMAXTAU': {'chinese_name': '张力计算最大L', 'description': '角动量', 'physical_meaning': '角动量', 'recommendation': '-1', 'warnings':[]},
    'LMETAGGA': {'chinese_name': 'meta-GGA计算', 'description': '启用', 'physical_meaning': '启用', 'recommendation': '默认', 'warnings':[]},
    'LMONO': {'chinese_name': '单极矩计算', 'description': '计算静电', 'physical_meaning': '静电', 'recommendation': '默认', 'warnings':[]},
    'LNMR_SYM_RED': {'chinese_name': 'NMR对称性约化', 'description': '约化', 'physical_meaning': '约化', 'recommendation': '.TRUE.', 'warnings':[]},
    'LVEL': {'chinese_name': '速度计算', 'description': '原子速度', 'physical_meaning': '速度', 'recommendation': '默认', 'warnings':[]},
    'ML_MODE': {'chinese_name': 'ML训练模式', 'description': '模式', 'physical_meaning': '模式', 'recommendation': '0', 'warnings':[]},
    'ML_FF_LMLFF': {'chinese_name': '机器学习力场', 'description': '启用机器学习力场', 'physical_meaning': '开关', 'recommendation': '需要时开启', 'warnings':[]},
    'ML_FF_LMLMB': {'chinese_name': 'ML多体势能面', 'description': '训练', 'physical_meaning': '训练', 'recommendation': '默认', 'warnings':[]},
    'ML_FF_ISTART': {'chinese_name': 'ML初始化模式', 'description': '控制', 'physical_meaning': '控制', 'recommendation': '0=从头; 1=预测; 2=继续', 'warnings':[]},
    'ML_FF_MCONF': {'chinese_name': 'ML训练构型数', 'description': '数量', 'physical_meaning': '集大小', 'recommendation': '1000', 'warnings':[]},
    'ML_FF_MCONF_NEW': {'chinese_name': 'ML新构型数', 'description': '数量', 'physical_meaning': '增量', 'recommendation': '50', 'warnings':[]},
    'ML_FF_MHIS': {'chinese_name': 'ML历史步数', 'description': '步数', 'physical_meaning': '步数', 'recommendation': '10', 'warnings':[]},
    'ML_FF_LCONF_DISCARD': {'chinese_name': 'ML丢弃低置信度', 'description': '控制', 'physical_meaning': '控制', 'recommendation': '默认', 'warnings':[]},
    'ML_FF_LBASIS_DISCARD': {'chinese_name': 'ML丢弃基组', 'description': '基组', 'physical_meaning': '基组', 'recommendation': '默认', 'warnings':[]},
    'ML_FF_LCRITERIA': {'chinese_name': 'ML使用学习标准', 'description': '标准', 'physical_meaning': '标准', 'recommendation': '默认', 'warnings':[]},
    'ML_FF_LEATOM_MB': {'chinese_name': 'ML使用原子能量', 'description': '参考', 'physical_meaning': '参考', 'recommendation': '默认', 'warnings':[]},
    'ML_FF_LHEAT_MB': {'chinese_name': 'ML计算热流', 'description': 'MD热', 'physical_meaning': '热', 'recommendation': '默认', 'warnings':[]},
    'ML_FF_CSIG': {'chinese_name': 'ML信号噪声比', 'description': '阈值', 'physical_meaning': '阈值', 'recommendation': '3.0', 'warnings':[]},
    'ML_FF_CSLOPE': {'chinese_name': 'ML斜率缩放', 'description': '因子', 'physical_meaning': '因子', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_CTIFOR': {'chinese_name': 'ML离子力置信', 'description': '阈值', 'physical_meaning': '阈值', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_WTIFOR': {'chinese_name': 'ML离子力权重', 'description': '权重', 'physical_meaning': '权重', 'recommendation': '1.0', 'warnings':[]},
    'ML_FF_WTOTEN': {'chinese_name': 'ML能量权重', 'description': '权重', 'physical_meaning': '权重', 'recommendation': '0.01', 'warnings':[]},
    'ML_FF_WTSIF': {'chinese_name': 'ML应力权重', 'description': '权重', 'physical_meaning': '权重', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_NWRITE': {'chinese_name': 'ML写入模式', 'description': '控制', 'physical_meaning': '控制', 'recommendation': '2', 'warnings':[]},
    'ML_FF_ISAMPLE': {'chinese_name': 'ML采样模式', 'description': '策略', 'physical_meaning': '策略', 'recommendation': '3', 'warnings':[]},
    'ML_FF_NDIM_SCALAPACK': {'chinese_name': 'ML维数', 'description': '优化', 'physical_meaning': '优化', 'recommendation': '-1', 'warnings':[]},
    'ML_FF_IERR': {'chinese_name': 'ML错误处理', 'description': '方式', 'physical_meaning': '方式', 'recommendation': '0', 'warnings':[]},
    'ML_FF_IWEIGHT': {'chinese_name': 'ML权重计算', 'description': '方式', 'physical_meaning': '方式', 'recommendation': '0', 'warnings':[]},
    'ML_FF_AFILT2_MB': {'chinese_name': 'ML二阶滤波', 'description': '宽度', 'physical_meaning': '宽度', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_LAFILT2_MB': {'chinese_name': 'ML启用二阶滤波', 'description': '滤波', 'physical_meaning': '滤波', 'recommendation': '默认', 'warnings':[]},
    'ML_FF_IAFILT2_MB': {'chinese_name': 'ML原子滤波指标', 'description': '指标', 'physical_meaning': '指标', 'recommendation': '0', 'warnings':[]},
    'ML_FF_LMAX2_MB': {'chinese_name': 'ML第二角动量', 'description': '阶数', 'physical_meaning': '阶数', 'recommendation': '0', 'warnings':[]},
    'ML_FF_LNORM1_MB': {'chinese_name': 'ML第一归一化', 'description': '方式', 'physical_meaning': '方式', 'recommendation': '默认', 'warnings':[]},
    'ML_FF_LNORM2_MB': {'chinese_name': 'ML第二归一化', 'description': '方式', 'physical_meaning': '方式', 'recommendation': '默认', 'warnings':[]},
    'ML_FF_NR1_MB': {'chinese_name': 'ML第一径向网格', 'description': '点数', 'physical_meaning': '点数', 'recommendation': '20', 'warnings':[]},
    'ML_FF_NR2_MB': {'chinese_name': 'ML第二径向网格', 'description': '点数', 'physical_meaning': '点数', 'recommendation': '20', 'warnings':[]},
    'ML_FF_NHYP1_MB': {'chinese_name': 'ML第一双曲势', 'description': '阶数', 'physical_meaning': '阶数', 'recommendation': '0', 'warnings':[]},
    'ML_FF_NHYP2_MB': {'chinese_name': 'ML第二双曲势', 'description': '阶数', 'physical_meaning': '阶数', 'recommendation': '0', 'warnings':[]},
    'ML_FF_MRB1_MB': {'chinese_name': 'ML第一径向基', 'description': '数量', 'physical_meaning': '数量', 'recommendation': '16', 'warnings':[]},
    'ML_FF_MRB2_MB': {'chinese_name': 'ML第二径向基', 'description': '数量', 'physical_meaning': '数量', 'recommendation': '16', 'warnings':[]},
    'ML_FF_MSPL1_MB': {'chinese_name': 'ML第一样条点', 'description': '点数', 'physical_meaning': '点数', 'recommendation': '1000', 'warnings':[]},
    'ML_FF_MSPL2_MB': {'chinese_name': 'ML第二样条点', 'description': '点数', 'physical_meaning': '点数', 'recommendation': '1000', 'warnings':[]},
    'ML_FF_SION1_MB': {'chinese_name': 'ML第一离子噪声', 'description': '噪声', 'physical_meaning': '噪声', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_SION2_MB': {'chinese_name': 'ML第二离子噪声', 'description': '噪声', 'physical_meaning': '噪声', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_IBROAD1_MB': {'chinese_name': 'ML第一广播索引', 'description': '索引', 'physical_meaning': '索引', 'recommendation': '0', 'warnings':[]},
    'ML_FF_IBROAD2_MB': {'chinese_name': 'ML第二广播索引', 'description': '索引', 'physical_meaning': '索引', 'recommendation': '0', 'warnings':[]},
    'ML_FF_ICUT1_MB': {'chinese_name': 'ML第一截断索引', 'description': '索引', 'physical_meaning': '索引', 'recommendation': '0', 'warnings':[]},
    'ML_FF_ICUT2_MB': {'chinese_name': 'ML第二截断索引', 'description': '索引', 'physical_meaning': '索引', 'recommendation': '0', 'warnings':[]},
    'ML_FF_RCUT1_MB': {'chinese_name': 'ML第一截断半径', 'description': '半径', 'physical_meaning': '半径', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_RCUT2_MB': {'chinese_name': 'ML第二截断半径', 'description': '半径', 'physical_meaning': '半径', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_ISOAP1_MB': {'chinese_name': 'ML第一SOAP', 'description': '类型', 'physical_meaning': '类型', 'recommendation': '0', 'warnings':[]},
    'ML_FF_ISOAP2_MB': {'chinese_name': 'ML第二SOAP', 'description': '类型', 'physical_meaning': '类型', 'recommendation': '0', 'warnings':[]},
    'ML_FF_W1_MB': {'chinese_name': 'ML权重因子1', 'description': '因子', 'physical_meaning': '因子', 'recommendation': '1.0', 'warnings':[]},
    'ML_FF_W2_MB': {'chinese_name': 'ML权重因子2', 'description': '因子', 'physical_meaning': '因子', 'recommendation': '1.0', 'warnings':[]},
    'ML_FF_MB_MB': {'chinese_name': 'ML多体矩阵', 'description': '配置', 'physical_meaning': '配置', 'recommendation': '1', 'warnings':[]},
    'ML_FF_EATOM': {'chinese_name': 'ML原子能量', 'description': '参考', 'physical_meaning': '参考', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_CDOUB': {'chinese_name': 'ML双层因子', 'description': '因子', 'physical_meaning': '因子', 'recommendation': '1.0', 'warnings':[]},
    'ML_FF_CSF': {'chinese_name': 'ML置信度缩放', 'description': '缩放', 'physical_meaning': '缩放', 'recommendation': '1.0', 'warnings':[]},
    'ML_FF_SIGV0_MB': {'chinese_name': 'ML势能噪声', 'description': '估计', 'physical_meaning': '估计', 'recommendation': '0.001', 'warnings':[]},
    'ML_FF_SIGW0_MB': {'chinese_name': 'ML力噪声', 'description': '估计', 'physical_meaning': '估计', 'recommendation': '0.001', 'warnings':[]},
    'ML_FF_ISCALE_TOTEN_MB': {'chinese_name': 'ML能量缩放', 'description': '因子', 'physical_meaning': '因子', 'recommendation': '0', 'warnings':[]},
    'ML_FF_ICOUPLE_MB': {'chinese_name': 'ML耦合索引', 'description': '索引', 'physical_meaning': '索引', 'recommendation': '0', 'warnings':[]},
    'ML_FF_LCOUPLE_MB': {'chinese_name': 'ML启用耦合', 'description': '耦合', 'physical_meaning': '耦合', 'recommendation': '默认', 'warnings':[]},
    'ML_FF_RCOUPLE_MB': {'chinese_name': 'ML耦合半径', 'description': '截断', 'physical_meaning': '截断', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_NATOM_COUPLED_MB': {'chinese_name': 'ML耦合原子数', 'description': '数量', 'physical_meaning': '数量', 'recommendation': '0', 'warnings':[]},
    'ML_FF_IREG_MB': {'chinese_name': 'ML正则化索引', 'description': '索引', 'physical_meaning': '索引', 'recommendation': '0', 'warnings':[]},
    'ML_FF_NMDINT': {'chinese_name': 'ML动力学间隔', 'description': '采样', 'physical_meaning': '采样', 'recommendation': '100', 'warnings':[]},
    'M_CONSTR': {'chinese_name': '约束质量', 'description': '惯性', 'physical_meaning': '惯性', 'recommendation': '0.0', 'warnings':[]},
    'NGX': {'chinese_name': 'X网格', 'description': '实空间', 'physical_meaning': '实空间', 'recommendation': '0', 'warnings':[]},
    'NGXF': {'chinese_name': 'X傅里叶网格', 'description': '倒空间', 'physical_meaning': '倒空间', 'recommendation': '0', 'warnings':[]},
    'NGY': {'chinese_name': 'Y网格', 'description': '实空间', 'physical_meaning': '实空间', 'recommendation': '0', 'warnings':[]},
    'NGYF': {'chinese_name': 'Y傅里叶网格', 'description': '倒空间', 'physical_meaning': '倒空间', 'recommendation': '0', 'warnings':[]},
    'NGYROMAG': {'chinese_name': '磁性实空间网格', 'description': '分辨率', 'physical_meaning': '分辨率', 'recommendation': '默认', 'warnings':[]},
    'NGZ': {'chinese_name': 'Z网格', 'description': '实空间', 'physical_meaning': '实空间', 'recommendation': '0', 'warnings':[]},
    'NGZF': {'chinese_name': 'Z傅里叶网格', 'description': '倒空间', 'physical_meaning': '倒空间', 'recommendation': '0', 'warnings':[]},
    'NSUBSYS': {'chinese_name': 'MD子系统', 'description': '配置', 'physical_meaning': '配置', 'recommendation': '默认', 'warnings':[]},
    'STM': {'chinese_name': 'STM模拟偏压', 'description': '显微镜', 'physical_meaning': '偏压', 'recommendation': '0.0', 'warnings':[]},
    'WC': {'chinese_name': '权重因子', 'description': '优化', 'physical_meaning': '优化步长', 'recommendation': '1.0', 'warnings':[]}
}

# 动态组装为内部快速调用的格式
INTEGRATED_PARAMS = {}
for k, v in INCAR_PARAMS_RAW.items():
    desc = v.get('description', '')
    phys = v.get('physical_meaning', '')
    text = desc + ("：" + phys if phys and phys not in desc else "")
    INTEGRATED_PARAMS[k] = {
        'chinese_name': v.get('chinese_name', k),
        'physical_meaning': text,
        'recommendation': v.get('recommendation', ''),
        'warnings': v.get('warnings',[])
    }

# ==========================================
# 核心逻辑模块
# ==========================================
def guess_calculation_type(incar):
    nsw = int(incar.get("NSW", 0))
    ibrion = int(incar.get("IBRION", -1))
    icharg = int(incar.get("ICHARG", 2))
    mlff = parse_vasp_bool(incar.get("ML_FF_LMLFF", False))
    
    if mlff: return "机器学习力场训练 (ML-FF)"
    elif nsw > 0 and ibrion > 0: return "结构优化 / 弛豫 (Geometry Relaxation)"
    elif nsw > 0 and ibrion == 0: return "分子动力学 (Molecular Dynamics)"
    elif nsw == 0 or ibrion == -1:
        if icharg == 11: return "能带 / 态密度计算 (Band Structure / DOS)"
        else: return "静态自洽计算 (Static SCF Point)"
    return "自定义计算类型"

def analyze_incar(user_incar, poscar, calc_type):
    structure = poscar.structure
    # 启发式判断 2D Slab 模型
    is_probably_2d = any(l > 14.0 for l in[structure.lattice.a, structure.lattice.b, structure.lattice.c])
    
    expert_set = MPRelaxSet(structure) if "Relaxation" in calc_type else MPStaticSet(structure)
    expert_incar = expert_set.incar
    
    # 提取 POSCAR 元素顺序（保持原始出现顺序且去重）
    poscar_el_seq =[]
    for site in structure.sites:
        sym = site.species_string
        if not poscar_el_seq or poscar_el_seq[-1] != sym:
            if sym not in poscar_el_seq:
                poscar_el_seq.append(sym)

    # 通过本地字典解析
    u_elements_found =[]
    rec_u_list = []
    rec_ul_list = []
    mag_elements_found =[]
    
    for sym in poscar_el_seq:
        u_val = 0
        u_l = -1
        if sym in DFT_U_VALUES:
            if DFT_U_VALUES[sym]['d'] > 0:
                u_val = DFT_U_VALUES[sym]['d']
                u_l = 2
            elif DFT_U_VALUES[sym]['f'] > 0:
                u_val = DFT_U_VALUES[sym]['f']
                u_l = 3
            if u_val > 0:
                u_elements_found.append(f"**{sym}** (U={u_val}eV)")
        rec_u_list.append(str(u_val))
        rec_ul_list.append(str(u_l))
        
        if sym in ELEMENT_MAGNETIC_MOMENTS and ELEMENT_MAGNETIC_MOMENTS[sym] > 0:
            mag_elements_found.append(f"**{sym}**")

    # 去重
    u_elements_found = list(dict.fromkeys(u_elements_found))
    mag_elements_found = list(dict.fromkeys(mag_elements_found))
    
    # 【彻底修复：U值诊断优先级】
    # 1. 优先提取底层 Pymatgen 材料库认为正确的 U 值
    expert_u_str = str(expert_incar.get("LDAUU", ""))
    expert_ul_str = str(expert_incar.get("LDAUL", ""))
    expert_u_list =[float(x) for x in expert_u_str.split() if x.replace('.','',1).lstrip('-').isdigit()]
    
    expert_has_nonzero_u = any(x > 0 for x in expert_u_list)
    local_has_nonzero_u = any(float(x) > 0 for x in rec_u_list)
    
    # 只要 Pymatgen 专家库或本地字典有一个觉得需要加非零 U 值，才算真的需要加 U
    needs_u_total = local_has_nonzero_u or expert_has_nonzero_u

    # 智能融合：如果专家库认为必须加非零U，而我们本地字典全是0 (例如 Bi W O)，强制采用专家库推荐值
    if expert_has_nonzero_u and not local_has_nonzero_u:
        final_rec_UU = expert_u_str
        final_rec_UL = expert_ul_str
    else:
        final_rec_UU = " ".join(rec_u_list)
        final_rec_UL = " ".join(rec_ul_list)

    # 磁矩处理：专家库拥有精确的带数字前缀的磁矩 (如 16*0.6)
    expert_mag_str = str(expert_incar.get("MAGMOM", ""))
    expert_has_mag = bool(expert_mag_str)
    local_has_mag = bool(mag_elements_found)
    needs_mag_total = local_has_mag or expert_has_mag
    
    final_rec_mag = expert_mag_str if expert_has_mag else " ".join([str(ELEMENT_MAGNETIC_MOMENTS.get(sym, 0)) for sym in poscar_el_seq])

    analysis_results = []
    top_warnings =[]
    all_tags = set(user_incar.keys()).union(set(expert_incar.keys()))
    
    is_user_ldau = parse_vasp_bool(user_incar.get("LDAU", False))
    is_user_spin = parse_vasp_bool(user_incar.get("ISPIN", False)) or str(user_incar.get("ISPIN", "")) in ["2", "2.0"]
    is_soc = parse_vasp_bool(user_incar.get("LSORBIT", False))
    is_hse = parse_vasp_bool(user_incar.get("LHFCALC", False))
    ibrion_val = user_incar.get("IBRION", "未设置")
    nsw_val = user_incar.get("NSW", "未设置")
    
    # 强制将这些关键标签纳入审查列表
    if needs_u_total:
        all_tags.update(["LDAU", "LDAUU", "LDAUL", "LDAUJ", "LMAXMIX"])
    if needs_mag_total:
        all_tags.update(["ISPIN", "MAGMOM", "LMAXMIX"])
    if is_probably_2d: 
        all_tags.add("IDIPOL")
    if "LORBIT" in user_incar or "DOS" in calc_type:
        all_tags.add("RWIGS")

    for tag in all_tags:
        user_val = user_incar.get(tag, "未设置")
        expert_val = expert_incar.get(tag, "未设置")
        
        # 从全量知识库中提取文案
        kbase = INTEGRATED_PARAMS.get(tag, {})
        if kbase:
            desc_text = f"**{kbase.get('chinese_name', tag)}**：{kbase.get('physical_meaning', '')}"
            if kbase.get('recommendation'): desc_text += f"<br>💡 <i>建议</i>：{kbase.get('recommendation')}"
            if kbase.get('warnings'): desc_text += f"<br>⚠️ <i>避坑</i>：{'；'.join(kbase.get('warnings'))}"
        else:
            desc_text = "VASP高级/偏僻参数，详情请查阅官方手册。"
            
        advice = "✅ 设置正常"
        
        # ----------------------------------------
        # 核心防呆诊断逻辑
        # ----------------------------------------
        if tag == "LDAU":
            if needs_u_total and (str(user_val) == "未设置" or not is_user_ldau):
                if u_elements_found:
                    advice = f"🚨 带隙塌陷警告: 检测到强关联元素 {', '.join(u_elements_found)}。必须开启 DFT+U 计算 (设置 LDAU=.TRUE.)！"
                else:
                    advice = f"🚨 带隙塌陷警告: 材料库指出该体系含强关联电子，必须开启 DFT+U 计算 (设置 LDAU=.TRUE.)！"
                top_warnings.append(advice)
                
        elif tag == "LDAUU":
            # 如果真实需要 U，但用户没设置或全设了0
            if needs_u_total and (str(user_val) == "未设置" or not any(float(x) > 0 for x in str(user_val).split() if x.replace('.','',1).lstrip('-').isdigit())):
                advice = f"🚨 U值缺失/错误: POSCAR 元素顺序为 **{' '.join(poscar_el_seq)}**。必须配套设置 LDAUU = **{final_rec_UU}**。"
                top_warnings.append(advice)

        elif tag == "LDAUL":
            if needs_u_total and str(user_val) == "未设置":
                advice = f"🚨 轨道通道缺失: 根据 POSCAR 元素，必须设置 LDAUL = **{final_rec_UL}** (-1不加，2代表d，3代表f)。"
                top_warnings.append(advice)

        elif tag == "MAGMOM":
            if needs_mag_total and str(user_val) == "未设置":
                if mag_elements_found:
                    advice = f"⚠️ 磁性丢失警告: 检测到磁性元素 {', '.join(mag_elements_found)}。若不赋予初始 MAGMOM，极易掉入非磁高能态！<br>推荐: MAGMOM = **{final_rec_mag}**"
                else:
                    advice = f"⚠️ 磁性丢失警告: Pymatgen 强烈建议该体系需赋予初始磁矩！<br>推荐: MAGMOM = **{final_rec_mag}**"
                top_warnings.append(advice)
                
        elif tag == "ISPIN":
            if needs_mag_total and not is_user_spin:
                advice = f"🚨 自旋关闭警告: 体系应具有磁性基态，必须强制开启自旋极化 (设置 ISPIN = 2)！"
                top_warnings.append(advice)

        elif tag == "LMAXMIX":
            if (is_user_spin or is_user_ldau or is_soc):
                if str(user_val) == "未设置" or int(user_val) < 4:
                    req_val = 6 if "3" in final_rec_UL else 4
                    advice = f"🚨 收敛黑洞警告: 因开启了磁性或+U，必须手动指定 LMAXMIX={req_val}，否则电荷极难收敛！"
                    top_warnings.append(advice)

        elif tag == "NSW":
            if str(user_val) != "未设置" and int(user_val) > 0 and ibrion_val in ["-1", "未设置"]:
                advice = "🚨 冲突: NSW>0(要求弛豫)，但 IBRION=-1(静态)，任务会报错停机！"
                top_warnings.append(advice)
                
        elif tag == "ISIF":
            if str(user_val) == "3" and is_probably_2d:
                advice = "🚨 毁灭性错误: 检测到大真空层 Slab 模型。绝对不能用 ISIF=3！真空会被压没导致废算，必须改 2 或 4。"
                top_warnings.append(advice)
                
        elif tag == "ISMEAR":
            if str(user_val) == "-5" and str(nsw_val) != "未设置" and int(nsw_val) > 0:
                advice = "🚨 物理错误: 结构弛豫【绝对不能】用 ISMEAR=-5 (四面体法)，会导致受力算错！立即改为 0(半导体) 或 1(金属)。"
                top_warnings.append(advice)

        elif tag == "IDIPOL":
            if is_probably_2d and str(user_val) == "未设置":
                advice = "⚠️ 偶极校正提示: 体系包含真空层。如果是表面不对称吸附或极性面，务必开启 IDIPOL=3 配合 LDIPOL=.TRUE. 防能级倾斜。"
                top_warnings.append(advice)

        elif tag == "RWIGS":
            if "LORBIT" in user_incar and int(user_incar.get("LORBIT", 10)) < 10 and str(user_val) == "未设置":
                advice = "⚠️ 半径缺失: LORBIT < 10 算 DOS 时，须手动设 RWIGS 原子半径数组！建议直接改 LORBIT = 11。"
                top_warnings.append(advice)

        if advice == "✅ 设置正常":
            if str(user_val) == "未设置":
                advice = f"ℹ️ 未设置，使用 VASP 默认值。(专家库参考: {expert_val})"
            elif str(expert_val) != "未设置" and str(user_val) != str(expert_val):
                advice = f"ℹ️ 提示: 您设置为 {user_val}，MP经典参考值为 {expert_val}。"

        analysis_results.append({
            "参数标签 (Tag)": f"**{tag}**",
            "您的设置": str(user_val),
            "专家库推荐": str(expert_val),
            "专家诊断与建议": advice,
            "内置百科": desc_text
        })
        
    df = pd.DataFrame(analysis_results)
    # 按危险等级排序
    df['优先级'] = df['专家诊断与建议'].apply(
        lambda x: 0 if "🚨" in x else (1 if "⚠️" in x else (2 if "✅" in x else 3))
    )
    df = df.sort_values(by=['优先级', '参数标签 (Tag)']).drop(columns=['优先级']).reset_index(drop=True)
    
    # 警告去重
    top_warnings = list(dict.fromkeys(top_warnings))
    return df, top_warnings, final_rec_UU, final_rec_UL, final_rec_mag, needs_u_total, needs_mag_total, poscar_el_seq

# ==========================================
# 网页前端渲染模块 (彻底修复折叠与源码泄漏)
# ==========================================
def render_html_table(df):
    """生成带有自适应换行 CSS 的原生 HTML"""
    df_html = df.copy()
    
    # 转译 Markdown 为 HTML 标签，并移除所有可能破坏 Streamlit 解析的原始隐藏换行符
    for col in df_html.columns:
        df_html[col] = df_html[col].astype(str).apply(lambda x: re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', x))
        
    raw_html = df_html.to_html(escape=False, index=False)
    
    # 强力 CSS 控制表格折行
    style = """
    <style>
    .vasp-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 14.5px;
        margin-bottom: 25px;
        line-height: 1.6;
        color: #1a1a1a;
    }
    .vasp-table th {
        background-color: #f1f3f5;
        color: #2c3e50;
        font-weight: 700;
        padding: 14px 16px;
        border: 1px solid #dee2e6;
        text-align: left;
    }
    .vasp-table td {
        padding: 14px 16px;
        border: 1px solid #dee2e6;
        word-wrap: break-word !important;
        white-space: normal !important;
        word-break: break-word !important;
        vertical-align: top;
    }
    .vasp-table tr:hover {
        background-color: #f8f9fa;
    }
    </style>
    """
    
    # 【100% 防泄漏修复】：把 \n 字符彻底替换为空，因为 Streamlit Markdown 遇 \n 会中断标签解析！
    combined_html = (style + raw_html).replace('\n', '')
    final_html = combined_html.replace('<table border="1" class="dataframe">', '<table class="vasp-table">')
    return final_html


st.title("🔬 VASP INCAR 专家级全量防呆审查系统")
st.markdown("> **底层引擎**：材料库高通量物理规则 + **全量 250+ 参数百科**  |  **特色**：解析POSCAR直连元素报警、100%强制换行显示。")

col1, col2 = st.columns(2)
with col1:
    incar_file = st.file_uploader("📂 上传您的 INCAR", type=None)
with col2:
    poscar_file = st.file_uploader("📂 上传您的 POSCAR", type=None)

if incar_file and poscar_file:
    try:
        incar_str = incar_file.read().decode("utf-8")
        poscar_str = poscar_file.read().decode("utf-8")
        
        user_incar = Incar.from_str(incar_str)
        user_poscar = Poscar.from_str(poscar_str)
        
        calc_type = guess_calculation_type(user_incar)
        st.success(f"**🤖 AI 自动推断该任务类型为**: 【{calc_type}】")
        
        with st.spinner("🧠 正在比对全量参数库知识与物理防呆规则..."):
            df_result, top_warnings, final_rec_UU, final_rec_UL, final_rec_mag, needs_u, needs_mag, poscar_el_seq = analyze_incar(user_incar, user_poscar, calc_type)
        
        # -----------------------------
        # 前置直达严重警告输出区
        # -----------------------------
        if top_warnings:
            st.markdown("### ⚠️ 核心物理报错速览")
            for warn in top_warnings:
                display_warn = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', warn)
                display_warn = display_warn.replace("<br>", " ")
                if "🚨" in warn:
                    st.error(display_warn, icon="🚨")
                else:
                    st.warning(display_warn, icon="⚠️")
            st.markdown("---")
        
        # -----------------------------
        # 强制自适应换行的纯 HTML 表格
        # -----------------------------
        st.subheader("📊 INCAR 深度审查与参数全百科")
        st.info("💡 下方表格支持文字自适应完全换行，不会折叠任何长文本。")
        
        # 注入处理过的安全 HTML 字符串
        st.markdown(render_html_table(df_result), unsafe_allow_html=True)
        
        # -----------------------------
        # 完美版 INCAR 下载生成
        # -----------------------------
        st.subheader("📥 智能纠错与补全：下载优化版 INCAR")
        st.markdown("系统已自动保留您原有的合理设置，并根据 `POSCAR` 为您精准填补了致命缺失参数。")
        
        expert_class = MPRelaxSet if "Relaxation" in calc_type else MPStaticSet
        perfect_incar = Incar(user_incar)
        expert_incar_data = expert_class(user_poscar.structure).incar
        
        # 智能补全核心参数
        if "ENCUT" not in perfect_incar and "ENCUT" in expert_incar_data:
            perfect_incar["ENCUT"] = expert_incar_data["ENCUT"]
            
        if needs_u:
            perfect_incar["LDAU"] = ".TRUE."
            perfect_incar["LDAUTYPE"] = 2
            perfect_incar["LDAUL"] = final_rec_UL
            perfect_incar["LDAUU"] = final_rec_UU
            perfect_incar["LDAUJ"] = " ".join(["0"] * len(poscar_el_seq))
            perfect_incar["LMAXMIX"] = 6 if "3" in final_rec_UL else 4
            
        if needs_mag:
            perfect_incar["ISPIN"] = 2
            if "MAGMOM" not in perfect_incar and final_rec_mag:
                perfect_incar["MAGMOM"] = final_rec_mag
                
        st.download_button(
            label="🔽 下载 AI 修复补全版 INCAR",
            data=str(perfect_incar),
            file_name="INCAR_Optimized",
            mime="text/plain",
            type="primary"
        )
        
    except Exception as e:
        st.error(f"❌ 解析异常！请检查文件是否为标准 VASP 格式。\n\n报错详情: {str(e)}")
