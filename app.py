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

def clean_val(v):
    """鲁棒地清洗并格式化 Pymatgen 的返回值，防止数组解析崩溃"""
    if v is None: return "未设置"
    if isinstance(v, list): 
        return " ".join(map(str, v)).replace('[','').replace(']','').replace(',',' ')
    return str(v)

# ==========================================
# 元素磁性与 DFT+U 知识库 
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
    'Ti': {'d': 3.5, 'f': None}, 'V': {'d': 3.5, 'f': None}, 'Cr': {'d': 3.5, 'f': None},
    'Mn': {'d': 3.5, 'f': None}, 'Fe': {'d': 3.5, 'f': None}, 'Co': {'d': 3.5, 'f': None},
    'Ni': {'d': 4.0, 'f': None}, 'Cu': {'d': 4.0, 'f': None},
    'Zn': {'d': 0, 'f': None}, 'Y': {'d': 0, 'f': None}, 'Zr': {'d': 0, 'f': None},
    'Nb': {'d': 0, 'f': None}, 'Mo': {'d': 0, 'f': None}, 'Tc': {'d': 0, 'f': None},
    'Ru': {'d': 0, 'f': None}, 'Rh': {'d': 0, 'f': None}, 'Pd': {'d': 0, 'f': None},
    'Ag': {'d': 0, 'f': None}, 'Hf': {'d': 0, 'f': None}, 'Ta': {'d': 0, 'f': None},
    'W': {'d': 0, 'f': None}, 'Re': {'d': 0, 'f': None}, 'Os': {'d': 0, 'f': None},
    'Ir': {'d': 0, 'f': None}, 'Pt': {'d': 0, 'f': None}, 'Au': {'d': 0, 'f': None},
    'La': {'d': 0, 'f': 6}, 'Ce': {'d': 0, 'f': 5}, 'Pr': {'d': 0, 'f': 5},
    'Nd': {'d': 0, 'f': 5}, 'Pm': {'d': 0, 'f': 5}, 'Sm': {'d': 0, 'f': 5},
    'Eu': {'d': 0, 'f': 6}, 'Gd': {'d': 0, 'f': 6}, 'Tb': {'d': 0, 'f': 5},
    'Dy': {'d': 0, 'f': 5}, 'Ho': {'d': 0, 'f': 5}, 'Er': {'d': 0, 'f': 5},
    'Tm': {'d': 0, 'f': 5}, 'Yb': {'d': 0, 'f': 4}, 'Lu': {'d': 0, 'f': 0},
    'U': {'d': 0, 'f': 4}, 'Np': {'d': 0, 'f': 4}, 'Pu': {'d': 0, 'f': 4}, 'Am': {'d': 0, 'f': 4}
}

# ==========================================
# 完整无删减 VASP 250+ 参数知识库
# ==========================================
INCAR_PARAMS_RAW = {
    'SYSTEM': {'chinese_name': '系统名称', 'description': '计算的系统名称或注释', 'physical_meaning': '用于标识计算的字符串', 'recommendation': '建议设置，描述计算内容'},
    'ISTART': {'chinese_name': '波函数初始化', 'description': '波函数初始化选项', 'physical_meaning': '0=从头开始，1=读取WAVECAR', 'recommendation': '首次0；续算1', 'warnings':['续算需确保文件存在']},
    'ICHARG': {'chinese_name': '电荷密度初始化', 'description': '电荷密度初始化方式', 'physical_meaning': '2=自洽，11=非自洽读取', 'recommendation': '能带计算用11或12', 'warnings':['能带/DOS计算必用11跳过自洽']},
    'ENCUT': {'chinese_name': '平面波截断能', 'description': '动能截断值 (eV)', 'physical_meaning': '决定计算精度的上限', 'recommendation': '设为最大ENMAX的1.0-1.3倍；变体积弛豫需1.3倍以上', 'warnings': ['过低不可靠；过高增加成本']},
    'PREC': {'chinese_name': '计算精度', 'description': '精度控制', 'physical_meaning': '影响FFT网格和基底', 'recommendation': '高精度用Accurate；常规用Normal'},
    'EDIFF': {'chinese_name': '电子收敛标准', 'description': '电子步能量差 (eV)', 'physical_meaning': 'SCF循环收敛标准', 'recommendation': '常规1E-5；高精度1E-6', 'warnings':['金属收敛难可适当放宽']},
    'EDIFFG': {'chinese_name': '离子收敛标准', 'description': '几何优化判据', 'physical_meaning': '正值=能量，负值=力标准(eV/Å)', 'recommendation': '推荐用力收敛: -1E-2 到 -1E-3', 'warnings': ['正值的能量收敛容易假收敛']},
    'IALGO': {'chinese_name': '电子算法', 'description': '对角化算法', 'physical_meaning': '子空间旋转的算法', 'recommendation': '38=Davidson，48=RMM-DIIS', 'warnings':['RMM-DIIS对某些体系不收敛']},
    'ISMEAR': {'chinese_name': '占据数展宽', 'description': 'Smearing方法', 'physical_meaning': '处理费米面占据', 'recommendation': '金属1或2，绝缘体0，静态DOS -5', 'warnings':['结构弛豫(NSW>0)绝对不能用ISMEAR=-5']},
    'SIGMA': {'chinese_name': '展宽宽度', 'description': 'Smearing宽度 (eV)', 'physical_meaning': '影响熵和收敛', 'recommendation': '金属: 0.1-0.2；绝缘体: 0.05', 'warnings': ['SIGMA过大引入严重误差']},
    'ISPIN': {'chinese_name': '自旋极化', 'description': '自旋总开关', 'physical_meaning': '1=闭壳层，2=开壳层', 'recommendation': '磁性材料必须设2', 'warnings': ['含磁性元素不开会导致能量极高']},
    'MAGMOM': {'chinese_name': '原子磁矩', 'description': '初始磁矩猜想 (μB)', 'physical_meaning': '初始自旋极化密度分布', 'recommendation': '过渡金属如 Fe=4, Co=3, Ni=2', 'warnings':['只有开启ISPIN=2时本参数才有效']},
    'LSORBIT': {'chinese_name': '自旋轨道耦合', 'description': 'SOC开关', 'physical_meaning': '相对论自旋轨道耦合', 'recommendation': '重元素、拓扑计算需设为.TRUE.', 'warnings': ['必须关闭对称性 ISYM=-1']},
    'ICHIBERN': {'chinese_name': '磁化方向', 'description': '非共线磁化', 'physical_meaning': '初始磁密方向分布', 'recommendation': '通常1'},
    'LDAU': {'chinese_name': 'DFT+U开关', 'description': 'Hubbard U总开关', 'physical_meaning': '对强关联d/f电子加U', 'recommendation': '过渡金属氧化物必须.TRUE.', 'warnings': ['不加U强关联带隙会严重偏小甚至变金属']},
    'LDAUTYPE': {'chinese_name': 'DFT+U方法', 'description': '加U方法', 'physical_meaning': '2=Dudarev(最常用)', 'recommendation': '2'},
    'LDAUL': {'chinese_name': 'U作用轨道', 'description': '加U的角动量', 'physical_meaning': '2=d, 3=f, -1=无', 'recommendation': '过渡金属用2；稀土用3', 'warnings': ['LDAU开启时必须设置，顺序需对应POSCAR']},
    'LDAUU': {'chinese_name': 'U值', 'description': '元素的U值 (eV)', 'physical_meaning': 'Hubbard U参数'},
    'LDAUJ': {'chinese_name': 'J值', 'description': 'Hund交换J', 'physical_meaning': 'Dudarev方法设0', 'recommendation': '0'},
    'LDAUPRINT': {'chinese_name': 'DFT+U输出', 'description': '输出控制', 'physical_meaning': '控制DFT+U日志', 'recommendation': '0'},
    'GGA': {'chinese_name': 'GGA泛函', 'description': '泛函类型', 'physical_meaning': '广义梯度近似具体形式', 'recommendation': 'PE=PBE'},
    'METAGGA': {'chinese_name': 'Meta-GGA', 'description': 'meta-GGA泛函', 'physical_meaning': '如SCAN泛函', 'recommendation': 'SCAN'},
    'LHFCALC': {'chinese_name': 'HF混合', 'description': '杂化泛函开关', 'physical_meaning': 'Hartree-Fock精确交换', 'recommendation': 'HSE计算设为.TRUE.', 'warnings': ['计算极其昂贵，配合ALGO=Damped']},
    'AEXX': {'chinese_name': 'HF交换比例', 'description': '精确交换比例', 'physical_meaning': 'HSE06为0.25', 'recommendation': '0.25'},
    'HFSCREEN': {'chinese_name': 'HF屏蔽参数', 'description': '屏蔽参数', 'physical_meaning': 'HSE屏蔽', 'recommendation': '0.207'},
    'IBRION': {'chinese_name': '离子优化算法', 'description': '原子移动算法', 'physical_meaning': '-1=固定，2=CG', 'recommendation': '结构优化设2', 'warnings':['只要做弛豫(NSW>0)绝不能为-1']},
    'ISIF': {'chinese_name': '优化自由度', 'description': '控制晶胞优化', 'physical_meaning': '2=仅原子，3=全优化', 'recommendation': '体块3；表面2', 'warnings':['二维表面大真空绝对不能用ISIF=3']},
    'NSW': {'chinese_name': '最大离子步数', 'description': '弛豫步数', 'physical_meaning': '最大结构优化步数', 'recommendation': '弛豫设100-300，静态设0'},
    'ISYM': {'chinese_name': '对称性', 'description': '对称开关', 'physical_meaning': '利用对称性加速', 'recommendation': '默认1', 'warnings': ['计算SOC时必须设为-1或0']},
    'LWAVE': {'chinese_name': '波函数输出', 'description': '输出WAVECAR', 'physical_meaning': '写波函数', 'recommendation': '续算时.TRUE.'},
    'LCHARG': {'chinese_name': '电荷密度输出', 'description': '输出CHGCAR', 'physical_meaning': '写电荷', 'recommendation': '能带前需.TRUE.'},
    'LVTOT': {'chinese_name': '静电势输出', 'description': '输出LOCPOT', 'physical_meaning': '写静电势', 'recommendation': '算功函数时.TRUE.'},
    'NELECT': {'chinese_name': '电子总数', 'description': '系统总电子数', 'physical_meaning': '用于带电缺陷', 'warnings':['设错会导致完全错误的计算']},
    'SMASS': {'chinese_name': '热浴参数', 'description': 'MD热浴', 'physical_meaning': '-3=NVT', 'recommendation': '-3'},
    'TEBEG': {'chinese_name': '初始温度', 'description': 'MD温度(K)', 'physical_meaning': '初始热力学温度', 'recommendation': '300'},
    'TEEND': {'chinese_name': '结束温度', 'description': 'MD结束温度(K)', 'physical_meaning': '退火模拟使用'},
    'IMAGES': {'chinese_name': '中间图像数', 'description': 'NEB插入图像', 'physical_meaning': 'NEB结构数', 'recommendation': '简单反应3-5', 'warnings': ['需配合IBRION=3或1']},
    'NELM': {'chinese_name': '最大电子迭代', 'description': 'SCF循环上限', 'physical_meaning': '电子自洽最大步数', 'recommendation': '常规60，磁性200'},
    'NELMIN': {'chinese_name': '最小电子迭代', 'description': '最小步数', 'physical_meaning': '防假收敛', 'recommendation': '2-6'},
    'NPAR': {'chinese_name': '并行参数', 'description': '能带并行', 'physical_meaning': '并行度', 'recommendation': '节点核心数的约数'},
    'NCORE': {'chinese_name': '并行能带数', 'description': '并行控制', 'physical_meaning': '每核能带', 'recommendation': '4-16', 'warnings': ['与NPAR二选一']},
    'AMIX': {'chinese_name': '电荷混合参数', 'description': '线性混合比', 'physical_meaning': 'SCF电荷混合', 'recommendation': '难收敛降至0.1', 'warnings': ['过大导致电荷震荡']},
    'BMIX': {'chinese_name': 'Kerker衰减', 'description': '长波衰减', 'physical_meaning': '防震荡'},
    'AMIX_MAG': {'chinese_name': '磁性混合参数', 'description': '自旋通道混合', 'physical_meaning': '磁性SCF', 'recommendation': '可增至1.6'},
    'BMIX_MAG': {'chinese_name': '磁性Kerker', 'description': '磁性长波衰减', 'physical_meaning': '防震荡'},
    'MAXMIX': {'chinese_name': '最大迭代历史', 'description': 'Broyden历史', 'physical_meaning': '电荷推断', 'recommendation': '难收敛设40'},
    'LREAL': {'chinese_name': '实空间投影', 'description': '局域投影空间', 'physical_meaning': '加速大体系', 'recommendation': '原子多用Auto，高精度用False'},
    'VOSKOWN': {'chinese_name': 'VWN插值', 'description': 'LDA相关', 'physical_meaning': '插值', 'recommendation': '1'},
    'NWRITE': {'chinese_name': '写入频率', 'description': 'OUTCAR详细度', 'physical_meaning': '输出级别', 'recommendation': '2'},
    'INIWAV': {'chinese_name': '初始波函数', 'description': '初猜波函数', 'physical_meaning': '生成方式', 'recommendation': '1'},
    'ADDGRID': {'chinese_name': '额外FFT网格', 'description': '细化网格', 'physical_meaning': '高精度电荷', 'recommendation': '.TRUE.'},
    'LSCALAPACK': {'chinese_name': 'ScaLAPACK并行', 'description': '并行库', 'physical_meaning': '大体系加速', 'recommendation': '.TRUE.'},
    'POTIM': {'chinese_name': '时间步长', 'description': '缩放/时间', 'physical_meaning': 'MD或弛豫步长', 'recommendation': '弛豫结构炸开时减小至0.1'},
    'RWIGS': {'chinese_name': 'Wigner-Seitz半径', 'description': '原子半径', 'physical_meaning': '决定Bader电荷与分波DOS投影', 'recommendation': '共价半径50-70%', 'warnings':['算DOS时不设且LORBIT<10会导致投影出错']},
    'RIMPODATA': {'chinese_name': '离子半径', 'description': '分析用'},
    'NBLOCK': {'chinese_name': '写入间隔', 'description': '数据写入频率', 'physical_meaning': '输出控制', 'recommendation': 'MD可增大'},
    'KBLOCK': {'chinese_name': '波函数块写入', 'description': '波函数频率', 'physical_meaning': '输出控制'},
    'LELF': {'chinese_name': '电子局域化', 'description': 'ELF函数', 'physical_meaning': '分析成键', 'recommendation': '分析时开启'},
    'LVHAR': {'chinese_name': '静电势输出', 'description': 'Hartree势', 'physical_meaning': '写入LOCPOT'},
    'LORBIT': {'chinese_name': '局域态密度', 'description': 'DOS投影控制', 'physical_meaning': '11=直接输出PROCAR无需RWIGS', 'recommendation': '强烈建议设为11'},
    'NEDOS': {'chinese_name': 'DOS能量点数', 'description': 'DOS平滑度', 'physical_meaning': '能量区间点数', 'recommendation': '画图用 1000-3000'},
    'EMAX': {'chinese_name': 'DOS能量范围', 'description': '能量上限', 'physical_meaning': 'DOS范围'},
    'EMIN': {'chinese_name': 'DOS最小能量', 'description': '能量下限', 'physical_meaning': 'DOS范围'},
    'WEIMIN': {'chinese_name': '权重最小值', 'description': '子空间防错', 'physical_meaning': '数值稳定', 'recommendation': '0.0001'},
    'EBREAK': {'chinese_name': '电子收敛阈值', 'description': '内部判断'},
    'SYMPREC': {'chinese_name': '对称性精度', 'description': '容差', 'physical_meaning': '找对称性容差', 'recommendation': '晶格稍畸变可改1E-6'},
    'SPRING': {'chinese_name': '弹簧常数', 'description': 'NEB力常数', 'physical_meaning': '图像间作用力', 'recommendation': '-5'},
    'LCLIMB': {'chinese_name': '爬坡', 'description': 'CI-NEB开关', 'physical_meaning': '找精准鞍点', 'recommendation': '.TRUE.'},
    'ALGO': {'chinese_name': '宏观算法', 'description': '电子步算法组合', 'physical_meaning': 'Normal, Fast, Damped', 'recommendation': '杂化泛函必须Damped/All'},
    'NBANDS': {'chinese_name': '能带数', 'description': '计算能带总数', 'physical_meaning': '占据与空带', 'recommendation': '算光学需成倍增加'},
    'KSPACING': {'chinese_name': 'k点间距', 'description': '自动K点网格', 'physical_meaning': '取代KPOINTS', 'recommendation': '绝缘0.2，金属0.15'},
    'KGAMMA': {'chinese_name': 'Gamma点', 'description': '居中网格', 'physical_meaning': 'KSPACING生成', 'recommendation': '.TRUE.'},
    'NKRED': {'chinese_name': 'k点缩减', 'description': '减少点数', 'physical_meaning': '加速HSE'},
    'NLSPLINE': {'chinese_name': 'k点插值', 'description': '样条', 'physical_meaning': '插值算法'},
    'IVDW': {'chinese_name': '范德华校正', 'description': '弱相互作用', 'physical_meaning': '色散力', 'recommendation': '11(D3) 或 12(D3-BJ)', 'warnings':['层状和吸附体系必须开启']},
    'VDW_S6': {'chinese_name': 'D3比例因子', 'description': '缩放'},
    'VDW_SR': {'chinese_name': 'D3短程', 'description': '短程'},
    'VDW_A1': {'chinese_name': 'D3_A1', 'description': 'BJ参数'},
    'VDW_A2': {'chinese_name': 'D3_A2', 'description': 'BJ参数'},
    'VDW_RADIUS': {'chinese_name': 'vdW半径', 'description': '截断'},
    'LUSE_VDW': {'chinese_name': '使用vdW', 'description': 'MBD校正'},
    'ENAUG': {'chinese_name': '增强截断能', 'description': 'PAW截断', 'recommendation': '1.5*ENCUT'},
    'ENCUTFOCK': {'chinese_name': 'FOCK截断能', 'description': '交换截断'},
    'ROPT': {'chinese_name': '投影精度', 'description': '实空间投影', 'recommendation': '-1E-3'},
    'LASPH': {'chinese_name': '非球面校正', 'description': '势场校正', 'physical_meaning': '精确PAW', 'recommendation': '加U和SOC体系强烈建议开启'},
    'LMAXFOCK': {'chinese_name': 'Fock最大l', 'description': '角动量'},
    'LMAXMIX': {'chinese_name': '混合最大l', 'description': '电荷混合L', 'physical_meaning': '高角动量恢复', 'recommendation': '含d且加U/SOC必设4，含f设6', 'warnings':['此项不设会导致磁性/加U体系严重难以收敛']},
    'MDALGO': {'chinese_name': 'MD算法', 'description': '积分算法', 'recommendation': '0=Verlet'},
    'LANGEVIN_GAMMA': {'chinese_name': 'Langevin阻尼', 'description': 'Langevin'},
    'PSTRESS': {'chinese_name': '静水压', 'description': '压力 (kB)', 'recommendation': '0.0'},
    'PMASS': {'chinese_name': '离子赝质量', 'description': '质量'},
    'LEPSILON': {'chinese_name': '高频介电', 'description': '介电常数', 'recommendation': '光学计算时设.TRUE.'},
    'LOPTICS': {'chinese_name': '光学计算', 'description': '介电虚部', 'recommendation': '算吸收谱设.TRUE.'},
    'CSHIFT': {'chinese_name': '复位移', 'description': '展宽参数'},
    'CLL': {'chinese_name': 'CL规范', 'description': '规范'},
    'ICORELEVEL': {'chinese_name': '芯能级', 'description': '芯空穴'},
    'ENCUTGW': {'chinese_name': 'GW截断能', 'description': 'GW响应截断', 'recommendation': '等于ENCUT'},
    'NOMEGA': {'chinese_name': '频率点数', 'description': 'GW频率积分', 'recommendation': '50'},
    'OMEGAMAX': {'chinese_name': '最大频率', 'description': 'GW上限'},
    'LWANNIER90': {'chinese_name': 'Wannier90', 'description': 'W90接口', 'recommendation': '投影需开启'},
    'LWANNIER90_RUN': {'chinese_name': 'Wannier运行', 'description': '直连W90'},
    'LNONCOLLINEAR': {'chinese_name': '非共线磁性', 'description': '非共线自旋', 'recommendation': '复杂磁结构开启'},
    'SAXIS': {'chinese_name': '自旋轴', 'description': '量化方向', 'recommendation': '0 0 1'},
    'ICHIBARE': {'chinese_name': '手征密度', 'description': '自旋流'},
    'IDIPOL': {'chinese_name': '偶极校正', 'description': '电场校正方向', 'physical_meaning': '抵消周期性边界假偶极', 'recommendation': '表面/不对称体系强烈建议设3', 'warnings':['大真空层单面吸附如果不加，能级会倾斜错误']},
    'LORBMOM': {'chinese_name': '轨道矩', 'description': '轨道磁矩', 'recommendation': 'SOC计算建议输出'},
    'NUPDOWN': {'chinese_name': '自旋差', 'description': '强制固定磁矩'},
    'LCALCPOL': {'chinese_name': '极化输出', 'description': 'Berry相位极化', 'recommendation': '铁电计算开启'},
    'LBERRY': {'chinese_name': 'Berry相', 'description': 'Berry曲率'},
    'I_CONSTRAINED_M': {'chinese_name': '磁矩约束', 'description': '约束原子'},
    'CONSTRAINED_M': {'chinese_name': '约束强度', 'description': '惩罚能'},
    'LAMBDA': {'chinese_name': '拉格朗日', 'description': '乘子'},
    'AGGAX': {'chinese_name': 'GGA交换', 'description': '比例'},
    'PHON_NSTRUCT': {'chinese_name': '声子结构数', 'description': '有限差分'},
    'IMIX': {'chinese_name': '混合方式', 'description': '底层混合'},
    'NELMDL': {'chinese_name': '延迟开始', 'description': '推迟电荷更新', 'recommendation': '极难收敛用-5'},
    'EFIELD': {'chinese_name': '电场', 'description': '外加电场'},
    'EFIELD_PEAD': {'chinese_name': 'PEAD场', 'description': '有效电场'},
    'FERWE': {'chinese_name': 'Fermi面权重', 'description': '权重'},
    'MAXMEM': {'chinese_name': '最大内存', 'description': '单核上限', 'recommendation': '内存够可加大'},
    'NSIM': {'chinese_name': '同时迭代', 'description': '能带成组', 'recommendation': '4'},
    'LASYNC': {'chinese_name': '异步IO', 'description': 'IO加速', 'recommendation': '大超胞用.TRUE.'},
    'GGA_COMPAT': {'chinese_name': 'GGA兼容', 'description': '向后兼容'},
    'PRECFOCK': {'chinese_name': 'FOCK精度', 'description': '精确交换精度', 'recommendation': 'Accurate'},
    'ENCUTLF': {'chinese_name': 'LF截断', 'description': '局域场'},
    'DARWINR': {'chinese_name': 'Darwin标量', 'description': '标量相对论'},
    'DARWINV': {'chinese_name': 'Darwin矢量', 'description': '矢量相对论'},
    'LSOL': {'chinese_name': '溶剂化', 'description': '隐式溶剂模型', 'warnings':['必须重新编译VASPsol才能生效']},
    'LADDER': {'chinese_name': '能带输出', 'description': 'BSE相关'},
    'LAECHG': {'chinese_name': '全电荷密度', 'description': 'Bader电荷用', 'recommendation': '需要算Bader时设.TRUE.'},
    'LPARD': {'chinese_name': '投影态密度', 'description': '部分电荷'},
    'NBMOD': {'chinese_name': '能带模式', 'description': '模式'},
    'IBAND': {'chinese_name': '能带索引', 'description': '计算指定能带'},
    'EINT': {'chinese_name': '能量范围', 'description': '部分积分能量'},
    'DIPOL': {'chinese_name': '偶极中心', 'description': '参考中心'},
    'AMIN': {'chinese_name': '最小混合', 'description': '混合下限'},
    'LMODELHF': {'chinese_name': '模型HF', 'description': '屏蔽交换'},
    'HFLMAX': {'chinese_name': 'HF最大l', 'description': '角动量'},
    'HFRCUT': {'chinese_name': 'HF截断', 'description': '实空间'},
    'LRHFCALC': {'chinese_name': '相对论HF', 'description': 'HF'},
    'LHFONE': {'chinese_name': '单中心HF', 'description': '单中心'},
    'HFSCREENC': {'chinese_name': '屏蔽类型', 'description': '类型'},
    'CMBJ': {'chinese_name': 'MBJ势', 'description': '校正带隙'},
    'CMBJA': {'chinese_name': 'MBJ参数A', 'description': 'A'},
    'CMBJB': {'chinese_name': 'MBJ参数B', 'description': 'B'},
    'LNICSALL': {'chinese_name': 'NMR位移', 'description': 'NMR张量'},
    'LCHIMAG': {'chinese_name': '化学位移', 'description': 'NMR'},
    'LDOWNSAMPLE': {'chinese_name': '降采样', 'description': '减小数据'},
    'ANDERSEN_PROB': {'chinese_name': 'Andersen概率', 'description': 'MD'},
    'HILLS_BIN': {'chinese_name': 'Hills采样', 'description': 'Metadynamics'},
    'HILLS_H': {'chinese_name': 'Hills高度', 'description': '高度'},
    'HILLS_W': {'chinese_name': 'Hills宽度', 'description': '宽度'},
    'APACO': {'chinese_name': '层间距', 'description': '径向分布'},
    'NPACO': {'chinese_name': 'PACO点数', 'description': '点数'},
    'TIME': {'chinese_name': '时间参数', 'description': 'MD时间'},
    'STEP_MAX': {'chinese_name': '最大步长', 'description': 'MD步长'},
    'STEP_SIZE': {'chinese_name': '步长', 'description': '步数'},
    'MINROT': {'chinese_name': '最小旋转', 'description': '旋转容差'},
    'MIXFIRST': {'chinese_name': '先混合', 'description': '先混后算'},
    'ANORTH': {'chinese_name': '非正交盒', 'description': '非正交'},
    'LATTICE_CONSTRAINTS': {'chinese_name': '晶格约束', 'description': '限制轴'},
    'QSPIRAL': {'chinese_name': '螺旋q矢量', 'description': 'q矢量'},
    'LANGEVIN_GAMMA_L': {'chinese_name': '晶格阻尼', 'description': '阻尼'},
    'SCSRAD': {'chinese_name': 'SCS半径', 'description': '自洽筛选'},
    'TSUBSYS': {'chinese_name': '热浴', 'description': '独立温度'},
    'VCUTOFF': {'chinese_name': '截断速度', 'description': '速度'},
    'OFIELD_A': {'chinese_name': '有序场A', 'description': '结晶'},
    'OFIELD_KAPPA': {'chinese_name': '有序场kappa', 'description': '结晶'},
    'OFIELD_Q6_FAR': {'chinese_name': 'Q6远场', 'description': '参数'},
    'OFIELD_Q6_NEAR': {'chinese_name': 'Q6近场', 'description': '参数'},
    'LEFG': {'chinese_name': 'EFG', 'description': '电场梯度', 'recommendation': '算NMR时开启'},
    'QUAD_EFG': {'chinese_name': '四极矩', 'description': '四极矩'},
    'RANDOM_SEED': {'chinese_name': '随机种子', 'description': 'MD种子'},
    'PARAM1': {'chinese_name': '参数1', 'description': '自定义'},
    'PARAM2': {'chinese_name': '参数2', 'description': '自定义'},
    'LGAUGE': {'chinese_name': '规范固定', 'description': '规范'},
    'LRPAFORCE': {'chinese_name': 'RPA力', 'description': 'RPA受力'},
    'LFXC': {'chinese_name': 'FXC', 'description': '自能'},
    'LTCTE': {'chinese_name': 'TCTE', 'description': 'RPA总能'},
    'LTETE': {'chinese_name': 'TETE', 'description': '能量'},
    'LTRIPLET': {'chinese_name': '三态', 'description': '三重激发'},
    'LUSEW': {'chinese_name': 'USEW', 'description': 'W矩阵'},
    'NUCIND': {'chinese_name': '核独立', 'description': '同位素'},
    'NTAUPAR': {'chinese_name': '时间并行', 'description': '并行'},
    'NTARGET_STATES': {'chinese_name': '目标态', 'description': '激发态'},
    'LOCPROJ': {'chinese_name': '局域投影', 'description': '投影轨道'},
    'POMASS': {'chinese_name': '离子质量', 'description': '质量'},
    'PROUTINE': {'chinese_name': '打印程序', 'description': '底层打印'},
    'PTHRESHOLD': {'chinese_name': '打印阈值', 'description': '底层'},
    'LMUSIC': {'chinese_name': 'MUSIC', 'description': 'MUSIC接口'},
    'SHIFTRED': {'chinese_name': '偏移缩减', 'description': '减小计算'},
    'NKREDX': {'chinese_name': 'X向K缩减', 'description': 'X方向'},
    'NKREDY': {'chinese_name': 'Y向K缩减', 'description': 'Y方向'},
    'NKREDZ': {'chinese_name': 'Z向K缩减', 'description': 'Z方向'},
    'KPOINT_BSE': {'chinese_name': 'BSE k点', 'description': 'BSE点'},
    'KPUSE': {'chinese_name': '使用k点', 'description': '使用K'},
    'EVENONLY': {'chinese_name': '偶k点', 'description': '偶数K'},
    'EVENONLYGW': {'chinese_name': 'GW偶k点', 'description': 'GW'},
    'ODDONLY': {'chinese_name': '奇k点', 'description': '奇数K'},
    'ODDONLYGW': {'chinese_name': 'GW奇k点', 'description': 'GW'},
    'NBANDSGW': {'chinese_name': 'GW能带数', 'description': 'GW计算'},
    'NBANDSO': {'chinese_name': '占据能带数', 'description': '占据'},
    'NBANDSV': {'chinese_name': '虚能带数', 'description': '空带'},
    'NOMEGAPAR': {'chinese_name': '频率并行', 'description': '并行'},
    'NOMEGAR': {'chinese_name': '实频率点', 'description': '采样'},
    'OMEGAMIN': {'chinese_name': '最小频率', 'description': '最小'},
    'OMEGATL': {'chinese_name': '频率尾参数', 'description': '积分'},
    'SELFENERGY': {'chinese_name': '自能计算', 'description': 'GW自能'},
    'LFERMIGW': {'chinese_name': 'Fermi更新', 'description': '更新费米'},
    'LSINGLES': {'chinese_name': '单粒子', 'description': '近似'},
    'ALDA': {'chinese_name': 'ALDA校正', 'description': 'TDDFT'},
    'ENCUTGWSOFT': {'chinese_name': 'GW软截断', 'description': '软'},
    'ENINI': {'chinese_name': '初始能量', 'description': '下限'},
    'PHON_LBOSE': {'chinese_name': '声子展宽', 'description': '展宽'},
    'PHON_LMC': {'chinese_name': '声子MC', 'description': '蒙特卡洛'},
    'PHON_NTLIST': {'chinese_name': '声子点', 'description': '点数'},
    'PHON_TLIST': {'chinese_name': '声子温度', 'description': '温度'},
    'WANPROJ': {'chinese_name': 'Wannier投影', 'description': 'W90投影'},
    'LWRITE_MMN_AMN': {'chinese_name': '写MMN/AMN', 'description': 'W90文件'},
    'LWRITE_UNK': {'chinese_name': '写UNK', 'description': 'W90文件'},
    'LWRITE_WANPROJ': {'chinese_name': '写投影', 'description': 'W90'},
    'CH_LSPEC': {'chinese_name': '芯空穴谱', 'description': '芯空穴'},
    'CH_NEDOS': {'chinese_name': '空穴DOS点', 'description': '点数'},
    'CH_SIGMA': {'chinese_name': '空穴展宽', 'description': '展宽'},
    'CLN': {'chinese_name': 'CL规范', 'description': '参数'},
    'CLNT': {'chinese_name': 'CL类型', 'description': '参数'},
    'CLZ': {'chinese_name': 'CL_Z', 'description': '参数'},
    'IEPSILON': {'chinese_name': '介电索引', 'description': '索引'},
    'IGPAR': {'chinese_name': '光学方向', 'description': '方向'},
    'IPEAD': {'chinese_name': 'PEAD', 'description': '相因子'},
    'LORBITALREAL': {'chinese_name': '实空间轨道', 'description': '投影'},
    'NMAXFOCKAE': {'chinese_name': 'AE最大索引', 'description': 'AE'},
    'AGGAC': {'chinese_name': 'GGA相关', 'description': '相关能'},
    'ALDAC': {'chinese_name': 'LDA相关', 'description': '相关能'},
    'LMIXTAU': {'chinese_name': '自旋混合', 'description': '常数'},
    'LNABLA': {'chinese_name': '梯度输出', 'description': '梯度'},
    'MAGPOS': {'chinese_name': '磁矩位置', 'description': '位置'},
    'ORBITALMAG': {'chinese_name': '轨道磁性', 'description': '轨道磁性', 'recommendation': 'SOC计算开启'},
    'MAGDIPOLOUT': {'chinese_name': '磁偶极输出', 'description': '偶极'},
    'ISPIND': {'chinese_name': '分立自旋', 'description': '分立'},
    'ICALCEPS': {'chinese_name': '介电开关', 'description': '宏观介电'},
    'FINDIFF': {'chinese_name': '有限差分', 'description': '差分'},
    'DQ': {'chinese_name': '位移增量', 'description': '增量'},
    'DEPER': {'chinese_name': '能量步长', 'description': '步长'},
    'DIMER_DIST': {'chinese_name': '二聚体距离', 'description': 'Dimer距离'},
    'IWAVPR': {'chinese_name': '波函数处理', 'description': '处理'},
    'LCOMPAT': {'chinese_name': '兼容性', 'description': 'VASP4兼容'},
    'LCORR': {'chinese_name': '电荷校正', 'description': 'Harris-Foulkes'},
    'LDIAG': {'chinese_name': '对角化', 'description': '子空间'},
    'LDIPOL': {'chinese_name': '局域偶极校正', 'description': '静电势修正', 'recommendation': '配合IDIPOL使用'},
    'LLRAUG': {'chinese_name': 'LR_AUG', 'description': '平滑增强'},
    'LSYMGRAD': {'chinese_name': '对称梯度', 'description': '对称加速'},
    'VALUE_MAX': {'chinese_name': '最大值', 'description': '约束条件'},
    'VALUE_MIN': {'chinese_name': '最小值', 'description': '约束条件'},
    'ENMAX': {'chinese_name': '最大ENMAX', 'description': '来自POTCAR'},
    'ENMIN': {'chinese_name': '最小ENMIN', 'description': '来自POTCAR'},
    'ENAVG': {'chinese_name': '平均截断能', 'description': '来自POTCAR'},
    'PFLAT': {'chinese_name': 'PFLAT', 'description': '投影'},
    'PSUBSYS': {'chinese_name': '参数子系统', 'description': '组'},
    'QMAXFOCKAE': {'chinese_name': 'QMAX_AE', 'description': '交换'},
    'ZVAL': {'chinese_name': 'ZVAL', 'description': '价电子数'},
    'NBLK': {'chinese_name': '输出块大小', 'description': '矩阵操作'},
    'NCRPA_BANDS': {'chinese_name': 'CRPA能带', 'description': '数量'},
    'NPPSTR': {'chinese_name': '投影方向', 'description': '方向'},
    'NBSEEIG': {'chinese_name': 'BSE本征值', 'description': '数量'},
    'PLEVEL': {'chinese_name': '打印级别', 'description': '打印'},
    'INIMIX': {'chinese_name': '初始混合', 'description': '初态混合'},
    'MIXPRE': {'chinese_name': '混合预处理', 'description': '预处理'},
    'NFREE': {'chinese_name': '有限差分步数', 'description': '力常数'},
    'NDAV': {'chinese_name': 'Davidson迭代', 'description': '最大迭代'},
    'INCREM': {'chinese_name': '增量参数', 'description': '搜索'},
    'ANTIRES': {'chinese_name': '反共振计算', 'description': '光学'},
    'HITOLER': {'chinese_name': '高精度容差', 'description': '高精度'},
    'SHAKEMAXITER': {'chinese_name': 'Shake迭代', 'description': '约束MD'},
    'SHAKETOL': {'chinese_name': 'Shake容差', 'description': '约束MD'},
    'EPSILON': {'chinese_name': '介电常数', 'description': '溶剂化模型'},
    'SMEARINGS': {'chinese_name': 'Smearing列表', 'description': '列表'},
    'LGauss': {'chinese_name': '高斯展宽', 'description': '高斯'},
    'LVDWEXPANSION': {'chinese_name': 'vdW展开', 'description': '展开'},
    'LVDW_EWALD': {'chinese_name': 'vdW Ewald', 'description': '求和'},
    'VDW_C6': {'chinese_name': 'C6系数', 'description': 'C6'},
    'VDW_R0': {'chinese_name': 'R0半径', 'description': 'R0'},
    'VDW_CNRADIUS': {'chinese_name': '截断半径', 'description': '截断'},
    'VDW_D': {'chinese_name': 'D参数', 'description': '阻尼'},
    'VDW_S8': {'chinese_name': 'S8参数', 'description': '缩放'},
    'ZAB_VDW': {'chinese_name': 'vdW半径', 'description': '半径'},
    'TAU': {'chinese_name': '温度耦合', 'description': 'MD热浴'},
    'LTEEPS': {'chinese_name': 'EEPS', 'description': '输出'},
    'LTHOMAS': {'chinese_name': 'Thomas', 'description': '输出'},
    'LFXCEPS': {'chinese_name': 'FXC_EPS', 'description': '输出'},
    'LFXHEG': {'chinese_name': 'FXC_HEG', 'description': '输出'},
    'LMAGBLOCH': {'chinese_name': '磁性Bloch变换', 'description': 'Bloch'},
    'LBLUEOUT': {'chinese_name': 'Bloch校正输出', 'description': '校正'},
    'LBONE': {'chinese_name': 'BondOrder输出', 'description': '分析'},
    'LCALCEPS': {'chinese_name': '介电常数输出', 'description': '输出'},
    'LHARTREE': {'chinese_name': 'Hartree势输出', 'description': '输出'},
    'LHYPERFINE': {'chinese_name': '超精细输出', 'description': '参数'},
    'LPEAD': {'chinese_name': 'PEAD输出', 'description': '输出'},
    'LPLANE': {'chinese_name': '平面波输出', 'description': '系数'},
    'LRPA': {'chinese_name': 'RPA输出', 'description': '参数'},
    'LSCAAWARE': {'chinese_name': 'SCA启用', 'description': '标度'},
    'LSCALU': {'chinese_name': 'LU分解输出', 'description': '输出'},
    'LSCSGRAD': {'chinese_name': 'SCS梯度输出', 'description': '梯度'},
    'LSELFENERGY': {'chinese_name': '自能输出', 'description': '输出'},
    'LSEPB': {'chinese_name': '分离带输出', 'description': '文件'},
    'LSEPK': {'chinese_name': '分离k点输出', 'description': '文件'},
    'LSPECTRAL': {'chinese_name': '谱函数输出', 'description': '输出'},
    'LSPECTRALGW': {'chinese_name': 'GW谱函数', 'description': '输出'},
    'LSPIRAL': {'chinese_name': '螺旋输出', 'description': '结构'},
    'LSUBROT': {'chinese_name': '子旋转输出', 'description': '输出'},
    'LZEROZ': {'chinese_name': 'Z方向零点', 'description': '输出'},
    'ISEARCH': {'chinese_name': '原子位置搜索', 'description': '算法'},
    'LFOCKAEDFT': {'chinese_name': 'HSE精确交换', 'description': '内层核'},
    'LKPROJ': {'chinese_name': 'Wannier投影', 'description': '投影'},
    'LMAXFOCKAE': {'chinese_name': 'Fock算符最大L', 'description': 'L'},
    'LMAXPAW': {'chinese_name': 'PAW投影最大L', 'description': 'L'},
    'LMAXTAU': {'chinese_name': '张力计算最大L', 'description': 'L'},
    'LMETAGGA': {'chinese_name': 'meta-GGA计算', 'description': '启用'},
    'LMONO': {'chinese_name': '单极矩计算', 'description': '静电'},
    'LNMR_SYM_RED': {'chinese_name': 'NMR对称性约化', 'description': '约化'},
    'LVEL': {'chinese_name': '速度计算', 'description': '原子速度'},
    'ML_MODE': {'chinese_name': 'ML训练模式', 'description': '模式'},
    'ML_FF_LMLFF': {'chinese_name': '机器学习力场', 'description': 'MLFF开关'},
    'ML_FF_LMLMB': {'chinese_name': 'ML多体势能面', 'description': '多体'},
    'ML_FF_ISTART': {'chinese_name': 'ML初始化模式', 'description': '预测或训练'},
    'ML_FF_MCONF': {'chinese_name': 'ML训练构型数', 'description': '库大小'},
    'ML_FF_MCONF_NEW': {'chinese_name': 'ML新构型数', 'description': '增量'},
    'ML_FF_MHIS': {'chinese_name': 'ML历史步数', 'description': '推断'},
    'ML_FF_LCONF_DISCARD': {'chinese_name': 'ML丢弃低置信度', 'description': '过滤'},
    'ML_FF_LBASIS_DISCARD': {'chinese_name': 'ML丢弃基组', 'description': '过滤'},
    'ML_FF_LCRITERIA': {'chinese_name': 'ML使用学习标准', 'description': '标准'},
    'ML_FF_LEATOM_MB': {'chinese_name': 'ML使用原子能量', 'description': '参考能'},
    'ML_FF_LHEAT_MB': {'chinese_name': 'ML计算热流', 'description': '分析'},
    'ML_FF_CSIG': {'chinese_name': 'ML信号噪声比', 'description': '阈值'},
    'ML_FF_CSLOPE': {'chinese_name': 'ML斜率缩放', 'description': '缩放'},
    'ML_FF_CTIFOR': {'chinese_name': 'ML离子力置信', 'description': '阈值'},
    'ML_FF_WTIFOR': {'chinese_name': 'ML离子力权重', 'description': '力权重'},
    'ML_FF_WTOTEN': {'chinese_name': 'ML能量权重', 'description': '能权重'},
    'ML_FF_WTSIF': {'chinese_name': 'ML应力权重', 'description': '应力权重'},
    'ML_FF_NWRITE': {'chinese_name': 'ML写入模式', 'description': '输出'},
    'ML_FF_ISAMPLE': {'chinese_name': 'ML采样模式', 'description': '策略'},
    'ML_FF_NDIM_SCALAPACK': {'chinese_name': 'ML维数', 'description': '矩阵'},
    'ML_FF_IERR': {'chinese_name': 'ML错误处理', 'description': '容错'},
    'ML_FF_IWEIGHT': {'chinese_name': 'ML权重计算', 'description': '加权'},
    'ML_FF_AFILT2_MB': {'chinese_name': 'ML二阶滤波', 'description': '滤波'},
    'ML_FF_LAFILT2_MB': {'chinese_name': 'ML启用二阶滤波', 'description': '滤波'},
    'ML_FF_IAFILT2_MB': {'chinese_name': 'ML原子滤波指标', 'description': '滤波'},
    'ML_FF_LMAX2_MB': {'chinese_name': 'ML第二角动量', 'description': '角动量'},
    'ML_FF_LNORM1_MB': {'chinese_name': 'ML第一归一化', 'description': '归一'},
    'ML_FF_LNORM2_MB': {'chinese_name': 'ML第二归一化', 'description': '归一'},
    'ML_FF_NR1_MB': {'chinese_name': 'ML第一径向网格', 'description': '网格'},
    'ML_FF_NR2_MB': {'chinese_name': 'ML第二径向网格', 'description': '网格'},
    'ML_FF_NHYP1_MB': {'chinese_name': 'ML第一双曲势', 'description': '势'},
    'ML_FF_NHYP2_MB': {'chinese_name': 'ML第二双曲势', 'description': '势'},
    'ML_FF_MRB1_MB': {'chinese_name': 'ML第一径向基', 'description': '基'},
    'ML_FF_MRB2_MB': {'chinese_name': 'ML第二径向基', 'description': '基'},
    'ML_FF_MSPL1_MB': {'chinese_name': 'ML第一样条点', 'description': '样条'},
    'ML_FF_MSPL2_MB': {'chinese_name': 'ML第二样条点', 'description': '样条'},
    'ML_FF_SION1_MB': {'chinese_name': 'ML第一离子噪声', 'description': '噪声'},
    'ML_FF_SION2_MB': {'chinese_name': 'ML第二离子噪声', 'description': '噪声'},
    'ML_FF_IBROAD1_MB': {'chinese_name': 'ML第一广播索引', 'description': '索引'},
    'ML_FF_IBROAD2_MB': {'chinese_name': 'ML第二广播索引', 'description': '索引'},
    'ML_FF_ICUT1_MB': {'chinese_name': 'ML第一截断索引', 'description': '索引'},
    'ML_FF_ICUT2_MB': {'chinese_name': 'ML第二截断索引', 'description': '索引'},
    'ML_FF_RCUT1_MB': {'chinese_name': 'ML第一截断半径', 'description': '截断'},
    'ML_FF_RCUT2_MB': {'chinese_name': 'ML第二截断半径', 'description': '截断'},
    'ML_FF_ISOAP1_MB': {'chinese_name': 'ML第一SOAP', 'description': 'SOAP'},
    'ML_FF_ISOAP2_MB': {'chinese_name': 'ML第二SOAP', 'description': 'SOAP'},
    'ML_FF_W1_MB': {'chinese_name': 'ML权重因子1', 'description': '因子'},
    'ML_FF_W2_MB': {'chinese_name': 'ML权重因子2', 'description': '因子'},
    'ML_FF_MB_MB': {'chinese_name': 'ML多体矩阵', 'description': '多体'},
    'ML_FF_EATOM': {'chinese_name': 'ML原子能量', 'description': '参考'},
    'ML_FF_CDOUB': {'chinese_name': 'ML双层因子', 'description': '因子'},
    'ML_FF_CSF': {'chinese_name': 'ML置信度缩放', 'description': '缩放'},
    'ML_FF_SIGV0_MB': {'chinese_name': 'ML势能噪声', 'description': '先验'},
    'ML_FF_SIGW0_MB': {'chinese_name': 'ML力噪声', 'description': '先验'},
    'ML_FF_ISCALE_TOTEN_MB': {'chinese_name': 'ML能量缩放', 'description': '缩放'},
    'ML_FF_ICOUPLE_MB': {'chinese_name': 'ML耦合索引', 'description': '耦合'},
    'ML_FF_LCOUPLE_MB': {'chinese_name': 'ML启用耦合', 'description': '耦合'},
    'ML_FF_RCOUPLE_MB': {'chinese_name': 'ML耦合半径', 'description': '耦合'},
    'ML_FF_NATOM_COUPLED_MB': {'chinese_name': 'ML耦合原子数', 'description': '原子'},
    'ML_FF_IREG_MB': {'chinese_name': 'ML正则化索引', 'description': '正则化'},
    'ML_FF_NMDINT': {'chinese_name': 'ML动力学间隔', 'description': '步长'},
    'M_CONSTR': {'chinese_name': '约束质量', 'description': 'MD约束'},
    'NGX': {'chinese_name': 'X网格', 'description': 'FFT'},
    'NGXF': {'chinese_name': 'X傅里叶网格', 'description': 'FFT细网格'},
    'NGY': {'chinese_name': 'Y网格', 'description': 'FFT'},
    'NGYF': {'chinese_name': 'Y傅里叶网格', 'description': 'FFT细网格'},
    'NGYROMAG': {'chinese_name': '磁性实空间网格', 'description': '分辨率'},
    'NGZ': {'chinese_name': 'Z网格', 'description': 'FFT'},
    'NGZF': {'chinese_name': 'Z傅里叶网格', 'description': 'FFT细网格'},
    'NSUBSYS': {'chinese_name': 'MD子系统', 'description': '多温控'},
    'STM': {'chinese_name': 'STM模拟偏压', 'description': '扫描隧道偏压'},
    'WC': {'chinese_name': '权重因子', 'description': '自洽权重'}
}

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
    is_probably_2d = any(l > 14.0 for l in[structure.lattice.a, structure.lattice.b, structure.lattice.c])
    
    expert_set = MPRelaxSet(structure) if "Relaxation" in calc_type else MPStaticSet(structure)
    expert_incar = expert_set.incar
    
    poscar_el_seq =[]
    for site in structure.sites:
        sym = site.species_string
        if not poscar_el_seq or poscar_el_seq[-1] != sym:
            if sym not in poscar_el_seq:
                poscar_el_seq.append(sym)

    # ==== 【提取 U 值逻辑】 ====
    rec_u_list = []
    rec_ul_list = []
    
    for sym in poscar_el_seq:
        u_val = 0
        u_l = -1
        if sym in DFT_U_VALUES:
            if DFT_U_VALUES[sym]['d'] is not None and DFT_U_VALUES[sym]['d'] > 0:
                u_val = DFT_U_VALUES[sym]['d']
                u_l = 2
            elif DFT_U_VALUES[sym]['f'] is not None and DFT_U_VALUES[sym]['f'] > 0:
                u_val = DFT_U_VALUES[sym]['f']
                u_l = 3
        rec_u_list.append(str(u_val))
        rec_ul_list.append(str(u_l))

    expert_u_str = clean_val(expert_incar.get("LDAUU"))
    expert_ul_str = clean_val(expert_incar.get("LDAUL"))
    expert_u_list = [float(x) for x in expert_u_str.split() if x.replace('.','',1).lstrip('-').isdigit()]
    
    expert_has_nonzero_u = any(x > 0 for x in expert_u_list)
    local_has_nonzero_u = any(float(x) > 0 for x in rec_u_list)
    
    needs_u_physical = local_has_nonzero_u or expert_has_nonzero_u

    if expert_has_nonzero_u and not local_has_nonzero_u:
        final_rec_UU = expert_u_str
        final_rec_UL = expert_ul_str
    else:
        final_rec_UU = " ".join(rec_u_list)
        final_rec_UL = " ".join(rec_ul_list)

    # 构建带确切数值的 U 提示文本
    u_summary_list = []
    final_u_vals = final_rec_UU.split()
    for idx, sym in enumerate(poscar_el_seq):
        if idx < len(final_u_vals):
            try:
                val = float(final_u_vals[idx])
                if val > 0:
                    u_summary_list.append(f"**{sym}** (推荐 U={val} eV)")
            except:
                pass
    u_summary_text = "、".join(u_summary_list) if u_summary_list else "无强关联原子，无需加 U"

    # ==== 【提取磁矩逻辑】 ====
    expert_mag_str = clean_val(expert_incar.get("MAGMOM"))
    expert_has_mag = bool(expert_mag_str) and expert_mag_str != "未设置"
    
    local_has_mag = False
    for sym in poscar_el_seq:
        if sym in ELEMENT_MAGNETIC_MOMENTS and ELEMENT_MAGNETIC_MOMENTS[sym] > 0:
            local_has_mag = True
            break

    needs_mag_physical = local_has_mag or expert_has_mag
    final_rec_mag = expert_mag_str if expert_has_mag else " ".join([str(ELEMENT_MAGNETIC_MOMENTS.get(sym, 0)) for sym in poscar_el_seq])

    # 包装系统综述对象，传给前端 UI 展示
    system_summary = {
        "elements": "、".join(poscar_el_seq),
        "u_summary": u_summary_text
    }

    is_user_ldau = parse_vasp_bool(user_incar.get("LDAU", False))
    is_user_spin = parse_vasp_bool(user_incar.get("ISPIN", False)) or str(user_incar.get("ISPIN", "")) in ["2", "2.0"]
    is_soc = parse_vasp_bool(user_incar.get("LSORBIT", False))
    ibrion_val = clean_val(user_incar.get("IBRION", "未设置"))
    nsw_val = clean_val(user_incar.get("NSW", "未设置"))

    analysis_results = []
    top_warnings = []
    all_tags = set(user_incar.keys()).union(set(expert_incar.keys()))
    
    if needs_u_physical: all_tags.add("LDAU")
    if is_user_ldau: all_tags.update(["LDAUU", "LDAUL", "LMAXMIX"])
    if "LDAUU" in user_incar: all_tags.add("LDAUU")
    if "LDAUL" in user_incar: all_tags.add("LDAUL")

    if needs_mag_physical: all_tags.add("ISPIN")
    if is_user_spin: all_tags.update(["MAGMOM", "LMAXMIX"])
    if "MAGMOM" in user_incar: all_tags.add("MAGMOM")
        
    if is_probably_2d: all_tags.add("IDIPOL")
    if "LORBIT" in user_incar or "DOS" in calc_type: all_tags.add("RWIGS")

    for tag in all_tags:
        user_val = clean_val(user_incar.get(tag))
        expert_val = clean_val(expert_incar.get(tag))
        
        kbase = INTEGRATED_PARAMS.get(tag, {})
        if kbase:
            desc_text = f"**{kbase.get('chinese_name', tag)}**：{kbase.get('physical_meaning', '')}"
            if kbase.get('recommendation'): desc_text += f"<br>💡 <i>建议</i>：{kbase.get('recommendation')}"
            if kbase.get('warnings'): desc_text += f"<br>⚠️ <i>避坑</i>：{'；'.join(kbase.get('warnings'))}"
        else:
            desc_text = "VASP高级/偏僻参数，详情请查阅官方手册。"
            
        advice = "✅ 设置正常"
        
        # ----- DFT+U 层级防呆 -----
        if tag == "LDAU":
            if needs_u_physical:
                if not is_user_ldau:
                    advice = f"🚨 核心总开关缺失: 体系包含强关联原子({u_summary_text})，强烈建议开启总开关 LDAU=.TRUE.！"
                    top_warnings.append(advice)
                else:
                    advice = "✅ 总开关已开启，正在进一步校验子参数..."
            else:
                if is_user_ldau:
                    advice = "⚠️ 提示: 体系物理上无需加U，但您强制开启了 LDAU=.TRUE.，请确保您的计算意图。"
                else:
                    advice = "✅ 体系无需 DFT+U 计算 (无强关联电子)，保持关闭即可。"
                
        elif tag == "LDAUU":
            if is_user_ldau:
                if user_val == "未设置" or not any(float(x) > 0 for x in user_val.split() if x.replace('.','',1).lstrip('-').isdigit()):
                    advice = f"🚨 致命缺失: 已开启总开关 LDAU=.TRUE.，必须配套设置非零子参数 LDAUU！推荐: **{final_rec_UU}**"
                    top_warnings.append(advice)
                else:
                    advice = "✅ U值已设置。"
            else:
                if user_val != "未设置":
                    advice = "🚨 越级无效设置: 您设置了子参数 LDAUU，但前置总开关 LDAU 未开启！VASP 会直接无视您的 U 值。"
                    top_warnings.append(advice)
                else:
                    advice = "✅ 未开启总开关，无需设置此项。"

        elif tag == "LDAUL":
            if is_user_ldau:
                if user_val == "未设置":
                    advice = f"🚨 致命缺失: 已开启总开关 LDAU=.TRUE.，必须配套设置作用轨道 LDAUL！推荐: **{final_rec_UL}**"
                    top_warnings.append(advice)
                else:
                    advice = "✅ 作用轨道已设置。"
            else:
                if user_val != "未设置":
                    advice = "🚨 越级无效设置: 您设置了子参数 LDAUL，但前置总开关 LDAU 未开启！VASP 会直接无视。"
                    top_warnings.append(advice)
                else:
                    advice = "✅ 未开启总开关，无需设置此项。"

        # ----- 磁性层级防呆 -----
        elif tag == "ISPIN":
            if needs_mag_physical:
                if not is_user_spin:
                    advice = f"🚨 自旋总开关关闭: 体系含有明显磁性元素，必须开启自旋总开关 ISPIN=2！否则极易算错基态。"
                    top_warnings.append(advice)
                else:
                    advice = "✅ 自旋总开关已开启，正在进一步校验磁矩..."
            else:
                if is_user_spin:
                    advice = "⚠️ 提示: 体系无明显磁性元素，但您开启了 ISPIN=2，计算量将翻倍。"
                else:
                    advice = "✅ 体系无需自旋极化，保持关闭即可。"

        elif tag == "MAGMOM":
            if is_user_spin:
                if user_val == "未设置":
                    advice = f"🚨 磁矩缺失: 已开启总开关 ISPIN=2，必须为其配套赋予初始磁矩 MAGMOM！推荐: **{final_rec_mag}**"
                    top_warnings.append(advice)
                else:
                    advice = "✅ 初始磁矩已设置。"
            else:
                if user_val != "未设置":
                    advice = "🚨 越级无效设置: 您写了子参数 MAGMOM，但自旋开关 ISPIN 被关闭！VASP 会直接忽略您的磁矩设置。"
                    top_warnings.append(advice)
                else:
                    advice = "✅ 未开启 ISPIN，无需设置磁矩。"

        # ----- 其他防呆 -----
        elif tag == "LMAXMIX":
            reasons = []
            if is_user_spin: reasons.append("自旋(ISPIN=2)")
            if is_user_ldau: reasons.append("加U(LDAU=.TRUE.)")
            if is_soc: reasons.append("自旋轨道耦合(LSORBIT)")
            
            if reasons:
                req_val = 6 if "3" in final_rec_UL else 4
                if user_val == "未设置" or int(float(user_val)) < req_val:
                    advice = f"🚨 收敛陷阱: 因开启了 {'、'.join(reasons)}，必须手动设置 LMAXMIX={req_val} 恢复高阶电荷，否则极难收敛！"
                    top_warnings.append(advice)
                else:
                    advice = "✅ LMAXMIX 高级截断要求已满足。"
            else:
                if user_val != "未设置":
                    advice = "ℹ️ 提示: 未开启磁性或加U，通常无需手动设置 LMAXMIX。"
                else:
                    advice = "✅ 无特殊物理需求，无需设置。"

        elif tag == "NSW":
            if user_val != "未设置" and int(user_val) > 0 and ibrion_val in ["-1", "未设置"]:
                advice = "🚨 逻辑冲突: NSW>0 (要求结构弛豫)，但 IBRION=-1 (不准移动原子)，VASP 将直接报错停机！"
                top_warnings.append(advice)
                
        elif tag == "ISIF":
            if str(user_val) == "3" and is_probably_2d:
                advice = "🚨 毁灭性错误: 检测到大真空层 Slab 模型。绝对不能用 ISIF=3！变晶胞会导致真空层被压没，必须改 2 或 4。"
                top_warnings.append(advice)
                
        elif tag == "ISMEAR":
            if str(user_val) == "-5" and nsw_val != "未设置" and int(nsw_val) > 0:
                advice = "🚨 物理错误: 结构弛豫(NSW>0)【绝对不能】用 ISMEAR=-5 (四面体法)，会导致受力算错！请改为 0 或 1。"
                top_warnings.append(advice)

        elif tag == "IDIPOL":
            if is_probably_2d and user_val == "未设置":
                advice = "⚠️ 偶极校正提示: 体系包含真空层。如存在极性面或单面吸附，务必开启 IDIPOL=3 配合 LDIPOL=.TRUE. 防能级倾斜。"
                top_warnings.append(advice)

        elif tag == "RWIGS":
            if "LORBIT" in user_incar and int(user_incar.get("LORBIT", 10)) < 10 and user_val == "未设置":
                advice = "⚠️ 半径缺失: LORBIT < 10 算 DOS 时，必须手动设置 RWIGS 原子半径数组！强烈建议直接改用 LORBIT = 11。"
                top_warnings.append(advice)

        if advice == "✅ 设置正常":
            if user_val == "未设置":
                if expert_val != "未设置" and expert_val not in ["0", "0.0", "False", ""]:
                    advice = f"ℹ️ 未设置，使用VASP默认值。(高通量推荐: {expert_val})"
            elif expert_val != "未设置" and user_val != expert_val:
                advice = f"ℹ️ 提示: 您设置为 {user_val}，材料库参考推荐为 {expert_val}。"

        analysis_results.append({
            "参数标签 (Tag)": f"**{tag}**",
            "您的设置": user_val,
            "专家库推荐": expert_val,
            "专家诊断与建议": advice,
            "内置百科": desc_text
        })
        
    df = pd.DataFrame(analysis_results)
    df['优先级'] = df['专家诊断与建议'].apply(
        lambda x: 0 if "🚨" in x else (1 if "⚠️" in x else (2 if "✅" in x else 3))
    )
    df = df.sort_values(by=['优先级', '参数标签 (Tag)']).drop(columns=['优先级']).reset_index(drop=True)
    top_warnings = list(dict.fromkeys(top_warnings))
    
    return df, top_warnings, final_rec_UU, final_rec_UL, final_rec_mag, needs_u_physical, needs_mag_physical, system_summary

# ==========================================
# 网页前端渲染模块
# ==========================================
def render_html_table(df):
    df_html = df.copy()
    for col in df_html.columns:
        df_html[col] = df_html[col].astype(str).apply(lambda x: re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', x))
        
    raw_html = df_html.to_html(escape=False, index=False)
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
    combined_html = (style + raw_html).replace('\n', '')
    return combined_html.replace('<table border="1" class="dataframe">', '<table class="vasp-table">')

# ==========================================
# UI 交互逻辑
# ==========================================
st.title("🔬 VASP INCAR 审查")
st.markdown("> **底层引擎**：材料库高通量物理规则 + **全量 250+ 参数百科**  ")

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
        
        with st.spinner("🧠 正在执行全量参数库比对与物理层级校验..."):
            df_result, top_warnings, final_rec_UU, final_rec_UL, final_rec_mag, needs_u, needs_mag, sys_summary = analyze_incar(user_incar, user_poscar, calc_type)
        
        # -----------------------------
        # 高清展示系统构成信息 (直击痛点)
        # -----------------------------
        st.info(f"""
        #### 📋 体系物理特征
        - **⚙️ 推断计算类型**：`{calc_type}`
        - **🧪 体系元素组成**：`{sys_summary['elements']}`
        - **🧲 DFT+U 需求判定**：{sys_summary['u_summary']}
        """)

        if top_warnings:
            st.markdown("### ⚠️ 核心报错速览")
            for warn in top_warnings:
                display_warn = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', warn).replace("<br>", " ")
                if "🚨" in warn: st.error(display_warn, icon="🚨")
                else: st.warning(display_warn, icon="⚠️")
            st.markdown("---")
        
        st.subheader("📊 INCAR 深度审查与参数百科")
        st.markdown(render_html_table(df_result), unsafe_allow_html=True)
        
        st.subheader("📥 智能纠错与补全：下载优化版 INCAR")
        st.markdown("系统已保留您原有合理设置，并**在确保已开启总开关的前提下**，自动为您补全缺失的安全截断能、磁矩及U值参数。")
        
        expert_class = MPRelaxSet if "Relaxation" in calc_type else MPStaticSet
        perfect_incar = Incar(user_incar)
        expert_incar_data = expert_class(user_poscar.structure).incar
        
        if "ENCUT" not in perfect_incar and "ENCUT" in expert_incar_data:
            perfect_incar["ENCUT"] = expert_incar_data["ENCUT"]
            
        if needs_u:
            perfect_incar["LDAU"] = ".TRUE."
            perfect_incar["LDAUTYPE"] = 2
            perfect_incar["LDAUL"] = final_rec_UL
            perfect_incar["LDAUU"] = final_rec_UU
            perfect_incar["LDAUJ"] = " ".join(["0"] * len(sys_summary['elements'].split('、')))
            perfect_incar["LMAXMIX"] = 6 if "3" in final_rec_UL else 4
            
        if needs_mag:
            perfect_incar["ISPIN"] = 2
            if "MAGMOM" not in perfect_incar and final_rec_mag:
                perfect_incar["MAGMOM"] = final_rec_mag
            perfect_incar["LMAXMIX"] = 6 if "3" in final_rec_UL else 4
                
        st.download_button(
            label="🔽 下载修复补全版 INCAR",
            data=str(perfect_incar),
            file_name="INCAR_Optimized",
            mime="text/plain",
            type="primary"
        )
        
    except Exception as e:
        st.error(f"❌ 解析异常！请检查文件是否为标准 VASP 格式。\n\n报错详情: {str(e)}")
