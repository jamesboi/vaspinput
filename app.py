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
# 元素磁性与 DFT+U 知识库 (精准提取)
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
    'Ti': {'d': 3.5, 'f': 0}, 'V': {'d': 3.5, 'f': 0}, 'Cr': {'d': 3.5, 'f': 0},
    'Mn': {'d': 3.5, 'f': 0}, 'Fe': {'d': 3.5, 'f': 0}, 'Co': {'d': 3.5, 'f': 0},
    'Ni': {'d': 4.0, 'f': 0}, 'Cu': {'d': 4.0, 'f': 0},
    'Y': {'d': 0, 'f': 0}, 'Zr': {'d': 0, 'f': 0}, 'Nb': {'d': 0, 'f': 0},
    'Mo': {'d': 0, 'f': 0}, 'Tc': {'d': 0, 'f': 0}, 'Ru': {'d': 0, 'f': 0},
    'Rh': {'d': 0, 'f': 0}, 'Pd': {'d': 0, 'f': 0}, 'Ag': {'d': 0, 'f': 0},
    'Hf': {'d': 0, 'f': 0}, 'Ta': {'d': 0, 'f': 0}, 'W': {'d': 0, 'f': 0},
    'Re': {'d': 0, 'f': 0}, 'Os': {'d': 0, 'f': 0}, 'Ir': {'d': 0, 'f': 0},
    'Pt': {'d': 0, 'f': 0}, 'Au': {'d': 0, 'f': 0},
    'La': {'d': 0, 'f': 6.0}, 'Ce': {'d': 0, 'f': 5.0}, 'Pr': {'d': 0, 'f': 5.0},
    'Nd': {'d': 0, 'f': 5.0}, 'Pm': {'d': 0, 'f': 5.0}, 'Sm': {'d': 0, 'f': 5.0},
    'Eu': {'d': 0, 'f': 6.0}, 'Gd': {'d': 0, 'f': 6.0}, 'Tb': {'d': 0, 'f': 5.0},
    'Dy': {'d': 0, 'f': 5.0}, 'Ho': {'d': 0, 'f': 5.0}, 'Er': {'d': 0, 'f': 5.0},
    'Tm': {'d': 0, 'f': 5.0}, 'Yb': {'d': 0, 'f': 4.0}, 'Lu': {'d': 0, 'f': 0},
    'U': {'d': 0, 'f': 4.0}, 'Np': {'d': 0, 'f': 4.0}, 'Pu': {'d': 0, 'f': 4.0}, 'Am': {'d': 0, 'f': 4.0}
}

# ==========================================
# 全量无删减 VASP 参数百科 (250+个)
# ==========================================
INCAR_PARAMS_RAW = {
    'SYSTEM': {'chinese_name': '系统名称', 'description': '计算的系统名称或注释', 'recommendation': '建议设置，描述计算内容', 'warnings':[]},
    'ISTART': {'chinese_name': '波函数初始化', 'description': '波函数初始化选项：0=从头开始，1=读取WAVECAR/CHGCAR', 'recommendation': '首次用0；续算用1', 'warnings': ['续算需确保WAVECAR存在']},
    'ICHARG': {'chinese_name': '电荷密度初始化', 'description': '电荷密度初始化：0=原子叠加，1=读取CHGCAR，2=自洽，11=固定电荷', 'recommendation': '自洽用2；能带用11', 'warnings':['能带/DOS计算必用11跳过自洽']},
    'ENCUT': {'chinese_name': '平面波截断能', 'description': '平面波基组的动能截断值 (eV)', 'recommendation': '设为最大ENMAX的1.0-1.3倍；变体积需1.3倍', 'warnings':['过低结果不可靠；过高增加成本']},
    'PREC': {'chinese_name': '计算精度', 'description': '精度控制：影响FFT网格、基底等', 'recommendation': '高精度用Accurate；常规用Normal', 'warnings': ['高精度避免用Low']},
    'EDIFF': {'chinese_name': '电子收敛标准', 'description': '电子自洽收敛标准 (eV)', 'recommendation': '常规1E-5；高精度1E-6', 'warnings': ['金属收敛难需配合展宽']},
    'EDIFFG': {'chinese_name': '离子收敛标准', 'description': '几何优化收敛判据。正=能量，负=力', 'recommendation': '力收敛: -1E-2 到 -1E-3 eV/Å', 'warnings': ['能量收敛不可靠，负值力收敛更好']},
    'IALGO': {'chinese_name': '电子算法', 'description': '子空间旋转的算法', 'recommendation': '38=Davidson(通用)，48=RMM-DIIS', 'warnings':['RMM-DIIS可能不收敛']},
    'ISMEAR': {'chinese_name': '占据数展宽', 'description': '能带占据smearing', 'recommendation': '金属1或2，绝缘体0，DOS静态-5', 'warnings': ['结构优化绝对不能用-5，受力会错']},
    'SIGMA': {'chinese_name': '展宽宽度', 'description': 'Smearing 宽度 (eV)', 'recommendation': '金属: 0.1-0.2；绝缘体: 0.05', 'warnings': ['SIGMA过大引入Entropy误差']},
    'ISPIN': {'chinese_name': '自旋极化', 'description': '是否考虑自旋(1=否，2=是)', 'recommendation': '磁性材料必须设2', 'warnings': ['不设2会丢失磁性信息']},
    'MAGMOM': {'chinese_name': '原子磁矩', 'description': '初始原子磁矩猜测 (μB)', 'recommendation': 'Fe=4, Co=3, Ni=2, Mn=5', 'warnings': ['不设可能导致亚稳态']},
    'LSORBIT': {'chinese_name': '自旋轨道耦合', 'description': '包含相对论自旋轨道耦合效应', 'recommendation': '重元素、拓扑计算需设为.TRUE.', 'warnings':['需关闭对称性 ISYM=-1']},
    'ICHIBERN': {'chinese_name': '磁化方向', 'description': '初始磁密方向分布', 'recommendation': '通常用1', 'warnings':[]},
    'LDAU': {'chinese_name': 'DFT+U开关', 'description': '是否启用DFT+U', 'recommendation': '强关联体系必须.TRUE.', 'warnings': ['不加U会导致强关联带隙偏小']},
    'LDAUTYPE': {'chinese_name': 'DFT+U方法', 'description': 'DFT+U类型', 'recommendation': '2=Dudarev(最常用)', 'warnings':[]},
    'LDAUL': {'chinese_name': 'U值作用轨道', 'description': '应用U的角动量(2=d, 3=f, -1=无)', 'recommendation': '过渡金属用2；稀土用3', 'warnings': ['需与POSCAR顺序对应']},
    'LDAUU': {'chinese_name': 'U值', 'description': '元素的U值 (eV)', 'recommendation': 'Fe=5.3, Ni=6.2等', 'warnings':[]},
    'LDAUJ': {'chinese_name': 'J值', 'description': 'Hund交换参数J', 'recommendation': 'Dudarev方法设为0', 'warnings':[]},
    'LDAUPRINT': {'chinese_name': 'DFT+U输出', 'description': 'DFT+U输出控制', 'recommendation': '正常计算用0', 'warnings':[]},
    'GGA': {'chinese_name': 'GGA泛函', 'description': 'GGA泛函类型', 'recommendation': 'PE=PBE(默认)；PS=PW91', 'warnings':[]},
    'METAGGA': {'chinese_name': 'Meta-GGA泛函', 'description': 'meta-GGA泛函', 'recommendation': 'SCAN；TPSS', 'warnings':[]},
    'LHFCALC': {'chinese_name': 'HF混合', 'description': 'Hartree-Fock混合开关', 'recommendation': 'HSE06计算设为.TRUE.', 'warnings': ['计算量大，需配合ALGO=Damped']},
    'AEXX': {'chinese_name': 'HF交换比例', 'description': 'HF交换比例', 'recommendation': 'HSE06: 0.25', 'warnings':[]},
    'HFSCREEN': {'chinese_name': 'HF屏蔽参数', 'description': 'HSE短程/长程分离', 'recommendation': 'HSE06: 0.207', 'warnings':[]},
    'IBRION': {'chinese_name': '离子优化算法', 'description': '原子位置优化算法', 'recommendation': '-1=静态；2=共轭梯度(常用)；8=声子', 'warnings':['弛豫时绝对不能用-1']},
    'ISIF': {'chinese_name': '优化自由度', 'description': '控制哪些自由度允许改变', 'recommendation': '2=仅原子位置；3=全优化', 'warnings':['二维材料和表面大真空绝对不能用ISIF=3']},
    'NSW': {'chinese_name': '最大离子步数', 'description': '离子弛豫或分子动力学的步数上限', 'recommendation': '静态0；弛豫100-300', 'warnings':['NSW=0且IBRION不为-1会冲突']},
    'ISYM': {'chinese_name': '对称性', 'description': '对称性开关', 'recommendation': '默认1；杂化推荐3', 'warnings': ['计算SOC时必须设为-1或0']},
    'KPOINTS': {'chinese_name': 'K点设置', 'description': 'k点网格文件', 'recommendation': '金属需稠密K点', 'warnings':[]},
    'LWAVE': {'chinese_name': '波函数输出', 'description': '是否输出波函数', 'recommendation': '续算用.TRUE.', 'warnings':[]},
    'LCHARG': {'chinese_name': '电荷密度输出', 'description': '是否输出电荷密度', 'recommendation': '算能带/DOS前需.TRUE.', 'warnings':[]},
    'LVTOT': {'chinese_name': '静电势输出', 'description': '是否输出静电势', 'recommendation': '分析功函数时用.TRUE.', 'warnings':[]},
    'NELECT': {'chinese_name': '电子总数', 'description': '强制设置体系总电子数', 'recommendation': '带电缺陷时设置', 'warnings':['设错会导致结果错误']},
    'SMASS': {'chinese_name': '热浴参数', 'description': 'MD热浴参数', 'recommendation': '-3=NVT；-1=NVE', 'warnings':['IBRION=0才生效']},
    'TEBEG': {'chinese_name': '初始温度', 'description': 'MD初始温度(K)', 'recommendation': '300 (室温)', 'warnings':[]},
    'TEEND': {'chinese_name': '结束温度', 'description': 'MD结束温度(K)', 'recommendation': '恒温设同TEBEG', 'warnings':[]},
    'IMAGES': {'chinese_name': '中间图像数', 'description': 'NEB中间图像数', 'recommendation': '简单反应3-5', 'warnings': ['配合IBRION=3']},
    'NELMIN': {'chinese_name': '最小电子迭代', 'description': '最小电子迭代次数', 'recommendation': '通常2-6', 'warnings':[]},
    'NPAR': {'chinese_name': '并行参数', 'description': '并行化参数', 'recommendation': '1或NCORE的约数', 'warnings':[]},
    'NCORE': {'chinese_name': '并行能带数', 'description': '共享轨道的CPU核数', 'recommendation': '1-16，根据核数调整', 'warnings': ['与NPAR二选一']},
    'AMIX': {'chinese_name': '电荷混合参数', 'description': '电荷线性混合比例', 'recommendation': '难收敛体系降至0.1', 'warnings': ['过大可能震荡']},
    'BMIX': {'chinese_name': 'Kerker衰减', 'description': '防止长波震荡', 'recommendation': '默认1.0', 'warnings':[]},
    'AMIX_MAG': {'chinese_name': '磁性混合参数', 'description': '磁性体系混合', 'recommendation': '难收敛可增至1.6', 'warnings':[]},
    'BMIX_MAG': {'chinese_name': '磁性Kerker', 'description': '自旋通道衰减', 'recommendation': '默认1.0', 'warnings':[]},
    'MAXMIX': {'chinese_name': '最大迭代历史', 'description': 'Broyden历史', 'recommendation': '难收敛增大到40', 'warnings':[]},
    'LREAL': {'chinese_name': '实空间投影', 'description': '实空间投影开关', 'recommendation': '原子>20可用Auto；高精度须False', 'warnings':[]},
    'VOSKOWN': {'chinese_name': 'VWN插值', 'description': 'VWN插值开关', 'recommendation': 'LDA设1', 'warnings':[]},
    'NWRITE': {'chinese_name': '写入频率', 'description': 'OUTCAR写入频率', 'recommendation': '2(详细)', 'warnings':[]},
    'INIWAV': {'chinese_name': '初始波函数', 'description': '初始波函数生成', 'recommendation': '1=原子叠加', 'warnings':[]},
    'ADDGRID': {'chinese_name': '额外FFT', 'description': '额外FFT网格', 'recommendation': '高精度建议.TRUE.', 'warnings':[]},
    'LSCALAPACK': {'chinese_name': 'ScaLAPACK', 'description': 'ScaLAPACK对角化', 'recommendation': '大体系开启', 'warnings':[]},
    'POTIM': {'chinese_name': '时间步长', 'description': '离子/MD步长', 'recommendation': 'MD用1.0；弛豫炸飞用0.1', 'warnings':[]},
    'RWIGS': {'chinese_name': 'Wigner-Seitz半径', 'description': '原子Wigner-Seitz半径', 'recommendation': '共价半径的50-70%', 'warnings':['不设可能导致DOS投影错误']},
    'RIMPODATA': {'chinese_name': '离子半径', 'description': '分析用离子半径', 'recommendation': '保持默认', 'warnings':[]},
    'NBLOCK': {'chinese_name': '写入间隔', 'description': 'CHGCAR写入间隔', 'recommendation': 'MD可增大', 'warnings':[]},
    'KBLOCK': {'chinese_name': '波函数块写入', 'description': 'WAVECAR块写入', 'recommendation': 'MD可增大', 'warnings':[]},
    'LELF': {'chinese_name': '电子局域化', 'description': '计算ELF', 'recommendation': '分析化学键设.TRUE.', 'warnings':[]},
    'LVHAR': {'chinese_name': 'Hartree势', 'description': '输出Hartree势', 'recommendation': '默认.FALSE.', 'warnings':[]},
    'LORBIT': {'chinese_name': '局域态密度', 'description': '输出分波态密度', 'recommendation': 'DOS强烈建议设为11输出PROCAR', 'warnings':[]},
    'NEDOS': {'chinese_name': 'DOS点数', 'description': 'DOS能量点数', 'recommendation': '画图用 1000-3000', 'warnings':[]},
    'EMAX': {'chinese_name': '能量上限', 'description': 'DOS上限', 'recommendation': '自动设置', 'warnings':[]},
    'WEIMIN': {'chinese_name': '权重最小值', 'description': '防数值问题', 'recommendation': '难收敛减小至0.0001', 'warnings':[]},
    'EBREAK': {'chinese_name': '电子收敛阈值', 'description': '电子步能量标准', 'recommendation': '自动设置', 'warnings':[]},
    'SYMPREC': {'chinese_name': '对称精度', 'description': '对称性容差', 'recommendation': '稍有形变改1E-6', 'warnings':[]},
    'SPRING': {'chinese_name': '弹簧常数', 'description': 'NEB弹簧常数', 'recommendation': '-5', 'warnings':[]},
    'LCLIMB': {'chinese_name': '爬坡', 'description': 'CI-NEB爬坡', 'recommendation': '标准NEB设.TRUE.', 'warnings':[]},
    'ALGO': {'chinese_name': '电子算法', 'description': '宏观算法', 'recommendation': 'Normal；Fast；Damped(杂化必备)', 'warnings':[]},
    'NBANDS': {'chinese_name': '能带数', 'description': '包含的能带数量', 'recommendation': '光学计算需增加', 'warnings':[]},
    'KSPACING': {'chinese_name': 'K点间距', 'description': '最大K点间距', 'recommendation': '半导体0.2; 金属0.15', 'warnings':[]},
    'KGAMMA': {'chinese_name': 'Gamma点', 'description': '包含Gamma', 'recommendation': '.TRUE.', 'warnings':[]},
    'NKRED': {'chinese_name': 'K点缩减', 'description': '自旋极化缩减', 'recommendation': '1', 'warnings':[]},
    'NLSPLINE': {'chinese_name': 'K点插值', 'description': '样条插值', 'recommendation': '.FALSE.', 'warnings':[]},
    'IVDW': {'chinese_name': '范德华校正', 'description': '添加色散力', 'recommendation': '11=D3; 12=D3(BJ)', 'warnings': ['层状材料必开']},
    'VDW_S6': {'chinese_name': 'D3缩放', 'description': 'D3整体缩放', 'recommendation': '默认', 'warnings':[]},
    'VDW_SR': {'chinese_name': 'D3短程', 'description': 'D3短程缩放', 'recommendation': '默认', 'warnings':[]},
    'VDW_A1': {'chinese_name': 'D3_A1', 'description': 'D3_A1参数', 'recommendation': '默认', 'warnings':[]},
    'VDW_A2': {'chinese_name': 'D3_A2', 'description': 'D3_A2参数', 'recommendation': '默认', 'warnings':[]},
    'VDW_RADIUS': {'chinese_name': 'vdW截断', 'description': 'vdW截断半径', 'recommendation': '默认', 'warnings':[]},
    'LUSE_VDW': {'chinese_name': '使用vdW', 'description': 'MBD方法', 'recommendation': '.FALSE.', 'warnings':[]},
    'ENAUG': {'chinese_name': '增强截断能', 'description': 'PAW增强平面波', 'recommendation': '1.5*ENCUT', 'warnings':[]},
    'ENCUTFOCK': {'chinese_name': 'FOCK截断能', 'description': '精确交换截断', 'recommendation': '等同ENCUT', 'warnings':[]},
    'ROPT': {'chinese_name': '投影精度', 'description': '投影参数', 'recommendation': '-1E-3', 'warnings':[]},
    'LASPH': {'chinese_name': '非球面校正', 'description': '非球面电荷密度', 'recommendation': 'DFT+U建议.TRUE.', 'warnings':[]},
    'LMAXFOCK': {'chinese_name': 'Fock最大L', 'description': 'Fock最大角动量', 'recommendation': '0', 'warnings':[]},
    'LMAXMIX': {'chinese_name': '混合最大L', 'description': '混合密度最大角动量', 'recommendation': '含d且加U必设4', 'warnings': ['不设电荷难收敛']},
    'MDALGO': {'chinese_name': 'MD算法', 'description': '积分算法', 'recommendation': '0(Verlet)', 'warnings':[]},
    'LANGEVIN_GAMMA': {'chinese_name': 'Langevin阻尼', 'description': '阻尼参数', 'recommendation': '默认', 'warnings':[]},
    'PSTRESS': {'chinese_name': '静水压', 'description': '外加静水压', 'recommendation': '默认0', 'warnings':[]},
    'PMASS': {'chinese_name': '离子赝质量', 'description': 'CP-MD赝质量', 'recommendation': '默认', 'warnings':[]},
    'LEPSILON': {'chinese_name': '高频介电', 'description': '高频介电常数', 'recommendation': '光学计算.TRUE.', 'warnings':[]},
    'LOPTICS': {'chinese_name': '光学计算', 'description': '计算光学性质', 'recommendation': '设为.TRUE.并增加NBANDS', 'warnings':[]},
    'CSHIFT': {'chinese_name': '复位移', 'description': '光学展宽', 'recommendation': '金属0.2', 'warnings':[]},
    'CLL': {'chinese_name': 'CL规范', 'description': '规范选择', 'recommendation': '0', 'warnings':[]},
    'ICORELEVEL': {'chinese_name': '芯能级', 'description': '芯能级处理', 'recommendation': '芯空穴设1', 'warnings':[]},
    'ENCUTGW': {'chinese_name': 'GW截断能', 'description': 'GW响应截断', 'recommendation': '等同ENCUT', 'warnings':[]},
    'NOMEGA': {'chinese_name': '频率点数', 'description': 'GW频率积分', 'recommendation': '50', 'warnings':[]},
    'OMEGAMAX': {'chinese_name': '最大频率', 'description': 'GW上限', 'recommendation': '默认', 'warnings':[]},
    'LWANNIER90': {'chinese_name': 'Wannier90', 'description': 'W90接口', 'recommendation': '需要时设.TRUE.', 'warnings':[]},
    'LWANNIER90_RUN': {'chinese_name': 'W90运行', 'description': '内部运行W90', 'recommendation': '.TRUE.', 'warnings':[]},
    'LNONCOLLINEAR': {'chinese_name': '非共线磁性', 'description': '允许非共线', 'recommendation': '复杂磁性设.TRUE.', 'warnings':[]},
    'SAXIS': {'chinese_name': '自旋轴', 'description': '量化方向', 'recommendation': '0 0 1', 'warnings':[]},
    'ICHIBARE': {'chinese_name': '手征密度', 'description': '手征电流', 'recommendation': '1', 'warnings':[]},
    'IDIPOL': {'chinese_name': '偶极校正', 'description': '偶极矩校正方向', 'recommendation': '极性面/功函数设3', 'warnings': ['不对称Slab必须开启']},
    'LORBMOM': {'chinese_name': '轨道矩', 'description': '轨道磁矩输出', 'recommendation': 'SOC计算设1', 'warnings':[]},
    'NUPDOWN': {'chinese_name': '自旋差', 'description': '固定自旋差', 'recommendation': '固定磁矩使用', 'warnings':[]},
    'LCALCPOL': {'chinese_name': '极化输出', 'description': 'Berry相位', 'recommendation': '铁电设.TRUE.', 'warnings':[]},
    'LBERRY': {'chinese_name': 'Berry相', 'description': '极化计算', 'recommendation': '.TRUE.', 'warnings':[]},
    'I_CONSTRAINED_M': {'chinese_name': '磁矩约束', 'description': '磁矩约束', 'recommendation': '1=原子', 'warnings':[]},
    'CONSTRAINED_M': {'chinese_name': '约束强度', 'description': '惩罚势', 'recommendation': '10.0', 'warnings':[]},
    'LAMBDA': {'chinese_name': '拉格朗日', 'description': '约束松紧', 'recommendation': '默认', 'warnings':[]},
    'AGGAX': {'chinese_name': 'GGA交换', 'description': '杂化泛函比例', 'recommendation': '1.0', 'warnings':[]},
    'PHON_NSTRUCT': {'chinese_name': '声子结构', 'description': '声子超胞数', 'recommendation': '-1', 'warnings':[]},
    'IMIX': {'chinese_name': '混合方式', 'description': '混合算法', 'recommendation': '金属4', 'warnings':[]},
    'NELMDL': {'chinese_name': '延迟迭代', 'description': '延迟电荷更新', 'recommendation': '难收敛-5', 'warnings':[]},
    'EFIELD': {'chinese_name': '电场', 'description': '外加电场', 'recommendation': '默认0', 'warnings':[]},
    'EFIELD_PEAD': {'chinese_name': 'PEAD电场', 'description': '矢量电场', 'recommendation': '0 0 0', 'warnings':[]},
    'FERWE': {'chinese_name': 'Fermi面权重', 'description': '权重列表', 'recommendation': '1.0', 'warnings':[]},
    'MAXMEM': {'chinese_name': '最大内存', 'description': '内存上限', 'recommendation': '大内存可增', 'warnings':[]},
    'NSIM': {'chinese_name': '同时迭代', 'description': '能带同时迭代数', 'recommendation': '4', 'warnings':[]},
    'LASYNC': {'chinese_name': '异步IO', 'description': '异步输入输出', 'recommendation': '大体系.TRUE.', 'warnings':[]},
    'GGA_COMPAT': {'chinese_name': 'GGA兼容', 'description': 'VASP4兼容', 'recommendation': '.TRUE.', 'warnings':[]},
    'PRECFOCK': {'chinese_name': 'FOCK精度', 'description': '精确交换精度', 'recommendation': 'Accurate', 'warnings':[]},
    'ENCUTLF': {'chinese_name': 'LF截断', 'description': '局域场截断', 'recommendation': '.FALSE.', 'warnings':[]},
    'DARWINR': {'chinese_name': 'Darwin标量', 'description': '相对论校正', 'recommendation': '.TRUE.', 'warnings':[]},
    'DARWINV': {'chinese_name': 'Darwin矢量', 'description': '矢量校正', 'recommendation': 'SOC设.TRUE.', 'warnings':[]},
    'LSOL': {'chinese_name': '溶剂化', 'description': '隐式溶剂化模型', 'recommendation': '溶液体系.TRUE.', 'warnings': ['需编译VASPsol']},
    'LADDER': {'chinese_name': '能带输出', 'description': '输出数据', 'recommendation': '.FALSE.', 'warnings':[]},
    'LAECHG': {'chinese_name': '全电荷密度', 'description': '全电荷成分', 'recommendation': '.FALSE.', 'warnings':[]},
    'LPARD': {'chinese_name': '投影态密度', 'description': '投影DOS', 'recommendation': '.FALSE.', 'warnings':[]},
    'NBMOD': {'chinese_name': '能带模式', 'description': '计算方式', 'recommendation': '-1', 'warnings':[]},
    'IBAND': {'chinese_name': '能带索引', 'description': '计算索引', 'recommendation': '默认', 'warnings':[]},
    'EINT': {'chinese_name': '能量范围', 'description': '积分范围', 'recommendation': '0.0', 'warnings':[]},
    'DIPOL': {'chinese_name': '偶极中心', 'description': '计算中心', 'recommendation': '0.5 0.5 0.5', 'warnings':[]},
    'AMIN': {'chinese_name': '最小混合', 'description': '最小权重', 'recommendation': '难收敛0.01', 'warnings':[]},
    'LMODELHF': {'chinese_name': '模型HF', 'description': '屏蔽交换', 'recommendation': '.FALSE.', 'warnings':[]},
    'HFLMAX': {'chinese_name': 'HF最大L', 'description': '精确交换最大L', 'recommendation': '-1', 'warnings':[]},
    'HFRCUT': {'chinese_name': 'HF截断', 'description': '截断半径', 'recommendation': '0.0', 'warnings':[]},
    'LRHFCALC': {'chinese_name': '相对论HF', 'description': '相对论', 'recommendation': '重元素.TRUE.', 'warnings':[]},
    'LHFONE': {'chinese_name': '单中心HF', 'description': '单中心', 'recommendation': '.FALSE.', 'warnings':[]},
    'HFSCREENC': {'chinese_name': '屏蔽类型', 'description': '类型开关', 'recommendation': '.FALSE.', 'warnings':[]},
    'CMBJ': {'chinese_name': 'MBJ势', 'description': 'MBJ参数', 'recommendation': '带隙校正', 'warnings':[]},
    'CMBJA': {'chinese_name': 'MBJ参数A', 'description': '参数A', 'recommendation': '0.0', 'warnings':[]},
    'CMBJB': {'chinese_name': 'MBJ参数B', 'description': '参数B', 'recommendation': '1.0', 'warnings':[]},
    'LNICSALL': {'chinese_name': 'NMR位移', 'description': 'NMR位移张量', 'recommendation': 'NMR设.TRUE.', 'warnings':[]},
    'LCHIMAG': {'chinese_name': '化学位移', 'description': '成像', 'recommendation': 'NMR设.TRUE.', 'warnings':[]},
    'LDOWNSAMPLE': {'chinese_name': '降采样', 'description': '减少数据量', 'recommendation': '.FALSE.', 'warnings':[]},
    'ANDERSEN_PROB': {'chinese_name': 'Andersen概率', 'description': '热浴', 'recommendation': '0.0', 'warnings':[]},
    'HILLS_BIN': {'chinese_name': 'Hills采样', 'description': 'Meta-dynamics', 'recommendation': '-1', 'warnings':[]},
    'HILLS_H': {'chinese_name': 'Hills高度', 'description': '高度', 'recommendation': '0.01', 'warnings':[]},
    'HILLS_W': {'chinese_name': 'Hills宽度', 'description': '宽度', 'recommendation': '0.05', 'warnings':[]},
    'APACO': {'chinese_name': '层间距', 'description': '约束距离', 'recommendation': '0.0', 'warnings':[]},
    'NPACO': {'chinese_name': 'PACO点数', 'description': '采样点', 'recommendation': '256', 'warnings':[]},
    'TIME': {'chinese_name': '时间参数', 'description': '时间', 'recommendation': '自动', 'warnings':[]},
    'STEP_MAX': {'chinese_name': '最大步长', 'description': 'MD离子步长', 'recommendation': '自动', 'warnings':[]},
    'STEP_SIZE': {'chinese_name': '步长', 'description': '积分步长', 'recommendation': '自动', 'warnings':[]},
    'MINROT': {'chinese_name': '最小旋转', 'description': '角度', 'recommendation': '0.0', 'warnings':[]},
    'MIXFIRST': {'chinese_name': '先混合', 'description': '先混合后MD', 'recommendation': '难收敛.TRUE.', 'warnings':[]},
    'ANORTH': {'chinese_name': '非正交盒', 'description': '非正交', 'recommendation': '0.0', 'warnings':[]},
    'LATTICE_CONSTRAINTS': {'chinese_name': '晶格约束', 'description': '约束列表', 'recommendation': '0 0 0 0 0 0', 'warnings':[]},
    'QSPIRAL': {'chinese_name': '螺旋q矢量', 'description': '波矢', 'recommendation': '0 0 0', 'warnings':[]},
    'LANGEVIN_GAMMA_L': {'chinese_name': '晶格阻尼', 'description': 'Langevin阻尼', 'recommendation': '1.0', 'warnings':[]},
    'SCSRAD': {'chinese_name': 'SCS半径', 'description': '自洽屏蔽半径', 'recommendation': '0.0', 'warnings':[]},
    'TSUBSYS': {'chinese_name': '热浴', 'description': '系统', 'recommendation': '1 1', 'warnings':[]},
    'VCUTOFF': {'chinese_name': '截断速度', 'description': 'MD速度截断', 'recommendation': '0.0', 'warnings':[]},
    'OFIELD_A': {'chinese_name': '有序场A', 'description': '参数A', 'recommendation': '0.0', 'warnings':[]},
    'OFIELD_KAPPA': {'chinese_name': '有序场kappa', 'description': '曲率', 'recommendation': '0.0', 'warnings':[]},
    'OFIELD_Q6_FAR': {'chinese_name': 'Q6远场', 'description': '八极', 'recommendation': '0.0', 'warnings':[]},
    'OFIELD_Q6_NEAR': {'chinese_name': 'Q6近场', 'description': '贡献', 'recommendation': '0.0', 'warnings':[]},
    'LEFG': {'chinese_name': 'EFG', 'description': '电场梯度', 'recommendation': 'NMR设.TRUE.', 'warnings':[]},
    'QUAD_EFG': {'chinese_name': '四极矩', 'description': '核四极矩', 'recommendation': '默认', 'warnings':[]},
    'RANDOM_SEED': {'chinese_name': '随机种子', 'description': '随机数种子', 'recommendation': '自动', 'warnings':[]},
    'PARAM1': {'chinese_name': '参数1', 'description': '额外参数', 'recommendation': '0.0', 'warnings':[]},
    'PARAM2': {'chinese_name': '参数2', 'description': '额外参数', 'recommendation': '0.0', 'warnings':[]},
    'LGAUGE': {'chinese_name': '规范固定', 'description': '固定', 'recommendation': '.FALSE.', 'warnings':[]},
    'LRPAFORCE': {'chinese_name': 'RPA力', 'description': '计算力', 'recommendation': 'RPA设.TRUE.', 'warnings':[]},
    'LFXC': {'chinese_name': 'FXC', 'description': '计算Fxc', 'recommendation': '.FALSE.', 'warnings':[]},
    'LTCTE': {'chinese_name': 'TCTE', 'description': '总能', 'recommendation': '.FALSE.', 'warnings':[]},
    'LTETE': {'chinese_name': 'TETE', 'description': '四极', 'recommendation': '.FALSE.', 'warnings':[]},
    'LTRIPLET': {'chinese_name': '三态', 'description': '三重态', 'recommendation': '.FALSE.', 'warnings':[]},
    'LUSEW': {'chinese_name': 'USEW', 'description': 'W矩阵', 'recommendation': '.FALSE.', 'warnings':[]},
    'NUCIND': {'chinese_name': '核独立', 'description': '独立优化', 'recommendation': '.FALSE.', 'warnings':[]},
    'NTAUPAR': {'chinese_name': '时间并行', 'description': 'MD时间', 'recommendation': '1', 'warnings':[]},
    'NTARGET_STATES': {'chinese_name': '目标态', 'description': '指定态', 'recommendation': '0', 'warnings':[]},
    'LOCPROJ': {'chinese_name': '局域投影', 'description': '投影轨道', 'recommendation': '0', 'warnings':[]},
    'POMASS': {'chinese_name': '离子质量', 'description': '各元素质量', 'recommendation': '自动', 'warnings':[]},
    'PROUTINE': {'chinese_name': '打印程序', 'description': '索引', 'recommendation': '0', 'warnings':[]},
    'PTHRESHOLD': {'chinese_name': '打印阈值', 'description': '阈值', 'recommendation': '1E-4', 'warnings':[]},
    'LMUSIC': {'chinese_name': 'MUSIC', 'description': '接口', 'recommendation': '.FALSE.', 'warnings':[]},
    'SHIFTRED': {'chinese_name': '偏移缩减', 'description': '缩减量', 'recommendation': '.FALSE.', 'warnings':[]},
    'NKREDX': {'chinese_name': 'X向K缩减', 'description': '各向异性', 'recommendation': '1', 'warnings':[]},
    'NKREDY': {'chinese_name': 'Y向K缩减', 'description': '各向异性', 'recommendation': '1', 'warnings':[]},
    'NKREDZ': {'chinese_name': 'Z向K缩减', 'description': '各向异性', 'recommendation': '1', 'warnings':[]},
    'KPOINT_BSE': {'chinese_name': 'BSE k点', 'description': '子集', 'recommendation': '0', 'warnings':[]},
    'KPUSE': {'chinese_name': '使用k点', 'description': '指定k', 'recommendation': '0', 'warnings':[]},
    'EVENONLY': {'chinese_name': '偶k点', 'description': '偶数', 'recommendation': '.FALSE.', 'warnings':[]},
    'EVENONLYGW': {'chinese_name': 'GW偶k点', 'description': '偶数', 'recommendation': '.FALSE.', 'warnings':[]},
    'ODDONLY': {'chinese_name': '奇k点', 'description': '奇数', 'recommendation': '.FALSE.', 'warnings':[]},
    'ODDONLYGW': {'chinese_name': 'GW奇k点', 'description': '奇数', 'recommendation': '.FALSE.', 'warnings':[]},
    'NBANDSGW': {'chinese_name': 'GW能带数', 'description': '数量', 'recommendation': '自动', 'warnings':[]},
    'NBANDSO': {'chinese_name': '占据能带数', 'description': '数量', 'recommendation': '自动', 'warnings':[]},
    'NBANDSV': {'chinese_name': '虚能带数', 'description': '非占据', 'recommendation': '自动', 'warnings':[]},
    'NOMEGAPAR': {'chinese_name': '频率并行', 'description': '并行', 'recommendation': '1', 'warnings':[]},
    'NOMEGAR': {'chinese_name': '实频率点', 'description': '采样', 'recommendation': '0', 'warnings':[]},
    'OMEGAMIN': {'chinese_name': '最小频率', 'description': '下限', 'recommendation': '-1.0', 'warnings':[]},
    'OMEGATL': {'chinese_name': '频率尾参数', 'description': '尾部', 'recommendation': '0.0', 'warnings':[]},
    'SELFENERGY': {'chinese_name': '自能计算', 'description': 'GW自能', 'recommendation': 'GW设.TRUE.', 'warnings':[]},
    'LFERMIGW': {'chinese_name': 'Fermi更新', 'description': '更新', 'recommendation': '.FALSE.', 'warnings':[]},
    'LSINGLES': {'chinese_name': '单粒子', 'description': '近似', 'recommendation': '.FALSE.', 'warnings':[]},
    'ALDA': {'chinese_name': 'ALDA校正', 'description': '校正', 'recommendation': '.FALSE.', 'warnings':[]},
    'ENCUTGWSOFT': {'chinese_name': 'GW软截断', 'description': '能量', 'recommendation': '自动', 'warnings':[]},
    'ENINI': {'chinese_name': '初始能量', 'description': '截断', 'recommendation': '自动', 'warnings':[]},
    'PHON_LBOSE': {'chinese_name': '声子展宽', 'description': '展宽', 'recommendation': '.FALSE.', 'warnings':[]},
    'PHON_LMC': {'chinese_name': '声子MC', 'description': 'MC方法', 'recommendation': '.FALSE.', 'warnings':[]},
    'PHON_NTLIST': {'chinese_name': '声子点', 'description': '温度点', 'recommendation': '0', 'warnings':[]},
    'PHON_TLIST': {'chinese_name': '声子温度', 'description': '参数', 'recommendation': '0.0', 'warnings':[]},
    'WANPROJ': {'chinese_name': 'Wannier投影', 'description': '投影', 'recommendation': 'Wannier设.TRUE.', 'warnings':[]},
    'LWRITE_MMN_AMN': {'chinese_name': '写MMN/AMN', 'description': '重叠', 'recommendation': '.FALSE.', 'warnings':[]},
    'LWRITE_UNK': {'chinese_name': '写UNK', 'description': '波函数', 'recommendation': '.FALSE.', 'warnings':[]},
    'LWRITE_WANPROJ': {'chinese_name': '写投影', 'description': '数据', 'recommendation': '.FALSE.', 'warnings':[]},
    'CH_LSPEC': {'chinese_name': '芯空穴谱', 'description': '谱函数', 'recommendation': '芯空穴设.TRUE.', 'warnings':[]},
    'CH_NEDOS': {'chinese_name': '空穴DOS点', 'description': '点数', 'recommendation': '0', 'warnings':[]},
    'CH_SIGMA': {'chinese_name': '空穴展宽', 'description': '展宽', 'recommendation': '0.1', 'warnings':[]},
    'CLN': {'chinese_name': 'CL规范', 'description': '类型', 'recommendation': '0', 'warnings':[]},
    'CLNT': {'chinese_name': 'CL类型', 'description': '计算', 'recommendation': '0', 'warnings':[]},
    'CLZ': {'chinese_name': 'CL_Z', 'description': '分量', 'recommendation': '0.0', 'warnings':[]},
    'IEPSILON': {'chinese_name': '介电索引', 'description': '模式', 'recommendation': '1', 'warnings':[]},
    'IGPAR': {'chinese_name': '光学方向', 'description': '极化', 'recommendation': '0', 'warnings':[]},
    'IPEAD': {'chinese_name': 'PEAD', 'description': '位移', 'recommendation': '0', 'warnings':[]},
    'LORBITALREAL': {'chinese_name': '实空间轨道', 'description': '投影', 'recommendation': '.FALSE.', 'warnings':[]},
    'NMAXFOCKAE': {'chinese_name': 'AE最大索引', 'description': '索引', 'recommendation': '0', 'warnings':[]},
    'AGGAC': {'chinese_name': 'GGA相关', 'description': 'AC参数', 'recommendation': '0.0', 'warnings':[]},
    'ALDAC': {'chinese_name': 'LDA相关', 'description': '参数', 'recommendation': '0.0', 'warnings':[]},
    'LMIXTAU': {'chinese_name': '自旋混合', 'description': '常数', 'recommendation': '.FALSE.', 'warnings':[]},
    'LNABLA': {'chinese_name': '梯度输出', 'description': '输出', 'recommendation': '.FALSE.', 'warnings':[]},
    'MAGPOS': {'chinese_name': '磁矩位置', 'description': '位置', 'recommendation': '.FALSE.', 'warnings':[]},
    'ORBITALMAG': {'chinese_name': '轨道磁性', 'description': '计算', 'recommendation': 'SOC设.TRUE.', 'warnings':[]},
    'MAGDIPOLOUT': {'chinese_name': '磁偶极输出', 'description': '输出', 'recommendation': '.FALSE.', 'warnings':[]},
    'ISPIND': {'chinese_name': '分立自旋', 'description': '处理', 'recommendation': '1', 'warnings':[]},
    'ICALCEPS': {'chinese_name': '介电开关', 'description': '计算', 'recommendation': '介电计算.TRUE.', 'warnings':[]},
    'FINDIFF': {'chinese_name': '有限差分', 'description': '方法', 'recommendation': '0', 'warnings':[]},
    'DQ': {'chinese_name': '位移增量', 'description': '位移', 'recommendation': '0.005', 'warnings':[]},
    'DEPER': {'chinese_name': '能量步长', 'description': '差', 'recommendation': '0.0', 'warnings':[]},
    'DIMER_DIST': {'chinese_name': '二聚体距离', 'description': '距离', 'recommendation': '0.0', 'warnings':[]},
    'IWAVPR': {'chinese_name': '波函数处理', 'description': '处理', 'recommendation': '1', 'warnings':[]},
    'LCOMPAT': {'chinese_name': '兼容性', 'description': 'VASP4', 'recommendation': '.FALSE.', 'warnings':[]},
    'LCORR': {'chinese_name': '电荷校正', 'description': '平均', 'recommendation': '.TRUE.', 'warnings':[]},
    'LDIAG': {'chinese_name': '对角化', 'description': '子空间', 'recommendation': '.TRUE.', 'warnings':[]},
    'LLRAUG': {'chinese_name': 'LR_AUG', 'description': '平滑', 'recommendation': '.FALSE.', 'warnings':[]},
    'LSYMGRAD': {'chinese_name': '对称梯度', 'description': '加速', 'recommendation': '.FALSE.', 'warnings':[]},
    'VALUE_MAX': {'chinese_name': '最大值', 'description': '约束上限', 'recommendation': '0.0', 'warnings':[]},
    'VALUE_MIN': {'chinese_name': '最小值', 'description': '约束下限', 'recommendation': '0.0', 'warnings':[]},
    'ENMAX': {'chinese_name': '最大ENMAX', 'description': 'POTCAR', 'recommendation': '自动', 'warnings':[]},
    'ENMIN': {'chinese_name': '最小ENMIN', 'description': 'POTCAR', 'recommendation': '自动', 'warnings':[]},
    'ENAVG': {'chinese_name': '平均截断能', 'description': 'POTCAR', 'recommendation': '自动', 'warnings':[]},
    'PFLAT': {'chinese_name': 'PFLAT', 'description': '参数', 'recommendation': '.FALSE.', 'warnings':[]},
    'PSUBSYS': {'chinese_name': '参数子系统', 'description': '系统', 'recommendation': '1', 'warnings':[]},
    'QMAXFOCKAE': {'chinese_name': 'QMAX_AE', 'description': '参数', 'recommendation': '0.0', 'warnings':[]},
    'ZVAL': {'chinese_name': 'ZVAL', 'description': '价电子', 'recommendation': '自动', 'warnings':[]},
    'NBLK': {'chinese_name': '输出块大小', 'description': '运算', 'recommendation': '-1', 'warnings':[]},
    'NCRPA_BANDS': {'chinese_name': 'CRPA能带', 'description': '范围', 'recommendation': '0', 'warnings':[]},
    'NPPSTR': {'chinese_name': '投影方向', 'description': '方向', 'recommendation': '0', 'warnings':[]},
    'NBSEEIG': {'chinese_name': 'BSE本征值', 'description': '数量', 'recommendation': '0', 'warnings':[]},
    'PLEVEL': {'chinese_name': '打印级别', 'description': '级别', 'recommendation': '0', 'warnings':[]},
    'INIMIX': {'chinese_name': '初始混合', 'description': '方式', 'recommendation': '1', 'warnings':[]},
    'MIXPRE': {'chinese_name': '混合预处理', 'description': '参数', 'recommendation': '0', 'warnings':[]},
    'NFREE': {'chinese_name': '有限差分步数', 'description': '步数', 'recommendation': '0', 'warnings':[]},
    'NDAV': {'chinese_name': 'Davidson迭代', 'description': '最大次', 'recommendation': '30', 'warnings':[]},
    'INCREM': {'chinese_name': '增量参数', 'description': '列表', 'recommendation': '0.015', 'warnings':[]},
    'ANTIRES': {'chinese_name': '反共振计算', 'description': '开关', 'recommendation': '0', 'warnings':[]},
    'HITOLER': {'chinese_name': '高精度容差', 'description': '容差', 'recommendation': '1E-5', 'warnings':[]},
    'SHAKEMAXITER': {'chinese_name': 'Shake迭代', 'description': '最大', 'recommendation': '50', 'warnings':[]},
    'SHAKETOL': {'chinese_name': 'Shake容差', 'description': '收敛', 'recommendation': '1E-5', 'warnings':[]},
    'FERDO': {'chinese_name': 'Fermi面积分D', 'description': '方向', 'recommendation': '0.0', 'warnings':[]},
    'EMIN': {'chinese_name': '最小能量', 'description': '下限', 'recommendation': '0.0', 'warnings':[]},
    'EPSILON': {'chinese_name': '介电常数', 'description': '背景', 'recommendation': '1.0', 'warnings':[]},
    'SMEARINGS': {'chinese_name': 'Smearing列表', 'description': '值', 'recommendation': '0.0', 'warnings':[]},
    'LGauss': {'chinese_name': '高斯展宽', 'description': '开关', 'recommendation': '.FALSE.', 'warnings':[]},
    'LVDWEXPANSION': {'chinese_name': 'vdW展开', 'description': '展开', 'recommendation': '.FALSE.', 'warnings':[]},
    'LVDW_EWALD': {'chinese_name': 'vdW Ewald', 'description': '求和', 'recommendation': '.FALSE.', 'warnings':[]},
    'VDW_C6': {'chinese_name': 'C6系数', 'description': '原子C6', 'recommendation': '自动', 'warnings':[]},
    'VDW_R0': {'chinese_name': 'R0半径', 'description': '半径', 'recommendation': '自动', 'warnings':[]},
    'VDW_CNRADIUS': {'chinese_name': '截断半径', 'description': '距离', 'recommendation': '0.0', 'warnings':[]},
    'VDW_D': {'chinese_name': 'D参数', 'description': 'damping', 'recommendation': '0.0', 'warnings':[]},
    'VDW_S8': {'chinese_name': 'S8参数', 'description': '缩放', 'recommendation': '0.0', 'warnings':[]},
    'ZAB_VDW': {'chinese_name': 'vdW半径', 'description': 'Hutson', 'recommendation': '0.0', 'warnings':[]},
    'TAU': {'chinese_name': '温度耦合', 'description': '常数', 'recommendation': '自动', 'warnings':[]},
    'LTEEPS': {'chinese_name': 'EEPS', 'description': '总能', 'recommendation': '.FALSE.', 'warnings':[]},
    'LTHOMAS': {'chinese_name': 'Thomas', 'description': '屏蔽', 'recommendation': '.FALSE.', 'warnings':[]},
    'LFXCEPS': {'chinese_name': 'FXC_EPS', 'description': '介电', 'recommendation': '.FALSE.', 'warnings':[]},
    'LFXHEG': {'chinese_name': 'FXC_HEG', 'description': '均匀电子', 'recommendation': '.FALSE.', 'warnings':[]},
    'LMAGBLOCH': {'chinese_name': '磁性Bloch变换', 'description': '开关', 'recommendation': '.FALSE.', 'warnings':[]},
    'LBLUEOUT': {'chinese_name': 'Bloch校正输出', 'description': '输出', 'recommendation': '.FALSE.', 'warnings':[]},
    'LBONE': {'chinese_name': 'BondOrder输出', 'description': '输出', 'recommendation': '.FALSE.', 'warnings':[]},
    'LCALCEPS': {'chinese_name': '介电常数输出', 'description': '输出', 'recommendation': '.FALSE.', 'warnings':[]},
    'LHARTREE': {'chinese_name': 'Hartree势输出', 'description': '输出', 'recommendation': '.FALSE.', 'warnings':[]},
    'LHYPERFINE': {'chinese_name': '超精细输出', 'description': '相互作用', 'recommendation': '.FALSE.', 'warnings':[]},
    'LPEAD': {'chinese_name': 'PEAD输出', 'description': '分析', 'recommendation': '.FALSE.', 'warnings':[]},
    'LPLANE': {'chinese_name': '平面波输出', 'description': '系数', 'recommendation': '.FALSE.', 'warnings':[]},
    'LRPA': {'chinese_name': 'RPA输出', 'description': '相关', 'recommendation': '.FALSE.', 'warnings':[]},
    'LSCAAWARE': {'chinese_name': 'SCA启用', 'description': '电荷', 'recommendation': '.FALSE.', 'warnings':[]},
    'LSCALU': {'chinese_name': 'LU分解输出', 'description': '分解', 'recommendation': '.FALSE.', 'warnings':[]},
    'LSCSGRAD': {'chinese_name': 'SCS梯度输出', 'description': '梯度', 'recommendation': '.FALSE.', 'warnings':[]},
    'LSELFENERGY': {'chinese_name': '自能输出', 'description': '计算', 'recommendation': '.FALSE.', 'warnings':[]},
    'LSEPB': {'chinese_name': '分离带输出', 'description': '能带', 'recommendation': '.FALSE.', 'warnings':[]},
    'LSEPK': {'chinese_name': '分离k点输出', 'description': 'k点', 'recommendation': '.FALSE.', 'warnings':[]},
    'LSPECTRAL': {'chinese_name': '谱函数输出', 'description': '开关', 'recommendation': '.FALSE.', 'warnings':[]},
    'LSPECTRALGW': {'chinese_name': 'GW谱函数', 'description': '开关', 'recommendation': '.FALSE.', 'warnings':[]},
    'LSPIRAL': {'chinese_name': '螺旋输出', 'description': '结构', 'recommendation': '.FALSE.', 'warnings':[]},
    'LSUBROT': {'chinese_name': '子旋转输出', 'description': '旋转', 'recommendation': '.FALSE.', 'warnings':[]},
    'LZEROZ': {'chinese_name': 'Z方向零点', 'description': '能量', 'recommendation': '.FALSE.', 'warnings':[]},
    'ISEARCH': {'chinese_name': '原子位置搜索', 'description': '搜索算法', 'recommendation': '0', 'warnings':[]},
    'LFOCKAEDFT': {'chinese_name': 'HSE精确交换', 'description': '计算精确交换', 'recommendation': '.FALSE.', 'warnings':[]},
    'LKPROJ': {'chinese_name': 'Wannier投影', 'description': '基组', 'recommendation': '.FALSE.', 'warnings':[]},
    'LMAXFOCKAE': {'chinese_name': 'Fock算符最大L', 'description': '角动量', 'recommendation': '0', 'warnings':[]},
    'LMAXPAW': {'chinese_name': 'PAW投影最大L', 'description': '角动量', 'recommendation': '-1', 'warnings':[]},
    'LMAXTAU': {'chinese_name': '张力计算最大L', 'description': '角动量', 'recommendation': '-1', 'warnings':[]},
    'LMETAGGA': {'chinese_name': 'meta-GGA计算', 'description': '启用', 'recommendation': '.FALSE.', 'warnings':[]},
    'LMONO': {'chinese_name': '单极矩计算', 'description': '计算静电', 'recommendation': '.FALSE.', 'warnings':[]},
    'LNMR_SYM_RED': {'chinese_name': 'NMR对称性约化', 'description': '应用约化', 'recommendation': '.TRUE.', 'warnings':[]},
    'LVEL': {'chinese_name': '速度计算', 'description': '计算原子速度', 'recommendation': '.FALSE.', 'warnings':[]},
    'ML_MODE': {'chinese_name': 'ML训练模式', 'description': '模式', 'recommendation': '0', 'warnings':[]},
    'ML_FF_LMLFF': {'chinese_name': '机器学习力场', 'description': '启用', 'recommendation': '.FALSE.', 'warnings':[]},
    'ML_FF_LMLMB': {'chinese_name': 'ML多体势能面', 'description': '训练', 'recommendation': '.FALSE.', 'warnings':[]},
    'ML_FF_ISTART': {'chinese_name': 'ML初始化模式', 'description': '控制', 'recommendation': '0', 'warnings':[]},
    'ML_FF_MCONF': {'chinese_name': 'ML训练构型数', 'description': '数量', 'recommendation': '1000', 'warnings':[]},
    'ML_FF_MCONF_NEW': {'chinese_name': 'ML新构型数', 'description': '数量', 'recommendation': '50', 'warnings':[]},
    'ML_FF_MHIS': {'chinese_name': 'ML历史步数', 'description': '步数', 'recommendation': '10', 'warnings':[]},
    'ML_FF_LCONF_DISCARD': {'chinese_name': 'ML丢弃低置信度', 'description': '控制', 'recommendation': '.FALSE.', 'warnings':[]},
    'ML_FF_LBASIS_DISCARD': {'chinese_name': 'ML丢弃基组', 'description': '基组', 'recommendation': '.FALSE.', 'warnings':[]},
    'ML_FF_LCRITERIA': {'chinese_name': 'ML使用学习标准', 'description': '标准', 'recommendation': '.FALSE.', 'warnings':[]},
    'ML_FF_LEATOM_MB': {'chinese_name': 'ML使用原子能量', 'description': '参考', 'recommendation': '.FALSE.', 'warnings':[]},
    'ML_FF_LHEAT_MB': {'chinese_name': 'ML计算热流', 'description': 'MD', 'recommendation': '.FALSE.', 'warnings':[]},
    'ML_FF_CSIG': {'chinese_name': 'ML信号噪声比', 'description': '阈值', 'recommendation': '3.0', 'warnings':[]},
    'ML_FF_CSLOPE': {'chinese_name': 'ML斜率缩放', 'description': '因子', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_CTIFOR': {'chinese_name': 'ML离子力置信', 'description': '阈值', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_WTIFOR': {'chinese_name': 'ML离子力权重', 'description': '权重', 'recommendation': '1.0', 'warnings':[]},
    'ML_FF_WTOTEN': {'chinese_name': 'ML能量权重', 'description': '权重', 'recommendation': '0.01', 'warnings':[]},
    'ML_FF_WTSIF': {'chinese_name': 'ML应力权重', 'description': '权重', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_NWRITE': {'chinese_name': 'ML写入模式', 'description': '控制', 'recommendation': '2', 'warnings':[]},
    'ML_FF_ISAMPLE': {'chinese_name': 'ML采样模式', 'description': '策略', 'recommendation': '3', 'warnings':[]},
    'ML_FF_NDIM_SCALAPACK': {'chinese_name': 'ML维数', 'description': '优化', 'recommendation': '-1', 'warnings':[]},
    'ML_FF_IERR': {'chinese_name': 'ML错误处理', 'description': '方式', 'recommendation': '0', 'warnings':[]},
    'ML_FF_IWEIGHT': {'chinese_name': 'ML权重计算', 'description': '方式', 'recommendation': '0', 'warnings':[]},
    'ML_FF_AFILT2_MB': {'chinese_name': 'ML二阶滤波', 'description': '宽度', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_LAFILT2_MB': {'chinese_name': 'ML启用二阶滤波', 'description': '滤波', 'recommendation': '.FALSE.', 'warnings':[]},
    'ML_FF_IAFILT2_MB': {'chinese_name': 'ML原子滤波指标', 'description': '指标', 'recommendation': '0', 'warnings':[]},
    'ML_FF_LMAX2_MB': {'chinese_name': 'ML第二角动量', 'description': '阶数', 'recommendation': '0', 'warnings':[]},
    'ML_FF_LNORM1_MB': {'chinese_name': 'ML第一归一化', 'description': '方式', 'recommendation': '.FALSE.', 'warnings':[]},
    'ML_FF_LNORM2_MB': {'chinese_name': 'ML第二归一化', 'description': '方式', 'recommendation': '.FALSE.', 'warnings':[]},
    'ML_FF_NR1_MB': {'chinese_name': 'ML第一径向网格', 'description': '点数', 'recommendation': '20', 'warnings':[]},
    'ML_FF_NR2_MB': {'chinese_name': 'ML第二径向网格', 'description': '点数', 'recommendation': '20', 'warnings':[]},
    'ML_FF_NHYP1_MB': {'chinese_name': 'ML第一双曲势', 'description': '阶数', 'recommendation': '0', 'warnings':[]},
    'ML_FF_NHYP2_MB': {'chinese_name': 'ML第二双曲势', 'description': '阶数', 'recommendation': '0', 'warnings':[]},
    'ML_FF_MRB1_MB': {'chinese_name': 'ML第一径向基', 'description': '数量', 'recommendation': '16', 'warnings':[]},
    'ML_FF_MRB2_MB': {'chinese_name': 'ML第二径向基', 'description': '数量', 'recommendation': '16', 'warnings':[]},
    'ML_FF_MSPL1_MB': {'chinese_name': 'ML第一样条点', 'description': '点数', 'recommendation': '1000', 'warnings':[]},
    'ML_FF_MSPL2_MB': {'chinese_name': 'ML第二样条点', 'description': '点数', 'recommendation': '1000', 'warnings':[]},
    'ML_FF_SION1_MB': {'chinese_name': 'ML第一离子噪声', 'description': '噪声', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_SION2_MB': {'chinese_name': 'ML第二离子噪声', 'description': '噪声', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_IBROAD1_MB': {'chinese_name': 'ML第一广播索引', 'description': '索引', 'recommendation': '0', 'warnings':[]},
    'ML_FF_IBROAD2_MB': {'chinese_name': 'ML第二广播索引', 'description': '索引', 'recommendation': '0', 'warnings':[]},
    'ML_FF_ICUT1_MB': {'chinese_name': 'ML第一截断索引', 'description': '索引', 'recommendation': '0', 'warnings':[]},
    'ML_FF_ICUT2_MB': {'chinese_name': 'ML第二截断索引', 'description': '索引', 'recommendation': '0', 'warnings':[]},
    'ML_FF_RCUT1_MB': {'chinese_name': 'ML第一截断半径', 'description': '半径', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_RCUT2_MB': {'chinese_name': 'ML第二截断半径', 'description': '半径', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_ISOAP1_MB': {'chinese_name': 'ML第一SOAP', 'description': '描述符', 'recommendation': '0', 'warnings':[]},
    'ML_FF_ISOAP2_MB': {'chinese_name': 'ML第二SOAP', 'description': '描述符', 'recommendation': '0', 'warnings':[]},
    'ML_FF_W1_MB': {'chinese_name': 'ML权重因子1', 'description': '因子', 'recommendation': '1.0', 'warnings':[]},
    'ML_FF_W2_MB': {'chinese_name': 'ML权重因子2', 'description': '因子', 'recommendation': '1.0', 'warnings':[]},
    'ML_FF_MB_MB': {'chinese_name': 'ML多体矩阵', 'description': '配置', 'recommendation': '1', 'warnings':[]},
    'ML_FF_EATOM': {'chinese_name': 'ML原子能量', 'description': '参考', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_CDOUB': {'chinese_name': 'ML双层因子', 'description': '因子', 'recommendation': '1.0', 'warnings':[]},
    'ML_FF_CSF': {'chinese_name': 'ML置信度缩放', 'description': '缩放', 'recommendation': '1.0', 'warnings':[]},
    'ML_FF_SIGV0_MB': {'chinese_name': 'ML势能噪声', 'description': '估计', 'recommendation': '0.001', 'warnings':[]},
    'ML_FF_SIGW0_MB': {'chinese_name': 'ML力噪声', 'description': '估计', 'recommendation': '0.001', 'warnings':[]},
    'ML_FF_ISCALE_TOTEN_MB': {'chinese_name': 'ML能量缩放', 'description': '因子', 'recommendation': '0', 'warnings':[]},
    'ML_FF_ICOUPLE_MB': {'chinese_name': 'ML耦合索引', 'description': '索引', 'recommendation': '0', 'warnings':[]},
    'ML_FF_LCOUPLE_MB': {'chinese_name': 'ML启用耦合', 'description': '耦合', 'recommendation': '.FALSE.', 'warnings':[]},
    'ML_FF_RCOUPLE_MB': {'chinese_name': 'ML耦合半径', 'description': '截断', 'recommendation': '0.0', 'warnings':[]},
    'ML_FF_NATOM_COUPLED_MB': {'chinese_name': 'ML耦合原子数', 'description': '数量', 'recommendation': '0', 'warnings':[]},
    'ML_FF_IREG_MB': {'chinese_name': 'ML正则化索引', 'description': '索引', 'recommendation': '0', 'warnings':[]},
    'ML_FF_NMDINT': {'chinese_name': 'ML动力学间隔', 'description': '采样', 'recommendation': '100', 'warnings':[]},
    'M_CONSTR': {'chinese_name': '约束质量', 'description': '惯性', 'recommendation': '0.0', 'warnings':[]},
    'NGX': {'chinese_name': 'X网格', 'description': '实空间', 'recommendation': '0', 'warnings':[]},
    'NGXF': {'chinese_name': 'X傅里叶网格', 'description': '倒空间', 'recommendation': '0', 'warnings':[]},
    'NGY': {'chinese_name': 'Y网格', 'description': '实空间', 'recommendation': '0', 'warnings':[]},
    'NGYF': {'chinese_name': 'Y傅里叶网格', 'description': '倒空间', 'recommendation': '0', 'warnings':[]},
    'NGYROMAG': {'chinese_name': '磁性实空间网格', 'description': '分辨率', 'recommendation': 'None', 'warnings':[]},
    'NGZ': {'chinese_name': 'Z网格', 'description': '实空间', 'recommendation': '0', 'warnings':[]},
    'NGZF': {'chinese_name': 'Z傅里叶网格', 'description': '倒空间', 'recommendation': '0', 'warnings':[]},
    'NSUBSYS': {'chinese_name': 'MD子系统', 'description': '配置', 'recommendation': 'None', 'warnings':[]},
    'STM': {'chinese_name': 'STM模拟偏压', 'description': '显微镜', 'recommendation': '0.0', 'warnings':[]},
    'WC': {'chinese_name': '权重因子', 'description': '优化', 'recommendation': '1.0', 'warnings':[]}
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
    
    # 提取 POSCAR 元素顺序
    poscar_el_seq =[]
    for site in structure.sites:
        sym = site.species_string
        if not poscar_el_seq or poscar_el_seq[-1] != sym:
            if sym not in poscar_el_seq:
                poscar_el_seq.append(sym)

    u_elements_found =[]
    rec_u_list = []
    rec_ul_list = []
    mag_elements_found =[]
    
    needs_u_from_poscar = False
    needs_mag_from_poscar = False
    
    # 解析本地字典
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
                needs_u_from_poscar = True
                u_elements_found.append(f"**{sym}** (推荐U={u_val}eV)")
        rec_u_list.append(str(u_val))
        rec_ul_list.append(str(u_l))
        
        if sym in ELEMENT_MAGNETIC_MOMENTS and ELEMENT_MAGNETIC_MOMENTS[sym] > 0:
            needs_mag_from_poscar = True
            mag_elements_found.append(f"**{sym}** (推荐初始 {ELEMENT_MAGNETIC_MOMENTS[sym]}μB)")

    u_elements_found = list(dict.fromkeys(u_elements_found))
    mag_elements_found = list(dict.fromkeys(mag_elements_found))
    
    # 【彻底修复：智能覆盖本地缺失字典】
    # 如果 Pymatgen 材料库认为需要加 U (例如对某些特殊的 W 氧化物)，但本地字典为 0，优先采用材料库的值
    is_expert_ldau = parse_vasp_bool(expert_incar.get("LDAU", False))
    expert_u = expert_incar.get("LDAUU", None)
    expert_ul = expert_incar.get("LDAUL", None)
    
    needs_u_total = needs_u_from_poscar or is_expert_ldau

    if is_expert_ldau and sum([float(x) for x in rec_u_list]) == 0 and expert_u is not None:
        final_rec_UU = str(expert_u)
        final_rec_UL = str(expert_ul)
    else:
        final_rec_UU = " ".join(rec_u_list)
        final_rec_UL = " ".join(rec_ul_list)

    # 同理，处理 Pymatgen 的精确初始磁矩(包含数量，如 16*0.6)
    is_expert_spin = parse_vasp_bool(expert_incar.get("ISPIN", False)) or str(expert_incar.get("ISPIN", "")) in["2", "2.0"]
    needs_mag_total = needs_mag_from_poscar or is_expert_spin
    final_rec_mag = str(expert_incar.get("MAGMOM", ""))

    analysis_results = []
    top_warnings =[]
    all_tags = set(user_incar.keys()).union(set(expert_incar.keys()))
    
    is_user_ldau = parse_vasp_bool(user_incar.get("LDAU", False))
    is_user_spin = parse_vasp_bool(user_incar.get("ISPIN", False)) or str(user_incar.get("ISPIN", "")) in ["2", "2.0"]
    is_soc = parse_vasp_bool(user_incar.get("LSORBIT", False))
    is_hse = parse_vasp_bool(user_incar.get("LHFCALC", False))
    ibrion_val = user_incar.get("IBRION", "未设置")
    nsw_val = user_incar.get("NSW", "未设置")
    
    # 强制将这些关键标签纳入审查
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
        
        # 知识库提取
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
            if needs_u_total and (str(user_val) == "未设置" or not any(c in str(user_val) for c in "123456789")):
                advice = f"🚨 U值缺失/错误: 您的 POSCAR 元素顺序为 **{' '.join(poscar_el_seq)}**。必须配套设置 LDAUU = **{final_rec_UU}**。"
                top_warnings.append(advice)

        elif tag == "LDAUL":
            if needs_u_total and str(user_val) == "未设置":
                advice = f"🚨 轨道通道缺失: 根据 POSCAR 元素顺序，必须配套设置 LDAUL = **{final_rec_UL}** (-1不加，2代表d，3代表f)。"
                top_warnings.append(advice)

        elif tag == "MAGMOM":
            if needs_mag_total and str(user_val) == "未设置":
                if mag_elements_found:
                    advice = f"⚠️ 磁性丢失警告: 检测到磁性元素 {', '.join(mag_elements_found)}。若不在 INCAR 赋予初始 MAGMOM，极易掉入非磁高能态！<br>推荐: MAGMOM = **{final_rec_mag}**"
                else:
                    advice = f"⚠️ 磁性丢失警告: 材料库建议该体系需赋予初始磁矩，否则易掉入非磁高能态！<br>推荐: MAGMOM = **{final_rec_mag}**"
                top_warnings.append(advice)
                
        elif tag == "ISPIN":
            if needs_mag_total and not is_user_spin:
                advice = f"🚨 自旋关闭警告: 体系含有磁性元素，必须强制开启自旋极化 (设置 ISPIN = 2)！"
                top_warnings.append(advice)

        elif tag == "LMAXMIX":
            if (is_user_spin or is_user_ldau or is_soc):
                if str(user_val) == "未设置" or int(user_val) < 4:
                    req_val = 6 if "3" in final_rec_UL else 4
                    advice = f"🚨 收敛黑洞警告: 因体系开启了磁性或+U，必须手动指定 LMAXMIX={req_val}，否则电子步电荷震荡极难收敛！"
                    top_warnings.append(advice)

        elif tag == "NSW":
            if str(user_val) != "未设置" and int(user_val) > 0 and ibrion_val in ["-1", "未设置"]:
                advice = "🚨 冲突: NSW>0(要求弛豫)，但 IBRION=-1(静态计算)，任务会直接报错停机！"
                top_warnings.append(advice)
                
        elif tag == "ISIF":
            if str(user_val) == "3" and is_probably_2d:
                advice = "🚨 毁灭性错误: 系统侦测到大真空层 Slab 模型。绝对不能用 ISIF=3！真空层会被压没导致计算报废，必须改为 2 或 4。"
                top_warnings.append(advice)
                
        elif tag == "ISMEAR":
            if str(user_val) == "-5" and str(nsw_val) != "未设置" and int(nsw_val) > 0:
                advice = "🚨 物理错误: 结构弛豫【绝对不能】用 ISMEAR=-5 (四面体法)，会导致受力完全错误！请立即改为 0(半导体) 或 1(金属)。"
                top_warnings.append(advice)

        elif tag == "IDIPOL":
            if is_probably_2d and str(user_val) == "未设置":
                advice = "⚠️ 偶极校正提示: 体系包含大真空层。单面分子吸附或不对称极性面，务必开启 IDIPOL=3 配合 LDIPOL=.TRUE. 防止能级倾斜。"
                top_warnings.append(advice)

        elif tag == "RWIGS":
            if "LORBIT" in user_incar and int(user_incar.get("LORBIT", 10)) < 10 and str(user_val) == "未设置":
                advice = "⚠️ 半径缺失: LORBIT < 10 计算 DOS 时，必须手动设置 RWIGS 原子半径数组！建议直接改用 LORBIT = 11。"
                top_warnings.append(advice)

        if advice == "✅ 设置正常":
            if str(user_val) == "未设置":
                advice = f"ℹ️ 未设置，使用 VASP 默认值。(专家库参考: {expert_val})"
            elif str(expert_val) != "未设置" and str(user_val) != str(expert_val):
                advice = f"ℹ️ 提示: 您的设置 ({user_val}) 与 MP 经典推荐值 ({expert_val}) 有偏差，供参考。"

        analysis_results.append({
            "参数标签 (Tag)": f"**{tag}**",
            "您的设置": str(user_val),
            "专家库推荐": str(expert_val),
            "专家诊断与建议": advice,
            "内置百科": desc_text
        })
        
    df = pd.DataFrame(analysis_results)
    df['优先级'] = df['专家诊断与建议'].apply(
        lambda x: 0 if "🚨" in x else (1 if "⚠️" in x else (2 if "✅" in x else 3))
    )
    df = df.sort_values(by=['优先级', '参数标签 (Tag)']).drop(columns=['优先级']).reset_index(drop=True)
    
    top_warnings = list(dict.fromkeys(top_warnings))
    return df, top_warnings, final_rec_UU, final_rec_UL, final_rec_mag, needs_u_total, needs_mag_total, poscar_el_seq

# ==========================================
# 网页前端渲染模块 (强制自适应换行，彻底修复折叠)
# ==========================================
def render_html_table(df):
    """将 DataFrame 转换为注入了强力换行 CSS 的原生 HTML"""
    df_html = df.copy()
    
    # 转换为 HTML 前，将 DataFrame 文本中的 Markdown 加粗转化为 <b> 标签
    for col in df_html.columns:
        df_html[col] = df_html[col].astype(str).apply(lambda x: re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', x))
        
    html_table = df_html.to_html(escape=False, index=False)
    
    # 强制开启 word-wrap 和 white-space 换行
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
    
    # 【最核心修复】：彻底剥离由于 dataframe to_html 产生的隐形换行符 \n
    # 避免 Streamlit 将这些换行符识别为 Markdown 断点，从而导致源码泄漏！
    html_table = html_table.replace('<table border="1" class="dataframe">', '<table class="vasp-table">')
    final_html = (style + html_table).replace('\n', '')
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
