# In[]
from qiskit import Aer, transpile, execute
from qiskit.quantum_info import state_fidelity
from qiskit import QuantumCircuit
import numpy as np

# 创建参考门电路：Cphase(π/4) - Cphase(π/4) - Cphase(π/4) - Cphase(π/4)
ref_circuit = QuantumCircuit(2)
for _ in range(4):
    ref_circuit.cp(np.pi/4, 0, 1)

# 创建DD保护的CZ门电路：Z - Cphase(π/4) - Z - Cphase(π/4) - Z - Cphase(π/4) - Z - Cphase(π/4)
dd_circuit = QuantumCircuit(2)
for _ in range(4):
    dd_circuit.z(0)  # 假设控制量子比特为qubit 0
    dd_circuit.cp(np.pi/4, 0, 1)

# 画出电路图
ref_circuit.draw('mpl')
# In[]
dd_circuit.draw('mpl')
# In[]
# 设定模拟器
simulator = Aer.get_backend('statevector_simulator')

# 获取理想状态向量
ideal_circuit = QuantumCircuit(2)
ideal_circuit.cp(np.pi/4, 0, 1)
ideal_circuit = transpile(ideal_circuit, simulator)
ideal_result = execute(ideal_circuit, simulator).result()
ideal_state = ideal_result.get_statevector(ideal_circuit)

# 获取DD保护CZ门执行后的状态向量
transpiled_dd_circuit = transpile(dd_circuit, simulator)
dd_result = execute(transpiled_dd_circuit, simulator).result()
dd_state = dd_result.get_statevector(transpiled_dd_circuit)

# 计算保真度
fidelity = state_fidelity(ideal_state, dd_state)
print(fidelity)

# 获取无DD保护Cphase门执行后的状态向量
transpiled_ref_circuit = transpile(ref_circuit, simulator)
ref_result = execute(transpiled_ref_circuit, simulator).result()
ref_state = ref_result.get_statevector(transpiled_ref_circuit)

# 计算保真度
fidelity_ref = state_fidelity(ideal_state, ref_state)
print(fidelity_ref)