# Importações necessárias
import sys
import tempfile
import math
from controller import Supervisor

# --- IMPORTAÇÕES NECESSÁRIAS PARA PLOTAGEM ---
import numpy as np
import matplotlib.pyplot as plt 
# --------------------------------------------------------

try:
    import ikpy
    from ikpy.chain import Chain
except ImportError:
    sys.exit('O módulo "ikpy" não está instalado.')

# --- Listas Globais para Coletar Dados (APENAS PARA A JUNTA LÍDER) ---
time_log = []
theta_t_log = []
theta_dot_t_log = [] 
theta_ddot_t_log = [] # <<< NOVA LISTA PARA ACELERAÇÃO
# ---------------------------------------------------------------------


# --- Funções de Geração de Trajetória (Posição) - SEM ALTERAÇÕES ---

def cubic_trajectory(t, T, theta_start, theta_end):
    s = t / T
    if s < 0: s = 0
    if s > 1: s = 1
    return theta_start + (theta_end - theta_start) * (3 * s**2 - 2 * s**3)

def quintic_trajectory(t, T, theta_start, theta_end):
    s = t / T
    if s < 0: s = 0
    if s > 1: s = 1
    return theta_start + (theta_end - theta_start) * (10 * s**3 - 15 * s**4 + 6 * s**5)

def trapezoidal_trajectory(t, T, theta_start, theta_end, max_accel, max_vel):
    delta_theta = theta_end - theta_start
    sign = 1 if delta_theta > 0 else -1
    delta_theta = abs(delta_theta)
    T_accel = max_vel / max_accel
    delta_theta_accel = 0.5 * max_accel * T_accel**2
    
    # GARANTE QUE T_const SEJA INICIALIZADO
    T_const = 0.0 # <<< CORREÇÃO AQUI
    
    if delta_theta_accel * 2 > delta_theta:
        T_accel = math.sqrt(delta_theta / max_accel)
        T_total = 2 * T_accel
        v_max = max_accel * T_accel
        # T_const permanece 0.0
    else:
        T_const = (delta_theta - 2 * delta_theta_accel) / max_vel
        T_total = 2 * T_accel + T_const
        v_max = max_vel
        
    if t < 0: return theta_start
    if t > T_total: return theta_end
    T1 = T_accel
    T2 = T_accel + T_const # <<< AGORA SEM ERRO
    if t <= T1:
        current_delta_theta = 0.5 * max_accel * t**2
    elif t <= T2:
        current_delta_theta = delta_theta_accel + v_max * (t - T1)
    else:
        current_delta_theta = delta_theta - 0.5 * max_accel * (T_total - t)**2
    return theta_start + sign * current_delta_theta

# --- Funções de Velocidade (Derivada Primeira) - SEM ALTERAÇÕES ---

def cubic_velocity(t, T, theta_start, theta_end):
    s = t / T
    if t < 0 or t > T: return 0.0
    return (theta_end - theta_start) * (6 * t / T**2 - 6 * t**2 / T**3)

def quintic_velocity(t, T, theta_start, theta_end):
    s = t / T
    if t < 0 or t > T: return 0.0
    return (theta_end - theta_start) * (30 * s**2 / T - 60 * s**3 / T + 30 * s**4 / T)

def trapezoidal_velocity(t, T, theta_start, theta_end, max_accel, max_vel):
    delta_theta = theta_end - theta_start
    sign = 1 if delta_theta > 0 else -1
    delta_theta = abs(delta_theta)
    T_accel = max_vel / max_accel
    delta_theta_accel = 0.5 * max_accel * T_accel**2
    
    # GARANTE QUE T_const SEJA INICIALIZADO
    T_const = 0.0 # <<< CORREÇÃO AQUI
    
    if delta_theta_accel * 2 > delta_theta:
        T_accel = math.sqrt(delta_theta / max_accel)
        T_total = 2 * T_accel
        v_max = max_accel * T_accel
    else:
        T_const = (delta_theta - 2 * delta_theta_accel) / max_vel
        T_total = 2 * T_accel + T_const
        v_max = max_vel
        
    if t < 0 or t >= T_total: return 0.0
    T1 = T_accel
    T2 = T_accel + T_const # <<< AGORA SEM ERRO
    if t <= T1:
        theta_dot = max_accel * t
    elif t <= T2:
        theta_dot = v_max
    else:
        theta_dot = max_accel * (T_total - t)
    return sign * theta_dot

# --- NOVAS FUNÇÕES: Aceleração (Derivada Segunda) ---

def cubic_acceleration(t, T, theta_start, theta_end):
    """ Calcula a derivada segunda do polinômio cúbico (Aceleração) """
    if t < 0 or t > T: return 0.0
    # theta_ddot(t) = (theta_end - theta_start) * s_ddot
    # s_ddot = (6/T^2) - (12*t/T^3)
    return (theta_end - theta_start) * (6 / T**2 - 12 * t / T**3)

def quintic_acceleration(t, T, theta_start, theta_end):
    """ Calcula a derivada segunda do polinômio quíntico (Aceleração) """
    s = t / T
    if t < 0 or t > T: return 0.0
    # theta_ddot(t) = (theta_end - theta_start) * s_ddot
    # s_ddot = (60*s/T^2) - (180*s^2/T^2) + (120*s^3/T^2)
    return (theta_end - theta_start) * (60 * s / T**2 - 180 * s**2 / T**2 + 120 * s**3 / T**2)

def trapezoidal_acceleration(t, T, theta_start, theta_end, max_accel, max_vel):
    """ Calcula a derivada segunda do perfil trapezoidal (Aceleração) """
    delta_theta = theta_end - theta_start
    sign = 1 if delta_theta > 0 else -1
    delta_theta = abs(delta_theta)
    T_accel = max_vel / max_accel
    delta_theta_accel = 0.5 * max_accel * T_accel**2
    
    # GARANTE QUE T_const SEJA INICIALIZADO
    T_const = 0.0 # <<< CORREÇÃO AQUI
    
    if delta_theta_accel * 2 > delta_theta:
        T_accel = math.sqrt(delta_theta / max_accel)
        T_total = 2 * T_accel
    else:
        T_const = (delta_theta - 2 * delta_theta_accel) / max_vel
        T_total = 2 * T_accel + T_const
        
    if t < 0 or t >= T_total: return 0.0
    T1 = T_accel
    T2 = T_accel + T_const # <<< AGORA SEM ERRO
    
    if t <= T1:
        theta_ddot = max_accel   
    elif t <= T2:
        theta_ddot = 0.0         
    else:
        theta_ddot = -max_accel   
        
    return sign * theta_ddot
# -------------------------------------------------------------------------

# --- Inicialização do Webots e IK (SEU CÓDIGO ORIGINAL) ---

IKPY_MAX_ITERATIONS = 4
TRAJECTORY_DURATION = 4 # Alterado para 1.5s para visualização mais rápida
TRAJECTORY_TYPE = 'trapezoidal' # Alterado para 'cubic' 'trapezoidal' 'quintic' para demonstrar a suavidade

# Parâmetros para a Trapezoidal
MAX_JOINT_ACCEL = 2.0
MAX_JOINT_VEL = 1.0

supervisor = Supervisor()
timeStep = int(4 * supervisor.getBasicTimeStep())

# 1. Obter URDF e criar a cadeia cinemática
filename = None
with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
    filename = file.name
    file.write(supervisor.getUrdf().encode('utf-8'))
armChain = Chain.from_urdf_file(filename, active_links_mask=[False, True, True, True, True, True, True, False])

# 2. Inicializar motores e encoders
motors = []
for link in armChain.links:
    if 'motor' in link.name:
        motor = supervisor.getDevice(link.name)
        motor.setVelocity(10.0) # Aumentado para permitir testes de velocidade
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timeStep)
        motors.append(motor)

# 3. Definir e Calcular os pontos de passagem (Start e End)
initial_position_ik = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]
theta_start_juntas = initial_position_ik[1:-1]

target_position_xyz = [0.5, -0.5, 0.5] # Exemplo

try:
    ik_results = armChain.inverse_kinematics(target_position_xyz, initial_position=initial_position_ik)
    theta_end_juntas = ik_results[1:-1]
    
    if any(math.isnan(angle) for angle in theta_end_juntas):
        print("AVISO: IK inválido (NaN). Usando a posição inicial.")
        theta_end_juntas = theta_start_juntas

except Exception as e:
    print(f"Erro no cálculo da Cinemática Inversa: {e}")
    theta_end_juntas = theta_start_juntas

# Identifica a Junta Líder para coleta de dados
delta_thetas = [abs(theta_end_juntas[i] - theta_start_juntas[i]) for i in range(len(motors))]
if delta_thetas:
    leading_joint_index = np.argmax(delta_thetas)
else:
    leading_joint_index = 0

print(f"Iniciando movimento com trajetória: {TRAJECTORY_TYPE} por {TRAJECTORY_DURATION}s")

# --- Loop de Execução da Trajetória (COM COLETA DE DADOS DE ACELERAÇÃO) ---
time_start = supervisor.getTime()

while supervisor.step(timeStep) != -1:
    t_elapsed = supervisor.getTime() - time_start
    
    if t_elapsed >= TRAJECTORY_DURATION:
        for i in range(len(motors)):
            motors[i].setPosition(theta_end_juntas[i])
        print("Movimento concluído. Gerando gráficos...")
        break
        
    for i in range(len(motors)):
        theta_start = theta_start_juntas[i]
        theta_end = theta_end_juntas[i]
        
        # 1. CÁLCULO DAS TRÊS VARIÁVEIS (POSIÇÃO, VELOCIDADE, ACELERAÇÃO)
        if TRAJECTORY_TYPE == 'cubic':
            theta_t = cubic_trajectory(t_elapsed, TRAJECTORY_DURATION, theta_start, theta_end)
            theta_dot_t = cubic_velocity(t_elapsed, TRAJECTORY_DURATION, theta_start, theta_end)
            theta_ddot_t = cubic_acceleration(t_elapsed, TRAJECTORY_DURATION, theta_start, theta_end) # <<< NOVO
            
        elif TRAJECTORY_TYPE == 'quintic':
            theta_t = quintic_trajectory(t_elapsed, TRAJECTORY_DURATION, theta_start, theta_end)
            theta_dot_t = quintic_velocity(t_elapsed, TRAJECTORY_DURATION, theta_start, theta_end)
            theta_ddot_t = quintic_acceleration(t_elapsed, TRAJECTORY_DURATION, theta_start, theta_end) # <<< NOVO
            
        elif TRAJECTORY_TYPE == 'trapezoidal':
            theta_t = trapezoidal_trajectory(t_elapsed, TRAJECTORY_DURATION, theta_start, theta_end, MAX_JOINT_ACCEL, MAX_JOINT_VEL)
            theta_dot_t = trapezoidal_velocity(t_elapsed, TRAJECTORY_DURATION, theta_start, theta_end, MAX_JOINT_ACCEL, MAX_JOINT_VEL)
            theta_ddot_t = trapezoidal_acceleration(t_elapsed, TRAJECTORY_DURATION, theta_start, theta_end, MAX_JOINT_ACCEL, MAX_JOINT_VEL) # <<< NOVO
            
        else:
            theta_t = theta_end 
            theta_dot_t = 0.0
            theta_ddot_t = 0.0
            
        # 2. APLICAÇÃO DA POSIÇÃO E VELOCIDADE (USANDO O LIMITE CALCULADO PARA MELHOR SIMULAÇÃO)
        motor = motors[i]
        motor.setVelocity(abs(theta_dot_t) + 1e-4) # Limita a velocidade pela derivada!
        motor.setPosition(theta_t)
        
        # 3. LOG DOS DADOS (APENAS PARA A JUNTA LÍDER)
        if i == leading_joint_index and t_elapsed < TRAJECTORY_DURATION:
            time_log.append(t_elapsed)
            theta_t_log.append(theta_t)
            theta_dot_t_log.append(theta_dot_t)
            theta_ddot_t_log.append(theta_ddot_t) # <<< NOVO LOG

# ---------------- GERAÇÃO DE GRÁFICOS ACADÊMICOS (AGORA COM 3 SUBPLOTS) ----------------

def plot_trajectory_data(time, position, velocity, acceleration, type_name):
    """
    Plota Posição, Velocidade e Aceleração em três subplots.
    
    """
    plt.figure(figsize=(15, 4.5)) # Tamanho ajustado para 3 gráficos
    
    # Gráfico 1: Posição (θ vs t)
    plt.subplot(1, 3, 1)
    plt.plot(time, position, color='blue')
    plt.title(rf'1. Posição ($\Theta$) - {type_name}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo (rad)')
    plt.grid(True)
    
    # Gráfico 2: Velocidade (θ̇ vs t)
    plt.subplot(1, 3, 2)
    plt.plot(time, velocity, color='green')
    plt.title(rf'2. Velocidade ($\dot{{\Theta}}$) - {type_name}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Velocidade (rad/s)')
    plt.grid(True)
    
    # Gráfico 3: Aceleração (θ̈ vs t) <<< NOVO GRÁFICO
    plt.subplot(1, 3, 3)
    plt.plot(time, acceleration, color='red')
    plt.title(rf'3. Aceleração ($\ddot{{\Theta}}$) - {type_name}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Aceleração (rad/s²)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'trajectory_profile_{type_name}.png')
    plt.show()

# Chamada da função de plotagem
if time_log:
    plot_trajectory_data(time_log, theta_t_log, theta_dot_t_log, theta_ddot_t_log, TRAJECTORY_TYPE)
    print(f"Gráficos de Posição, Velocidade e Aceleração gerados e salvos como trajectory_profile_{TRAJECTORY_TYPE}.png.")

# Loop de simulação final
while supervisor.step(timeStep) != -1:
    pass