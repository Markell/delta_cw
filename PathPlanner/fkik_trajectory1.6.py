
# =================================================================================================
# -- IMPORTS --------------------------------------------------------------------------------------
# =================================================================================================

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np

from numpy import sin, cos, tan, pi


def tand(theta):
	return tan(theta * pi / 180)


def sind(theta):
	return sin(theta * pi / 180)


def cosd(theta):
	return cos(theta * pi / 180)


import time
import math

import serial
import serial.tools.list_ports


# =================================================================================================
# -- Trajectory Planner (11th order, 4 points) ----------------------------------------------------
# =================================================================================================

class PathPlannerMLTP:
	"""
	Minimal multi-point planner: only the 11th-order 4-point polynomial method
	is needed here. It builds a smooth profile through exactly 4 way-points
	(theta0..theta3) on the normalised time interval [0, 1].
	"""

	def __init__(self, path, sampling_frequency=100):
		self.path 				= np.array(path, dtype=float)
		self.n 					= len(self.path) - 1
		self.sampling_frequency = sampling_frequency
		self.sampling_number 	= sampling_frequency * self.n + 1
		self.t_path				= np.linspace(0, 1, self.n + 1)
		self.T_path 			= self.t_path[1:self.n+1] - self.t_path[0:self.n]

	def mltp_polynomial11th_4point(self):

		if len(self.path) != 4:
			print("The length of the path is not equal to four")
			return None

		theta0 = self.path[0]
		theta1 = self.path[1]
		theta2 = self.path[2]
		theta3 = self.path[3]
		t      = np.linspace(0, 1, self.sampling_number)

		a0 	= theta0
		a1 	= 0
		a2 	= 0
		a3 	= 0
		a4 	= 0
		a5 	= 3690.5625 * theta1 - 3014.71875 * theta0 - 1845.28125 * theta2 + 1169.4375 * theta3
		a6 	= 18795.75 * theta0 - 23988.65625 * theta1 + 14762.25 * theta2 - 9569.34375 * theta3
		a7 	= 64584.84375 * theta1 - 49087.96875 * theta0 - 46132.03125 * theta2 + 30635.15625 * theta3
		a8 	= 68523.75 * theta0 - 92264.0625 * theta1 + 73811.25 * theta2 - 50070.9375 * theta3
		a9 	= 73811.25 * theta1 - 53835.15625 * theta0 - 64584.84375 * theta2 + 44608.75 * theta3
		a10 = 22549.5 * theta0 - 31369.78125 * theta1 + 29524.5 * theta2 - 20704.21875 * theta3
		a11 = 5535.84375 * theta1 - 3932.15625 * theta0 - 5535.84375 * theta2 + 3932.15625 * theta3

		theta      	= a11 * t**11 + a10 * t**10 + a9 * t**9 + a8 * t**8 + a7 * t**7 + a6 * t**6 + a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0
		theta_dot  	= 11 * a11 * t**10 + 10 * a10 * t**9 + 9 * a9 * t**8 + 8 * a8 * t**7 + 7 * a7 * t**6 + 6 * a6 * t**5 + 5 * a5 * t**4 + 4 * a4 * t**3 + 3 * a3 * t**2 + 2 * a2 * t + a1
		theta_ddot 	= 110 * a11 * t**9 + 90 * a10 * t**8 + 72 * a9 * t**7 + 56 * a8 * t**6 + 42 * a7 * t**5 + 30 * a6 * t**4 + 20 * a5 * t**3 + 12 * a4 * t**2 + 6 * a3 * t + 2 * a2
		theta_dddot = 990 * a11 * t**8 + 720 * a10 * t**7 + 504 * a9 * t**6 + 336 * a8 * t**5 + 210 * a7 * t**4 + 120 * a6 * t**3 + 60 * a5 * t**2 + 24 * a4 * t + 6 * a3

		return t, theta, theta_dot, theta_ddot, theta_dddot


# =================================================================================================
# -- Delta Robot Class ----------------------------------------------------------------------------
# =================================================================================================

class DeltaKinematics:
	def __init__(self, rod_b=0.1, rod_ee=0.25, r_b=0.1267, r_ee=0.045):
		'''
		configs the robot
		rod_B = length of the link connected to the base
		rod_ee = length of the link connected to the end-effector
		r_b   = radius of the base			(distance from center to pin joints)
		r_ee  = radius of the end effector 	(distance from center to universal joints)
		'''

		self.rod_b = rod_b
		self.rod_ee = rod_ee
		self.r_b = r_b
		self.r_ee = r_ee
		self.alpha = np.array([0, 120, 240])

	def fk(self, theta):
		# calculate FK, takes theta(deg)

		rod_b = self.rod_b
		rod_ee = self.rod_ee

		theta = np.array(theta)

		theta1 = theta[0]
		theta2 = theta[1]
		theta3 = theta[2]

		side_ee	= 2/tand(30)*self.r_ee
		side_b 	= 2/tand(30)*self.r_b

		t = (side_b - side_ee)*tand(30)/2

		y1 = -(t + rod_b*cosd(theta1))
		z1 = -rod_b*sind(theta1)

		y2 = (t + rod_b*cosd(theta2))*sind(30)
		x2 = y2*tand(60)
		z2 = -rod_b*sind(theta2)

		y3 = (t + rod_b*cosd(theta3))*sind(30)
		x3 = -y3*tand(60)
		z3 = -rod_b*sind(theta3)

		dnm = (y2 - y1)*x3 - (y3 - y1)*x2

		w1 = y1**2 + z1**2
		w2 = x2**2 + y2**2 + z2**2
		w3 = x3**2 + y3**2 + z3**2

		a1 = (z2-z1)*(y3-y1) - (z3-z1)*(y2-y1)
		b1 = -((w2-w1)*(y3-y1) - (w3-w1)*(y2-y1))/2

		a2 = -(z2-z1)*x3 + (z3-z1)*x2
		b2 = ((w2-w1)*x3 - (w3-w1)*x2)/2

		a = a1**2 + a2**2 + dnm**2
		b = 2*(a1*b1 + a2*(b2-y1*dnm) - z1*dnm**2)
		c = (b2 - y1*dnm)**2 + b1**2 + dnm**2*(z1**2 - rod_ee**2)

		d = b**2 - 4*a*c
		if d < 0:
			return -1

		z0 = -0.5*(b + d**0.5)/a
		x0 = (a1*z0 + b1)/dnm
		y0 = (a2*z0 + b2)/dnm

		return np.array([x0, y0, z0])

	def ik(self, _3d_pose):
		# calculates IK, returns theta(deg)
		[x0, y0, z0] = _3d_pose

		rod_ee = self.rod_ee
		rod_b = self.rod_b
		r_ee = self.r_ee
		r_b = self.r_b
		alpha = self.alpha

		F1_pos = ([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
		J1_pos = ([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
		theta = [0, 0, 0]

		for i in [0, 1, 2]:

			x = x0*cosd(alpha[i]) + y0*sind(alpha[i])
			y = -x0*sind(alpha[i]) + y0*cosd(alpha[i])
			z = z0

			ee_pos = np.array([x, y, z])

			E1_pos = ee_pos + np.array([0, -r_ee, 0])
			E1_prime_pos = np.array([0, E1_pos[1], E1_pos[2]])
			F1_pos[i] = np.array([0, -r_b, 0])

			_x0 = E1_pos[0]
			_y0 = E1_pos[1]
			_z0 = E1_pos[2]
			_yf = F1_pos[i][1]

			c1 = (_x0**2 + _y0**2 + _z0**2 + rod_b**2 - rod_ee**2 - _yf**2)/(2*_z0)
			c2 = (_yf - _y0)/_z0
			c3 = -(c1 + c2*_yf)**2 + (c2**2 + 1)*rod_b**2

			if c3 < 0:
				# print("non existing point")
				return int(-1)

			J1_y = (_yf - c1*c2 - c3**0.5)/(c2**2 + 1)
			J1_z = c1 + c2*J1_y
			F1_y = -r_b

			theta[i] = math.atan(-J1_z/(F1_y - J1_y))*180/pi

		return np.array(theta)


# =================================================================================================
# -- GUI ------------------------------------------------------------------------------------------
# =================================================================================================

class DeltaRobotGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Delta kursovaya rabota")

        # --- Полноэкранный размер 16:9 ---------------------------------------------------------
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        # подгоняем окно под экран, сохраняя 16:9
        win_w = sw
        win_h = int(win_w * 9 / 16)
        if win_h > sh:
            win_h = sh
            win_w = int(win_h * 16 / 9)
        self.root.geometry(f"{win_w}x{win_h}+0+0")
        try:
            self.root.state("zoomed")          # Windows: развернуть на весь экран
        except Exception:
            pass

        self.delta = DeltaKinematics()
        self.ser = None

        # Параметры преобразования угол → микросекунды
        self.us_per_degree = 660 / 90
        self.us_max = 2250          # крайнее верхнее положение (0 градусов)
        self.us_min = 1500          # крайнее нижнее положение

        # Состояние выполнения траектории
        self.traj_running = False
        self.traj_after_id = None
        self.traj_data = None
        self.traj_index = 0

        # Анти-дребезг / троттлинг отправки на UART
        self._last_us_sent = None       # последние реально отправленные мкс [u1,u2,u3]
        self._last_send_time = 0.0       # время последней отправки
        self.min_send_interval = 0.02    # минимум 20 мс между командами

        # маркер текущей позиции на 3D-графике (artist)
        self._marker = None

        # Состояние шагового двигателя (4 положения по 90°)
        self.stepper_position = 0     # 0,1,2,3

        self.create_widgets()

    # =============================================================================================
    # -- ВИДЖЕТЫ: двухколоночный layout (управление слева, график справа) -------------------------
    # =============================================================================================
    def create_widgets(self):
        # корневая сетка: колонка 0 — панель управления, колонка 1 — график (шире)
        self.root.columnconfigure(0, weight=0, minsize=560)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ---- ЛЕВАЯ КОЛОНКА: прокручиваемая панель управления ----
        left = ttk.Frame(self.root)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

        ttk.Label(left, text="Управление дельта роботом",
                  font=("Arial", 18, "bold")).pack(anchor="w", pady=(0, 8))

        # Подключение к Arduino
        serial_frame = ttk.LabelFrame(left, text="Подключение к Arduino (UART)", padding=10)
        serial_frame.pack(fill="x", pady=5)

        self.port_var = tk.StringVar()
        ttk.Label(serial_frame, text="COM Port:").grid(row=0, column=0, padx=5, pady=3, sticky="e")
        self.port_combo = ttk.Combobox(serial_frame, textvariable=self.port_var, width=14)
        self.port_combo.grid(row=0, column=1, padx=5, pady=3)
        ttk.Button(serial_frame, text="Обновить", command=self.update_ports).grid(row=0, column=2, padx=3)
        ttk.Button(serial_frame, text="Подключить", command=self.connect_serial).grid(row=1, column=0, padx=3, pady=3)
        ttk.Button(serial_frame, text="Отключить", command=self.disconnect_serial).grid(row=1, column=1, padx=3, pady=3)
        self.connection_status = tk.StringVar(value="Не подключено")
        self.conn_label = ttk.Label(serial_frame, textvariable=self.connection_status, foreground="red")
        self.conn_label.grid(row=1, column=2, padx=8)

        # IK
        ik_frame = ttk.LabelFrame(left, text="Решение обратной задачи кинематики (IK)", padding=10)
        ik_frame.pack(fill="x", pady=6)
        for i, label in enumerate(["X:", "Y:", "Z:"]):
            ttk.Label(ik_frame, text=label).grid(row=0, column=i*2, padx=4, pady=4, sticky="e")
            var = tk.DoubleVar(value=[0.0, -0.15, -0.42][i])
            setattr(self, f"xyz_vars{i}", var)
            ttk.Entry(ik_frame, textvariable=var, width=10).grid(row=0, column=i*2+1, padx=4)
        ttk.Button(ik_frame, text="Рассчитать IK",
                   command=self.calculate_ik).grid(row=0, column=6, padx=10, pady=4)
        self.ik_result = tk.StringVar(value="Результат IK появится здесь")
        ttk.Label(ik_frame, textvariable=self.ik_result, font=("Consolas", 10)).grid(
            row=1, column=0, columnspan=7, pady=5, sticky="w")

        # FK
        fk_frame = ttk.LabelFrame(left, text="Решение прямой задачи кинематики (FK)", padding=10)
        fk_frame.pack(fill="x", pady=6)
        for i, label in enumerate(["θ1 (°):", "θ2 (°):", "θ3 (°):"]):
            ttk.Label(fk_frame, text=label).grid(row=0, column=i*2, padx=4, pady=4, sticky="e")
            var = tk.DoubleVar(value=[0.96, 49.33, 49.33][i])
            setattr(self, f"t{i+1}_var", var)
            ttk.Entry(fk_frame, textvariable=var, width=10).grid(row=0, column=i*2+1, padx=4)
        ttk.Button(fk_frame, text="Рассчитать FK",
                   command=self.calculate_fk).grid(row=0, column=6, padx=10, pady=4)
        self.fk_result = tk.StringVar(value="Результат FK появится здесь")
        ttk.Label(fk_frame, textvariable=self.fk_result, font=("Consolas", 10)).grid(
            row=1, column=0, columnspan=7, pady=5, sticky="w")

        ttk.Button(left, text="Отправить текущие углы на Arduino",
                   command=self.send_current_angles).pack(fill="x", pady=8)

        # ---- Траектория (полином 11-го порядка, 4 точки) ----
        traj_frame = ttk.LabelFrame(left,
                     text="Построение траектории по 4 опорным точкам", padding=10)
        traj_frame.pack(fill="x", pady=6)

        default_pts = [
            (0.00, -0.15, -0.42),
            (0.05, -0.05, -0.40),
            (-0.05, 0.05, -0.40),
            (0.00,  0.15, -0.42),
        ]
        self.traj_vars = []
        for j, head in enumerate(["X (м)", "Y (м)", "Z (м)"]):
            ttk.Label(traj_frame, text=head, font=("Arial", 9, "bold")).grid(row=0, column=j+1, padx=4)
        for p in range(4):
            ttk.Label(traj_frame, text=f"Точка {p+1}:").grid(row=p+1, column=0, padx=4, pady=2, sticky="e")
            row_vars = []
            for c in range(3):
                v = tk.DoubleVar(value=default_pts[p][c])
                ttk.Entry(traj_frame, textvariable=v, width=9).grid(row=p+1, column=c+1, padx=4, pady=2)
                row_vars.append(v)
            self.traj_vars.append(tuple(row_vars))

        ttk.Label(traj_frame, text="Время движения (с):").grid(row=1, column=4, padx=(16, 4), sticky="e")
        self.traj_time_var = tk.DoubleVar(value=4.0)
        ttk.Entry(traj_frame, textvariable=self.traj_time_var, width=7).grid(row=1, column=5, padx=4)
        ttk.Label(traj_frame, text="Сэмплов/сегмент:").grid(row=2, column=4, padx=(16, 4), sticky="e")
        self.traj_samples_var = tk.IntVar(value=100)
        ttk.Entry(traj_frame, textvariable=self.traj_samples_var, width=7).grid(row=2, column=5, padx=4)

        ttk.Label(traj_frame, text="Метод:").grid(row=3, column=4, padx=(16, 4), sticky="e")
        self.traj_method_var = tk.StringVar(value="cubicspline")
        ttk.Combobox(traj_frame, textvariable=self.traj_method_var, width=14, state="readonly",
                     values=["cubicspline", "polynomial11th", "polynomial9th", "polynomial7th"]
                     ).grid(row=3, column=5, padx=4)

        ttk.Button(traj_frame, text="Построить и показать",
                   command=self.build_trajectory).grid(row=4, column=4, columnspan=2, padx=12, pady=6)
        ttk.Button(traj_frame, text="▶ Выполнить",
                   command=self.run_trajectory).grid(row=5, column=1, columnspan=2, pady=5)
        ttk.Button(traj_frame, text="■ Стоп",
                   command=self.stop_trajectory).grid(row=5, column=3, pady=5)
        self.traj_status = tk.StringVar(value="Траектория не построена")
        ttk.Label(traj_frame, textvariable=self.traj_status, font=("Consolas", 9),
                  foreground="blue", wraplength=520, justify="left").grid(
            row=6, column=0, columnspan=6, pady=5, sticky="w")

        # ---- Прямое управление микросекундами ----
        servo_frame = ttk.LabelFrame(left, text="Прямое управление сервоприводами (μs)", padding=10)
        servo_frame.pack(fill="x", pady=6)
        self.servo_vars = []
        for i in range(3):
            ttk.Label(servo_frame, text=f"Servo {i+1}:").grid(row=0, column=i*2, padx=6, pady=4, sticky="e")
            var = tk.IntVar(value=2250)
            self.servo_vars.append(var)
            ttk.Entry(servo_frame, textvariable=var, width=8).grid(row=0, column=i*2+1, padx=4)
        ttk.Button(servo_frame, text="Отправить μs",
                   command=self.send_microseconds).grid(row=1, column=0, columnspan=2, padx=6, pady=6)
        ttk.Button(servo_frame, text="В среднее (1875 μs)",
                   command=self.center_servos).grid(row=1, column=2, columnspan=3, padx=6, pady=6)

        # ---- Шаговый двигатель (A4988), 4 положения по 90° ----
        stepper_frame = ttk.LabelFrame(left, text="Шаговый двигатель", padding=10)
        stepper_frame.pack(fill="x", pady=6)

        ttk.Button(stepper_frame, text="⟳ Провернуть на следующее (90°)",
                   command=self.step_next_position).grid(row=0, column=0, columnspan=2, padx=6, pady=6, sticky="we")
        ttk.Button(stepper_frame, text="⟲ Назад (−90°)",
                   command=self.step_prev_position).grid(row=0, column=2, padx=6, pady=6, sticky="we")
        ttk.Button(stepper_frame, text="Сброс счётчика",
                   command=self.reset_stepper_counter).grid(row=0, column=3, padx=6, pady=6, sticky="we")

        self.stepper_status = tk.StringVar(value="Текущее положение: 0  (0°)")
        ttk.Label(stepper_frame, textvariable=self.stepper_status,
                  font=("Consolas", 10), foreground="purple").grid(
            row=1, column=0, columnspan=4, pady=4, sticky="w")

        # ---- ПРАВАЯ КОЛОНКА: большой 3D-график ----
        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._setup_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.update_ports()

    def _setup_axes(self):
        self.ax.set_xlabel("X (м)")
        self.ax.set_ylabel("Y (м)")
        self.ax.set_zlabel("Z (м)")
        self.ax.grid(True)

    # ====================== Преобразование углов / отправка ======================

    def angle_to_microseconds(self, angle_deg):
        """Преобразует угол сервопривода в микросекунды (с ограничением диапазона)."""
        us = self.us_max - (angle_deg * self.us_per_degree)
        us = max(self.us_min, min(self.us_max, us))
        return int(round(us))

    def _write_us(self, us_values, force=False):
        """
        Низкоуровневая отправка трёх значений μs на Arduino.
        Анти-дребезг: не отправляем повторно одинаковые значения и
        соблюдаем минимальный интервал между командами.
        """
        if not self.ser or not self.ser.is_open:
            return False

        us_values = [int(max(self.us_min, min(self.us_max, u))) for u in us_values]

        now = time.time()
        # пропускаем дубликат (главная причина дребезга в статике)
        if (not force) and self._last_us_sent == us_values:
            return True
        # троттлинг по времени
        if (not force) and (now - self._last_send_time) < self.min_send_interval:
            return True

        try:
            cmd = f"US1:{us_values[0]} US2:{us_values[1]} US3:{us_values[2]}\n"
            self.ser.write(cmd.encode())
            self.ser.flush()                 # гарантируем выталкивание буфера
            self._last_us_sent = us_values
            self._last_send_time = now
            print("Отправлено:", cmd.strip())
            return True
        except Exception as e:
            print("Ошибка отправки:", e)
            return False

    def send_command(self, theta, force=False):
        """Отправка углов (градусы) на Arduino, конвертируя в μs."""
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("Ошибка", "Arduino не подключено!")
            return False
        try:
            us_values = []
            for ang in theta:
                if ang < 0 or ang > 102:
                    raise ValueError(f"Недопустимый угол: {ang:.2f}")
                us_values.append(self.angle_to_microseconds(ang))
            return self._write_us(us_values, force=force)
        except Exception as e:
            print("Ошибка отправки:", e)
            return False

    def send_current_angles(self):
        try:
            theta = [getattr(self, f"t{i+1}_var").get() for i in range(3)]
            us_values = [self.angle_to_microseconds(t) for t in theta]
            if self.send_command(theta, force=True):
                info = "\n".join([
                    f"θ1 = {theta[0]:.3f}° → {us_values[0]} мкс",
                    f"θ2 = {theta[1]:.3f}° → {us_values[1]} мкс",
                    f"θ3 = {theta[2]:.3f}° → {us_values[2]} мкс"
                ])
                messagebox.showinfo("Отправлено на Arduino", info)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def send_microseconds(self):
        """Прямое управление: отправляем значения μs как есть (с клампом), принудительно."""
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("Ошибка", "Arduino не подключено!")
            return
        try:
            raw = [var.get() for var in self.servo_vars]
            us = [int(max(self.us_min, min(self.us_max, v))) for v in raw]
            # отразим клампнутые значения обратно в поля
            for var, u in zip(self.servo_vars, us):
                var.set(u)
            if self._write_us(us, force=True):
                messagebox.showinfo("Отправлено", f"μs: {us[0]}, {us[1]}, {us[2]}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def center_servos(self):
        """Поставить все серво в среднее положение (удобно для проверки связи)."""
        mid = int((self.us_min + self.us_max) / 2)
        for var in self.servo_vars:
            var.set(mid)
        if self.ser and self.ser.is_open:
            self._write_us([mid, mid, mid], force=True)

    # ====================== IK / FK ======================

    def calculate_ik(self):
        try:
            pose = [getattr(self, f"xyz_vars{i}").get() for i in range(3)]
            theta = self.delta.ik(pose)
            if theta is None or np.isscalar(theta):
                self.ik_result.set("❌ Недостижимая точка!")
                return
            self.ik_result.set(f"θ1 = {theta[0]:.3f}°   θ2 = {theta[1]:.3f}°   θ3 = {theta[2]:.3f}°")
            for i in range(3):
                getattr(self, f"t{i+1}_var").set(round(theta[i], 3))
            self.plot_position(pose)
        except Exception as e:
            messagebox.showerror("Ошибка IK", str(e))

    def calculate_fk(self):
        try:
            theta = [getattr(self, f"t{i+1}_var").get() for i in range(3)]
            pos = self.delta.fk(theta)
            if pos is None or np.isscalar(pos):
                self.fk_result.set("❌ Недостижимая конфигурация!")
                return
            self.fk_result.set(f"X = {pos[0]:.4f}   Y = {pos[1]:.4f}   Z = {pos[2]:.4f}")
            self.plot_position(pos)
        except Exception as e:
            messagebox.showerror("Ошибка FK", str(e))

    # ====================== Траектория ======================

    def build_trajectory(self):
        """Строит траекторию полиномом 11-го порядка через 4 точки и считает IK для каждого сэмпла."""
        try:
            pts = []
            for p in range(4):
                pts.append((self.traj_vars[p][0].get(),
                            self.traj_vars[p][1].get(),
                            self.traj_vars[p][2].get()))
            pts = np.array(pts, dtype=float)

            samples = max(2, int(self.traj_samples_var.get()))
            method = self.traj_method_var.get()

            planner_x = PathPlannerMLTP(pts[:, 0], sampling_frequency=samples)
            planner_y = PathPlannerMLTP(pts[:, 1], sampling_frequency=samples)
            planner_z = PathPlannerMLTP(pts[:, 2], sampling_frequency=samples)

            # Кубический сплайн непрерывен по скорости/ускорению в опорных точках
            # и не осциллирует при неравномерных длинах отрезков (в отличие от
            # полиномов высокой степени с фиксированными коэффициентами).
            method_map = {
                "cubicspline":     "mltp_cubicspline",
                "polynomial11th":  "mltp_polynomial11th_4point",
                "polynomial9th":   "mltp_polynomial9th_4point",
                "polynomial7th":   "mltp_polynomial7th_4point",
            }
            fn = method_map.get(method, "mltp_cubicspline")

            res_x = getattr(planner_x, fn)()
            res_y = getattr(planner_y, fn)()
            res_z = getattr(planner_z, fn)()

            if res_x is None or res_y is None or res_z is None:
                self.traj_status.set("❌ Нужно ровно 4 точки.")
                return

            t = res_x[0]
            xs, ys, zs = res_x[1], res_y[1], res_z[1]

            thetas, reachable = [], []
            for i in range(len(t)):
                th = self.delta.ik([xs[i], ys[i], zs[i]])
                if np.isscalar(th):
                    thetas.append(None)
                    reachable.append(False)
                else:
                    thetas.append(np.array(th, dtype=float))
                    reachable.append(True)

            n_bad = reachable.count(False)
            self.traj_data = {"t": t, "xs": xs, "ys": ys, "zs": zs,
                              "thetas": thetas, "reachable": reachable, "pts": pts}

            self.plot_trajectory()

            if n_bad == 0:
                self.traj_status.set(
                    f"✅ Построено {len(t)} точек. Все достижимы. "
                    f"Время движения: {self.traj_time_var.get():.1f} c.")
            else:
                self.traj_status.set(
                    f"⚠ Построено {len(t)} точек, {n_bad} недостижимы (будут пропущены).")
        except Exception as e:
            messagebox.showerror("Ошибка построения траектории", str(e))

    def plot_trajectory(self):
        """Полная отрисовка сцены: линия траектории + опорные точки. Создаёт маркер."""
        if not self.traj_data:
            return
        d = self.traj_data
        self.ax.clear()
        self.ax.set_title("Траектория рабочего органа")
        self._setup_axes()

        self.ax.plot(d["xs"], d["ys"], d["zs"], color="tab:blue", linewidth=2, label="траектория")
        pts = d["pts"]
        self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="red", s=80, label="опорные точки")
        for i, p in enumerate(pts):
            self.ax.text(p[0], p[1], p[2] + 0.01, f"P{i+1}", fontsize=9)

        # маркер текущей позиции (создаём один раз, потом только двигаем)
        self._marker = self.ax.plot([d["xs"][0]], [d["ys"][0]], [d["zs"][0]],
                                    marker="o", color="lime", markersize=12,
                                    markeredgecolor="black", linestyle="None")[0]
        self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw()

    def run_trajectory(self):
        """Запускает проигрывание траектории по таймеру с отправкой на Arduino."""
        if not self.traj_data:
            messagebox.showwarning("Нет траектории", "Сначала постройте траекторию.")
            return
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("Ошибка", "Arduino не подключено!")
            return
        if self.traj_running:
            return

        self.traj_running = True
        self.traj_index = 0
        n = len(self.traj_data["t"])
        total_ms = max(1.0, self.traj_time_var.get() * 1000.0)
        self.traj_step_ms = max(int(self.min_send_interval * 1000), int(total_ms / max(1, n)))

        # Сбрасываем анти-дребезг, чтобы первый кадр (начальная точка) не был
        # отброшен фильтром дубликата/троттлинга и робот стартовал именно из P1.
        self._last_us_sent = None
        self._last_send_time = 0.0

        # счётчик кадров для прореживания тяжёлой 3D-перерисовки
        self._draw_every = 5          # рисовать маркер раз в N кадров
        self._frame_counter = 0

        # стартуем через after(0), чтобы окно успело дорисоваться и не было
        # видимого "залипания" в момент клика по кнопке.
        self.traj_after_id = self.root.after(0, self._play_step)

    def _play_step(self):
        """Один шаг: двигаем ТОЛЬКО маркер (без полной перерисовки) + отправляем углы."""
        if not self.traj_running:
            return
        d = self.traj_data
        i = self.traj_index
        n = len(d["t"])

        if i >= n:
            # финал: гарантированно фиксируем последнюю достижимую точку один раз
            self._finish_trajectory()
            return

        # двигаем маркер без clear()/полной перерисовки. Прореживаем тяжёлую
        # 3D-перерисовку, чтобы она не блокировала своевременную отправку углов:
        # рисуем первый кадр, последний и каждый N-й.
        if self._marker is not None:
            self._marker.set_data([d["xs"][i]], [d["ys"][i]])
            self._marker.set_3d_properties([d["zs"][i]])
            self._frame_counter += 1
            if (i == 0) or (i == n - 1) or (self._frame_counter % self._draw_every == 0):
                self.canvas.draw_idle()

        if d["reachable"][i]:
            theta = d["thetas"][i]
            for k in range(3):
                getattr(self, f"t{k+1}_var").set(round(float(theta[k]), 3))
            # Первый кадр — принудительно, минуя анти-дребезг, чтобы старт был из P1.
            self.send_command(theta, force=(i == 0))
            self.traj_status.set(
                f"▶ Точка {i+1}/{n}  θ=({theta[0]:.1f}, {theta[1]:.1f}, {theta[2]:.1f})°")
        else:
            self.traj_status.set(f"⏭ Точка {i+1}/{n} недостижима — пропуск")

        self.traj_index += 1
        # Планируем следующий кадр только если он есть. На последнем кадре
        # НЕ ставим новый after — иначе висящий таймер после завершения
        # продолжает дёргать отправку финальной точки и перебивает
        # последующие ручные команды (углы/μs) по UART.
        if self.traj_index < n:
            self.traj_after_id = self.root.after(self.traj_step_ms, self._play_step)
        else:
            self.traj_after_id = None
            self._finish_trajectory()

    def _finish_trajectory(self):
        """Завершение: одна принудительная отправка финальной точки, затем покой (без дребезга)."""
        if not self.traj_running and self.traj_index < len(self.traj_data["t"]):
            # уже финализировано — не повторяем
            return
        self.traj_running = False
        self.traj_after_id = None
        d = self.traj_data
        # ищем последнюю достижимую точку
        for i in range(len(d["t"]) - 1, -1, -1):
            if d["reachable"][i]:
                theta = d["thetas"][i]
                for k in range(3):
                    getattr(self, f"t{k+1}_var").set(round(float(theta[k]), 3))
                self.send_command(theta, force=True)   # один раз, начисто
                break
        self.traj_status.set("✅ Траектория выполнена. Серво удерживают финальную точку.")

    def stop_trajectory(self):
        self.traj_running = False
        if self.traj_after_id is not None:
            try:
                self.root.after_cancel(self.traj_after_id)
            except Exception:
                pass
            self.traj_after_id = None
        self.traj_status.set("■ Остановлено пользователем.")

    # ====================== Serial ======================

    def update_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_var.set(ports[0])

    def connect_serial(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = serial.Serial(self.port_var.get(), 9600, timeout=1)
            time.sleep(2)                     # ждём авто-сброс Arduino после открытия порта
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception:
                pass
            self._last_us_sent = None
            self.connection_status.set("✅ Подключено")
            self.conn_label.config(foreground="green")
            messagebox.showinfo("Успех", f"Подключено к {self.port_var.get()}")
        except Exception as e:
            messagebox.showerror("Ошибка подключения", str(e))

    def disconnect_serial(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connection_status.set("Не подключено")
            self.conn_label.config(foreground="red")
            messagebox.showinfo("Отключено", "Соединение закрыто")

    # ====================== Шаговый двигатель (A4988) ======================

    def _send_raw(self, line):
        """Отправка произвольной текстовой команды на Arduino (с переводом строки)."""
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("Ошибка", "Arduino не подключено!")
            return False
        try:
            self.ser.write((line + "\n").encode())
            self.ser.flush()
            print("Отправлено:", line)
            return True
        except Exception as e:
            print("Ошибка отправки:", e)
            messagebox.showerror("Ошибка отправки", str(e))
            return False

    def _update_stepper_label(self):
        self.stepper_status.set(
            f"Текущее положение: {self.stepper_position}  ({self.stepper_position * 90}°)")

    def step_next_position(self):
        """Провернуть шаговик на следующее положение (по часовой, +90°)."""
        if self._send_raw("STEP90"):
            self.stepper_position = (self.stepper_position + 1) % 4
            self._update_stepper_label()

    def step_prev_position(self):
        """Провернуть шаговик на предыдущее положение (против часовой, −90°)."""
        if self._send_raw("STEP90 CCW"):
            self.stepper_position = (self.stepper_position - 1) % 4
            self._update_stepper_label()

    def reset_stepper_counter(self):
        """Сбросить программный счётчик положения (двигатель не вращается)."""
        self.stepper_position = 0
        self._update_stepper_label()

    # ====================== Графика статичной точки ======================

    def plot_position(self, pos):
        self.ax.clear()
        self.ax.set_title("Положение рабочего органа")
        self._setup_axes()
        self.ax.scatter(pos[0], pos[1], pos[2], color='red', s=120)
        self.ax.text(pos[0], pos[1], pos[2] + 0.02,
                     f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})", fontsize=10)
        self._marker = None
        self.canvas.draw()

    def run(self):
        self.root.mainloop()


# =================================================================================================
# -- main loop ----------------------------------------------------------------------------
# =================================================================================================
# FK: check
# IK: check

if __name__ == "__main__":
	app = DeltaRobotGUI()
	app.run()
