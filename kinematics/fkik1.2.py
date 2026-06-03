
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
	return tan(theta*pi/180)

def sind(theta):
	return sin(theta*pi/180)

def cosd(theta):
	return cos(theta*pi/180)
 
import time 
import math 
from numpy import pi

# =================================================================================================
# -- Delta Robot Class ----------------------------------------------------------------------------
# =================================================================================================

class DeltaKinematics: 
	def __init__(self, rod_b=0.2, rod_ee=0.46, r_b=0.24, r_ee=0.095):
		'''
		configs the robot
		rod_B = length of the link connected to the base
		rod_B = length of the link connected to the end-effector
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
			c3 = -(c1 + c2*_yf)**2 + (c2**2+ 1)*rod_b**2

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
        self.root.title("Delta Robot Control — Kinematics + Arduino")
        self.root.geometry("1100x780")

        self.delta = DeltaKinematics()  # Параметры по умолчанию
        self.ser = None

        self.create_widgets()

    def create_widgets(self):
        # Заголовок
        ttk.Label(self.root, text="Delta Robot Control", 
                 font=("Arial", 18, "bold")).pack(pady=8)

        # === Подключение к Arduino ===
        serial_frame = ttk.LabelFrame(self.root, text="Подключение к Arduino (UART)", padding=10)
        serial_frame.pack(fill="x", padx=15, pady=5)

        self.port_var = tk.StringVar()
        ttk.Label(serial_frame, text="COM Port:").pack(side="left", padx=5)
        self.port_combo = ttk.Combobox(serial_frame, textvariable=self.port_var, width=15)
        self.port_combo.pack(side="left", padx=5)
        
        ttk.Button(serial_frame, text="Обновить порты", command=self.update_ports).pack(side="left", padx=5)
        ttk.Button(serial_frame, text="Подключить", command=self.connect_serial).pack(side="left", padx=5)
        ttk.Button(serial_frame, text="Отключить", command=self.disconnect_serial).pack(side="left", padx=5)

        self.connection_status = tk.StringVar(value="Не подключено")
        ttk.Label(serial_frame, textvariable=self.connection_status, foreground="red").pack(side="left", padx=10)

        # === Обратная кинематика (IK) ===
        ik_frame = ttk.LabelFrame(self.root, text="Обратная кинематика (IK)", padding=12)
        ik_frame.pack(fill="x", padx=15, pady=8)

        for i, label in enumerate(["X:", "Y:", "Z:"]):
            ttk.Label(ik_frame, text=label).grid(row=0, column=i*2, padx=5, pady=5, sticky="e")
            var = tk.DoubleVar(value=[0.0, -0.15, -0.42][i])
            setattr(self, f"xyz_vars{i}", var)
            ttk.Entry(ik_frame, textvariable=var, width=12).grid(row=0, column=i*2+1, padx=5)

        ttk.Button(ik_frame, text="Рассчитать IK", 
                  command=self.calculate_ik).grid(row=0, column=6, padx=12, pady=5)

        self.ik_result = tk.StringVar(value="Результат IK появится здесь")
        ttk.Label(ik_frame, textvariable=self.ik_result, font=("Consolas", 11)).grid(
            row=1, column=0, columnspan=7, pady=6)

        # === Прямая кинематика (FK) ===
        fk_frame = ttk.LabelFrame(self.root, text="Прямая кинематика (FK)", padding=12)
        fk_frame.pack(fill="x", padx=15, pady=8)

        for i, label in enumerate(["θ1 (°):", "θ2 (°):", "θ3 (°):"]):
            ttk.Label(fk_frame, text=label).grid(row=0, column=i*2, padx=5, pady=5, sticky="e")
            var = tk.DoubleVar(value=[0.96, 49.33, 49.33][i])
            setattr(self, f"t{i+1}_var", var)
            ttk.Entry(fk_frame, textvariable=var, width=12).grid(row=0, column=i*2+1, padx=5)

        ttk.Button(fk_frame, text="Рассчитать FK", 
                  command=self.calculate_fk).grid(row=0, column=6, padx=12, pady=5)

        self.fk_result = tk.StringVar(value="Результат FK появится здесь")
        ttk.Label(fk_frame, textvariable=self.fk_result, font=("Consolas", 11)).grid(
            row=1, column=0, columnspan=7, pady=6)

        # === Кнопка отправки углов на Arduino ===
        send_frame = ttk.Frame(self.root)
        send_frame.pack(pady=10)
        ttk.Button(send_frame, text="Отправить текущие углы (θ1, θ2, θ3) на Arduino", 
                  command=self.send_current_angles, width=50).pack()

        # === Прямое управление по микросекундам ===
        servo_frame = ttk.LabelFrame(self.root, text="Прямое управление сервоприводами (микросекунды)", padding=12)
        servo_frame.pack(fill="x", padx=15, pady=8)

        self.servo_vars = []
        for i in range(3):
            ttk.Label(servo_frame, text=f"Servo {i+1} (μs):").grid(row=0, column=i*2, padx=8, pady=5, sticky="e")
            var = tk.IntVar(value=1500)
            self.servo_vars.append(var)
            ttk.Entry(servo_frame, textvariable=var, width=10).grid(row=0, column=i*2+1, padx=5)
        
        ttk.Button(servo_frame, text="Отправить микросекунды", 
                  command=self.send_microseconds).grid(row=0, column=6, padx=15)

        # === 3D График ===
        self.fig = plt.figure(figsize=(7, 5.2))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=10)

        self.update_ports()

    # ====================== Методы ======================

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
            self.connection_status.set("✅ Подключено")
            # Исправляем цвет
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.LabelFrame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Label) and child.cget("textvariable") == str(self.connection_status):
                            child.config(foreground="green")
            messagebox.showinfo("Успех", f"Подключено к {self.port_var.get()}")
        except Exception as e:
            messagebox.showerror("Ошибка подключения", str(e))

    def disconnect_serial(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connection_status.set("Не подключено")
            messagebox.showinfo("Отключено", "Соединение закрыто")

    def send_command(self, theta):
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("Ошибка", "Arduino не подключено!")
            return False
        try:
            cmd = f"A1:{int(theta[0])} A2:{int(theta[1])} A3:{int(theta[2])}\n"
            self.ser.write(cmd.encode())
            return True
        except:
            return False

    def calculate_ik(self):
        try:
            pose = [getattr(self, f"xyz_vars{i}").get() for i in range(3)]
            theta = self.delta.ik(pose)

            if theta is None:
                self.ik_result.set("❌ Недостижимая точка!")
                return

            result = f"θ1 = {theta[0]:.3f}°   θ2 = {theta[1]:.3f}°   θ3 = {theta[2]:.3f}°"
            self.ik_result.set(result)

            # Обновляем поля FK
            for i in range(3):
                getattr(self, f"t{i+1}_var").set(round(theta[i], 3))

            self.plot_position(pose)
        except Exception as e:
            messagebox.showerror("Ошибка IK", str(e))

    def send_current_angles(self):
        """Отправляет текущие углы из полей θ1, θ2, θ3"""
        try:
            theta = [getattr(self, f"t{i+1}_var").get() for i in range(3)]
            if self.send_command(theta):
                messagebox.showinfo("Отправлено", 
                                  f"Углы отправлены на Arduino:\nθ1={theta[0]:.1f}°  θ2={theta[1]:.1f}°  θ3={theta[2]:.1f}°")
        except Exception as e:
            messagebox.showerror("Ошибка отправки", str(e))

    def calculate_fk(self):
        try:
            theta = [getattr(self, f"t{i+1}_var").get() for i in range(3)]
            pos = self.delta.fk(theta)

            if pos is None:
                self.fk_result.set("❌ Недостижимая конфигурация!")
                return

            result = f"X = {pos[0]:.4f}   Y = {pos[1]:.4f}   Z = {pos[2]:.4f}"
            self.fk_result.set(result)
            self.plot_position(pos)
        except Exception as e:
            messagebox.showerror("Ошибка FK", str(e))

    def send_microseconds(self):
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("Ошибка", "Arduino не подключено!")
            return
        try:
            us = [var.get() for var in self.servo_vars]
            cmd = f"US1:{us[0]} US2:{us[1]} US3:{us[2]}\n"
            self.ser.write(cmd.encode())
            messagebox.showinfo("Отправлено", f"Микросекунды: {us[0]}, {us[1]}, {us[2]}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def plot_position(self, pos):
        self.ax.clear()
        self.ax.set_title("Положение End Effector")
        self.ax.set_xlabel("X (м)")
        self.ax.set_ylabel("Y (м)")
        self.ax.set_zlabel("Z (м)")
        self.ax.grid(True)

        self.ax.scatter(pos[0], pos[1], pos[2], color='red', s=120)
        self.ax.text(pos[0], pos[1], pos[2] + 0.02, 
                    f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})", fontsize=10)

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

	# test 
	#delta = DeltaKinematics(0.2, 0.46, 0.24, 0.095)
	#ik = delta.ik([0, -0.15, -0.42])
#	fk = delta.fk([0.9635977 , 49.33026229, 49.33026229])
	#print(ik)
#	print(fk)