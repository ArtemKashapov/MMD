import numpy as np
import math as mt
import matplotlib.pyplot as plt
import logging

# Единицы
A = 1e-10
kb = 0.008314 # Постояная Больцмана
sig = 3.73 # Сигма
eps = 148 * kb # Эписилон
NA = 6.022 * 1e23 # Число Авагадро
R = kb * NA # Универсальная газовая постояная

logging.basicConfig(filename='output.dat', level=logging.DEBUG)


class MMD:
    def __init__(self, rho, box, temp, ndim, cutoff, dt, Q):
        self.mass = 16.04       # Масса молекулы метана в г/моль
        self.box = box          # Размер коробки
        self.temp = temp        # Температура
        self.ndim = ndim        # Размер коробки
        self.cutoff = cutoff    # Срез
        self.V = self.box*self.box*self.box   # объем коробки 
        self.rho = rho          # Плотность
        self.dt = dt            # Шаг времени
        self.Q = Q              

    def PBC(self, c):
        """
        Работа с массивом и условиями PBC и MIC
        :param c: переданный массив 
        :return: Новый массив PBC/MIC
        """
        nPart = len(c)
        for i in range(nPart):
            for j in range(i + 1, nPart):
                # compute distance and adjust for PBC
                xyz = c[i, :] - c[j, :]
                for k, f in enumerate(xyz):
                    if f < -self.box / 2:
                        n = mt.floor((-self.box / 2 - f) / self.box) + 1
                        xyz[k] += self.box * n
                    elif f > self.box / 2:
                        n = mt.floor((f - self.box / 2) / self.box) + 1
                        xyz[k] -= self.box * n
        return c

    def rho_to_num(self):
        """
        Функция для расчета количества частиц по плотности
        :return: Количество частиц
        """
        N = float(self.rho) * (int(self.box) ** 3) * 3.75 * (10 ** (-5))
        # Вычисление выражения выглядит следующим образом; единицы указаны в квадратных скобках
        # N_частицы = rho[кг м^-3] * (L_коробки [10^-10 m])^3 * 6.023*10^23[молекул/моль]/ 16.04 [ 10^-3 кг/моль]
        # N_частицы = rho*(L_коробки**3)*3.755*(10**(-5)) [молекул]
        # print("Количество частиц метана: ", int(N))
        return int(N)

    def rdf(xyz, LxLyLz, n_bins=100, r_range=(0.01, 10.0)):
        """
        радиальная функция распределения пар
        :xyz: координаты в формате xyz на кадр
        :LxLyLz: длина блока в векторном формате.
        :n_bins: количество бинов
        :r_range: диапазон для вычисления rdf
        """

        g_r, edges = np.histogram([0], bins=n_bins, range=r_range)
        g_r[0] = 0
        g_r = g_r.astype(np.float64)
        rho = 0

        for i, xyz_i in enumerate(xyz):
            xyz_j = np.vstack([xyz[:i], xyz[i + 1:]])
            d = np.abs(xyz_i - xyz_j)
            d = np.where(d > 0.5 * LxLyLz, LxLyLz - d, d)
            d = np.sqrt(np.sum(d ** 2, axis=-1))
            temp_g_r, _ = np.histogram(d, bins=n_bins, range=r_range)
            g_r += temp_g_r

        rho += (i + 1) / np.prod(LxLyLz)
        r = 0.5 * (edges[1:] + edges[:-1])
        V = 4./3. * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
        norm = rho * i
        g_r /= norm * V

        return r, g_r

    def init_grid(self):
        """
        Функция для инициализации 3D-массива для системы
        :return: Массив с координатами системы
        """
        nPart = self.rho_to_num()
        # массив для координат
        coords = np.zeros((nPart, 3))
        # Количество частиц в каждой линии решетки
        nstart, nstop = 1/16 * self.box, 15/16 * self.box
        n = int(nPart ** (1 / 3))
        # Шаг решетки
        spac = (nstop - nstart) / n
        # индексация
        index = np.zeros(3)
        # Назначать позиции частиц
        for part in range(nPart):
                coords[part, :] = (index * spac) + nstart
                # Продвигать позицию частицы
                index[0] += 1
                # если достигнута последняя точка решетки, перейти к следующей строке
                if index[0] == n:
                    index[0] = 0
                    index[1] += 1
                    if index[1] == n:
                        index[1] = 0
                        index[2] += 1
        return coords - self.box/2, self.box

    def init_vel(self, coords):
        """
        Инициализация начальных скоростей частиц системы и установки скорости COM на ноль
        :coords: Координаты частиц в системе
        :return: Начальная скорость системы
        """
        N_part = len(coords)
        vels = np.random.uniform(0, 1, size=(N_part, self.ndim))  # случайные скорости
        vels_magn2 = np.sum(np.square(vels), axis=1)
        sumv = np.sum(np.sqrt(vels_magn2), axis=0)          # сумма всех скоростей
        sumv2 = np.sum(vels_magn2, axis=0)                # сумма кинетической энергии
        sumv = sumv/N_part                  # Скорость центра масс
        sumv2 = sumv2/N_part                 # средняя квадратичная скорость
        fs = mt.sqrt(self.ndim * kb * self.temp / (self.mass * sumv2 * 1e4))   # масштаб
        vels = (vels-sumv) * fs           # смещение центра масс скорости к нулю
        return vels

    def lj_force(self, coords):
        """
        Расчет межчастичных сил в системе с использованием потенциала ЛД
        :coords: Координаты частиц в системе
        :return: Массив с межчастичными силами в системе
        """
        # Пустой массив для сил
        forces = np.zeros(coords.shape)
        # Количество частиц
        nPart = np.shape(coords)[0]
        cutoff2 = self.cutoff * self.cutoff
        sig6 = sig ** 6
        sig12 = sig6 * sig6
        lj1 = 48 * eps * sig12
        lj2 = 24 * eps * sig6

        # вычислить силы между всеми частицами
        for i in range(nPart-1):
            for j in range(i + 1, nPart):
                # вычислить расстояние и скорректировать PBC
                xyz = coords[i, :] - coords[j, :]
                for k, c in enumerate(xyz):
                    if c < -self.box / 2:
                        n = mt.floor((-self.box / 2 - c) / self.box) + 1
                        xyz[k] += self.box * n
                    elif c > self.box / 2:
                        n = mt.floor((c - self.box / 2) / self.box) + 1
                        xyz[k] -= self.box * n
                vec = xyz
                rij2 = (vec ** 2).sum(axis=-1)
                if rij2 < cutoff2:
                    r2inv = 1 / rij2
                    r6inv = r2inv * r2inv * r2inv
                    force = (r2inv * r6inv * (lj1 * r6inv - lj2))
                    forces[i, :] += vec * force
                    forces[j, :] -= vec * force
                else:
                    forces[i, :] += vec * 0
                    forces[j, :] -= vec * 0
        return forces

    def pe_pressure(self, coords):
        """
        Расчет потенциальной энергии и давления системы
        :coords: Координаты частиц в системе
        :return: Потенциальная энергия и давление
        """
        N = self.rho_to_num()
        num_density = N / self.V
        Potential_energy = 0
        P_int = 0   
        nPart = coords.shape[0]
        cutoff2 = self.cutoff * self.cutoff
        # 
        for i in range(nPart):
            for j in range(i + 1, nPart):
                xyz = coords[i, :] - coords[j, :]
                for k, c in enumerate(xyz):
                    if c < -self.box / 2:
                        n = mt.floor((-self.box / 2 - c) / self.box) + 1
                        xyz[k] += self.box * n
                    elif c > self.box / 2:
                        n = mt.floor((c - self.box / 2) / self.box) + 1
                        xyz[k] -= self.box * n
                vec = xyz
                rij2 = (vec ** 2).sum(axis=-1)
                sig6 = sig ** 6
                sig12 = sig6 * sig6
                if rij2 < cutoff2:
                    r2inv = 1 / rij2
                    r6inv = r2inv * r2inv * r2inv
                    r12inv = r6inv * r6inv
                    Potential_energy += 4 * eps * ((sig12 * r12inv) - (sig6 * r6inv))
                    P_int += 4 * eps * ((-12 * sig12 * r12inv) + (6 * sig6 * r6inv))   # Конфиг. вклад давления
        pressure = (num_density * kb * self.temp) - (P_int / (3 * self.V))  # финальное полное давление
        return Potential_energy, pressure

    def temp_ke(self, vels):
        """
        Функция для расчета кинетической энергии и температуры системы
        :vels: Скорости частиц в системе
        :return: Кинетическая энергия и температура
        """
        nPart = vels.shape[0]
        vels2 = np.sum(np.power(vels, 2))
        kinetic_Energy = (0.5 * self.mass * vels2) * 1e4
        Temperature = kinetic_Energy / (0.5 * self.ndim * nPart * kb)
        return kinetic_Energy, Temperature

    def velocity_verlet_thermostat(self, position, velocity, forces, psi):
        """
        Реализация интегратора скорости-верле с термостатом
        :param position: Координаты частиц в системе
        :param Velocity: Скорости частиц в системе.
        :param force: Силы между частицами в системе.
        :парам пси:
        :return: Положение, скорость, силы и psi системы для следующего временного шага
        """
        dt2 = self.dt * self.dt
        imass = 1/self.mass
        nPart = position.shape[0]
        # обновление позиции для следующего временного шага
        position += velocity * self.dt + dt2 * 0.5 * ((forces * imass * 1e-4) - (psi * velocity))
        position = self.PBC(position)
        KE = self.temp_ke(velocity)[0]
        self.temp = self.temp_ke(velocity)[1]
        psi_half = psi + (self.dt * (0.5 / self.Q) * ((KE/nPart) - (1.5 * kb * self.temp)))
        v_half = velocity + (0.5 * self.dt * ((imass * forces * 1e-4) - (psi_half * velocity)))
        # обновление сил
        forces = self.lj_force(position)
        # рассчитать пси и скорости при t + dt и обновить силы
        KE_half = self.temp_ke(v_half)[0]
        self.temp = self.temp_ke(v_half)[1]
        psi = psi_half + (self.dt * 0.5 / self.Q) * ((KE_half/nPart) - (1.5 * kb * self.temp))
        velocity = (v_half + (0.5 * self.dt * forces * imass * 1e-4)) / (1 + (0.5 * self.dt * psi))
        return position, velocity, forces, psi


if __name__ == '__main__':
    # md = MMD(358.4, 30, 400, 3, 14, 1, 100)   # более плотные
    md = MMD(1.6, 182, 400, 3, 50, 1, 100)    # менее плотные
    md = MMD(rho=1.6, box=182, temp=400, ndim=3, cutoff=50, dt=1, Q=100)
    Position, _ = md.init_grid()
    velocity = md.init_vel(Position)
    fcs = np.zeros(np.shape(Position))
    psi = 0
    Nfreq = 100
    run = 10000
    logging.debug('Init_temp: {}'.format(md.temp))
    logging.debug('Rho: {}'.format(md.rho))
    logging.debug('Q: {}'.format(md.Q))
    Arr_Temp = []
    Arr_Press = []
    Arr_PE = []
    Arr_KE = []
    Arr_TE = []
    for step in range(run):
        # Position, velocity, fcs = md1.velocity_Verlet(Position, velocity, fcs)
        Position, velocity, fcs, psi = md.velocity_verlet_thermostat(Position, velocity, fcs, psi)
        md.temp = md.temp_ke(velocity)[1]
        Arr_Temp.append(md.temp)
        vels2sum = vels2 = np.sum(np.power(velocity, 2))
        Pressure = md.pe_pressure(Position)[1]
        Arr_Press.append(Pressure)
        Potential_Energy = md.pe_pressure(Position)[0]
        Arr_PE.append(Potential_Energy)
        Kinetic_Energy = md.temp_ke(velocity)[0]
        Arr_KE.append(Kinetic_Energy)
        Total_energy = Kinetic_Energy + Potential_Energy
        Arr_TE.append(Total_energy)
        if step % Nfreq == 0:
            logging.debug('Step: {}'.format(step))
            logging.debug('Vels2: {}'.format(vels2sum))
            logging.debug('temperature: {}'.format(md.temp))
            logging.debug('pressure: {}'.format(Pressure))
            logging.debug('pe: {}'.format(Potential_Energy))
            logging.debug('ke: {}'.format(Kinetic_Energy))
            logging.debug('te: {}'.format(Total_energy))
            logging.debug('\n')
    print("Готово.")