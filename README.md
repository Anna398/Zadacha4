import numpy
import tools
import numpy.fft as fft
import matplotlib.pyplot as plt

class GaussianMod:
    ''' Класс с уравнением плоской волны для модулированного гауссова сигнала в дискретном виде
    d - определяет задержку сигнала.
    w - определяет ширину сигнала.
    Nl - количество ячеек на длину волны.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    mu - относительная магнитная проницаемость среды, в которой расположен источник.
    '''

    def __init__(self, d, w, Nl, eps=1.0, mu=1.0, Sc=1.0):
        self.d = d
        self.w = w
        self.Nl = Nl
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getField(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        '''
        return (numpy.sin(2 * numpy.pi / self.Nl * (q * self.Sc - m * numpy.sqrt(self.eps * self.mu))) *
                numpy.exp(-(((q - m * numpy.sqrt(self.eps * self.mu) / self.Sc) - self.d) / self.w) ** 2))

if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Число Куранта
    Sc = 1.0

    #Скорость света
    c = 3e8

    # Время расчета в отсчетах
    maxTime = 4000

    #Размер области моделирования в метрах
    X = 1.0

    #Размер ячейки разбиения
    dx = 1e-3

    # Размер области моделирования в отсчетах
    maxSize = int(X / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    # Положение источника в отсчетах
    sourcePos = 50

    # Датчики для регистрации поля
    probesPos = [25,75]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    #1й слой диэлектрика
    eps1 = 4.5
    d1 = 0.17
    layer_1 = int(maxSize / 2) + int(d1 / dx)

    #2й слой диэлектрика
    eps2 = 3.5
    d2 = 0.08
    layer_2 = layer_1 + int(d2 / dx)

    #3й слой диэлектрика
    eps3 = 9.1
    d3 = 0.02
    layer_3 = layer_2 + int(d3 / dx)

    #3й слой диэлектрика
    eps4 = 7.3

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)
    eps[int(maxSize/2):layer_1] = eps1
    eps[layer_1:layer_2] = eps2
    eps[layer_2:layer_3] = eps3
    eps[layer_3:] = eps4
    
    # Магнитная проницаемость
    mu = numpy.ones(maxSize - 1)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    # Максимальная и манимальная частоты для отображения
    # графика зависимости коэффициента отражения от частоты
    Fmin = 4e9
    Fmax = 8e9

    # Параметры возбуждающего сигнала - модулированного гауссова импульса
    A0 = 100
    Amax = 100
    f0 = (Fmax + Fmin) / 2
    N1 = 1 / f0
    DeltaF = Fmax - Fmin

    wg = 2 * numpy.sqrt(numpy.log(Amax)) / (numpy.pi * DeltaF)
    dg = wg * numpy.sqrt(numpy.log(A0))

    wg = wg / dt
    dg = dg / dt
    N1 = Sc * N1 / dt

    source = GaussianMod(dg, wg, N1, eps[sourcePos], mu[sourcePos])
    
    # Коэффициенты для расчета ABC второй степени
    # Sc' для левой границы
    Sc1Left = Sc / numpy.sqrt(mu[0] * eps[0])

    k1Left = -1 / (1 / Sc1Left + 2 + Sc1Left)
    k2Left = 1 / Sc1Left - 2 + Sc1Left
    k3Left = 2 * (Sc1Left - 1 / Sc1Left)
    k4Left = 4 * (1 / Sc1Left + Sc1Left)

    # Sc' для правой границы
    Sc1Right = Sc / numpy.sqrt(mu[-1] * eps[-1])

    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)

    # Ez[0: 2] в предыдущий момент времени (q)
    oldEzLeft1 = numpy.zeros(3)

    # Ez[0: 2] в пред-предыдущий момент времени (q - 1)
    oldEzLeft2 = numpy.zeros(3)

    # Ez[-3: -1] в предыдущий момент времени (q)
    oldEzRight1 = numpy.zeros(3)

    # Ez[-3: -1] в пред-предыдущий момент времени (q - 1)
    oldEzRight2 = numpy.zeros(3)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel,dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])
    display.drawBoundary(int(maxSize / 2))
    display.drawBoundary(layer_1)
    display.drawBoundary(layer_2)
    display.drawBoundary(layer_3)

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getField(0, q)

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getField(-0.5, q + 0.5))

        # Граничные условия ABC второй степени (слева)
        Ez[0] = (k1Left * (k2Left * (Ez[2] + oldEzLeft2[0]) +
                           k3Left * (oldEzLeft1[0] + oldEzLeft1[2] - Ez[1] - oldEzLeft2[1]) -
                           k4Left * oldEzLeft1[1]) - oldEzLeft2[2])

        oldEzLeft2[:] = oldEzLeft1[:]
        oldEzLeft1[:] = Ez[0: 3]

        # Граничные условия ABC второй степени (справа)
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])

        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 10 == 0:
            display.updateData(display_field, q)

    display.stop()
    

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    # Максимальная и манимальная частоты для отображения
    # графика зависимости коэффициента отражения от частоты
    Fmin = 4e9
    Fmax = 8e9
    
    # Размер массива для ПФ
    size = 2 ** 18

    # Выдедение падающего поля 
    FallField = numpy.zeros(maxTime)
    FallField[:600] = probes[1].E[:600]

    # Нахождение БПФ падающего поля
    FallSpectr = abs(fft.fft(FallField, size))
    FallSpectr = fft.fftshift(FallSpectr)

    # Нахождение БПФ отраженного поля
    ScatteredSpectr = abs(fft.fft(probes[0].E, size))
    ScatteredSpectr = fft.fftshift(ScatteredSpectr)

    # шаг по частоте и определение частотной оси
    df = 1 / (size * dt)
    f = numpy.arange(-(size / 2) * df, (size / 2) * df, df)

    # Построение спектра падающего и рассеянного поля
    plt.figure()
    plt.plot(f * 1e-9, FallSpectr / numpy.max(FallSpectr))
    plt.plot(f * 1e-9, ScatteredSpectr / numpy.max(FallSpectr))
    plt.grid()
    plt.xlim(4e9 * 1e-9, 8e9 * 1e-9)
    plt.xlabel('f, ГГц')
    plt.ylabel('|S/Smax|')
    plt.legend(['Спектр падающего поля', 'Спектр отраженного поля'], loc=1)

    # Определение коэффициента отражения и построения графика
    plt.figure()
    plt.plot(f * 1e-9, (ScatteredSpectr / FallSpectr))
    plt.xlim(Fmin * 1e-9, Fmax * 1e-9)
    plt.ylim(0, 1)
    plt.grid()
    plt.xlabel('f, ГГц')
    plt.ylabel('|Г|')
    plt.show()
