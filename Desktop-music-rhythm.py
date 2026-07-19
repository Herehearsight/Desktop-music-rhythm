import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import wave
import tkinter as tk

# 音频配置
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 4096
WAVE_OUTPUT_FILENAME = 'audio_output.wav'
# 透明专用色，画布/窗口统一用这个颜色，系统会穿透该颜色
TRANSPARENT_BG = "#000001"
CMAP_NAME = 'rainbow' # 渐变颜色系
Alpha_page = 1
DOWNSAMPLE = 16   # x轴密度
Y_SPLIT = 2000    # y轴幅度分割点
Y_MAX_RAW = 2800
Y_LIM_RAW = 12000
Dlta_max = Y_MAX_RAW - Y_SPLIT
Dlta_lim = Y_LIM_RAW - Y_SPLIT


# 获取立体声混音内录设备
def findInternalRecordingDevice(p):
	target = '立体声混音'
	for i in range(p.get_device_count()):
		devInfo = p.get_device_info_by_index(i)
		if devInfo['name'].find(target) >= 0 and devInfo['hostApi'] == 0:
			return i
	print('无法找到内录设备!')
	return -1

class AudioWaveTK:
    def __init__(self, root):
        self.window = root
        self.window.title("实时音频波形")
        self.window.attributes("-topmost", 1)
        # ========== 删除全局窗口alpha，改用颜色穿透 ==========
        # self.window.attributes('-alpha', 0.9) 删掉这行
        self.window.attributes('-toolwindow', 1)
        self.window.overrideredirect(True)  # 移除标题栏边框

        # Windows专属：指定哪种颜色完全穿透桌面
        self.window.attributes("-transparentcolor", TRANSPARENT_BG)
        # 窗口底色设为穿透色
        self.window.config(bg=TRANSPARENT_BG)

        # 无边框窗口拖动变量
        self.x_offset = None
        self.y_offset = None
        self.window.bind("<ButtonPress-1>", self.start_move)
        self.window.bind("<ButtonRelease-1>", self.stop_move)
        self.window.bind("<B1-Motion>", self.do_move)

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        self.window.geometry(f'{int(screen_w)}x{int(screen_h*0.2)}+0+{int(screen_h*0.8)}')
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        # 音频相关变量
        self.audio = pyaudio.PyAudio()
        self.dev_index = findInternalRecordingDevice(self.audio)
        self.stream = None
        self.wave_file = None
        self.frames = []

        # 初始化绘图
        self.init_plot()
        self.open_audio_stream()
        self.update_waveform()

    # 无边框窗口拖动函数
    def start_move(self, event):
        self.x_offset = event.x
        self.y_offset = event.y

    def stop_move(self, event):
        self.x_offset = None
        self.y_offset = None

    def do_move(self, event):
        deltax = event.x - self.x_offset
        deltay = event.y - self.y_offset
        x = self.window.winfo_x() + deltax
        y = self.window.winfo_y() + deltay
        self.window.geometry(f"+{x}+{y}")

    def init_plot(self):
        self.fig, self.ax = plt.subplots(dpi=100)
        self.show_len = CHUNK//DOWNSAMPLE
        self.x = np.arange(0, self.show_len)  # FFT只取正频率一半
        #self.bars = self.ax.bar(self.x, np.zeros(CHUNK//DOWNSAMPLE), width=0.8, color='#00ccff', alpha=1)
        self.bars = []
        # 颜色归一化：x从0~show_len 映射到0~1色标区间
        norm = plt.Normalize(vmin=0, vmax=self.show_len)
        cmap = plt.get_cmap(CMAP_NAME)

        for xi in self.x:
            # 根据X位置取渐变颜色
            bar_color = cmap(norm(self.show_len - xi))
            bar = self.ax.bar(xi, 0, width=0.8, color=bar_color, alpha = Alpha_page)
            self.bars.append(bar[0])
            
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.fig.patch.set_facecolor(TRANSPARENT_BG)
        self.ax.set_facecolor(TRANSPARENT_BG)

        #self.fig.tight_layout(pad=0)
        for side in ['top','right','left','bottom']:
            self.ax.spines[side].set_visible(False)

        self.ax.set_xlim(0, self.show_len)
        #self.ax.set_yscale('log')
        self.ax.set_ylim(1, Y_MAX_RAW)  # 频谱幅度正数
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.config(bg=TRANSPARENT_BG)
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def open_audio_stream(self):
        if self.dev_index == -1:
            tk.messagebox.showerror("设备错误", "未检测到立体声混音设备，程序退出")
            self.window.destroy()
            return

        self.stream = self.audio.open(
            input_device_index=self.dev_index,
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        self.wave_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        self.wave_file.setnchannels(CHANNELS)
        self.audio.get_sample_size(FORMAT)
        self.wave_file.setframerate(RATE)

    def update_waveform(self):
        try:
            raw_data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
            self.frames.append(raw_data)
            
            # FFT计算频谱幅度
            fft_vals = np.fft.fft(audio_data)
            fft_amp = np.abs(fft_vals[:CHUNK//2])
            # 缩放适配显示
            fft_amp = fft_amp / 100

            for bar, val in zip(self.bars, fft_amp):
                if val > Y_SPLIT :
                    val = Y_SPLIT + (val - Y_SPLIT)* Dlta_max /Dlta_lim
                
                bar.set_height(val)

            self.canvas.draw()
        except Exception as e:
            print(f"音频读取异常: {e}")

        self.window.after(1, self.update_waveform)

    def on_close(self):
        print("正在关闭程序，保存音频文件...")
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.wave_file is not None:
            self.wave_file.writeframes(b''.join(self.frames))
            self.wave_file.close()
            print(f"音频已保存至: {WAVE_OUTPUT_FILENAME}")
        self.audio.terminate()
        self.window.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = AudioWaveTK(root)
    root.mainloop()
