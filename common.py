import progressbar
import psutil
import time

# A custom widget to track GPU/RAM while you tokenize
widgets = [
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
    progressbar.Variable('gpu_load', format=' GPU: {formatted_value}%'),
    progressbar.Variable('cpu_ram', format=' RAM: {formatted_value}%'),
]


bar = progressbar.ProgressBar(max_value=1000, widgets=widgets)
for i in bar(range(1000)):
    time.sleep(1)