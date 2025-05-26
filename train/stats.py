import numpy as np
import matplotlib.pyplot as plt


class BatchStats:
    def __init__(self, stat_names):
        self.stats = {name: {'sum': 0.0, 'count': 0, 'running_means': []} for name in stat_names}
    
    def add(self, name, value, count=1):
        if name not in self.stats:
            raise KeyError(f"Statistic '{name}' not initialized")
        self.stats[name]['sum'] += value
        self.stats[name]['count'] += count
    
    def aggregate(self):
        results = {}
        for name, data in self.stats.items():
            if data['count'] > 0:
                mean = data['sum'] / data['count']
                data['running_means'].append(mean)
                results[name] = mean
                # Reset for next batch
                data['sum'] = 0.0
                data['count'] = 0
        return results
    
    def running_means(self, name):
        return self.stats[name]['running_means']
    
    __getitem__ = running_means         # allow access with [] to running means
    
    def last(self, name):
        return self.stats[name]['running_means'][-1]


def moving_average(data, window_size):
    window = np.ones(window_size, dtype=float) / window_size
    return np.convolve(data, window, 'valid')


class UpdatablePlot:
    def __init__(self, labels, show_last_n=100):
        plt.ion()
        
        dim = np.shape(labels)
        if len(dim) == 1:
            dim = (1,) + dim
        
        fig, axs = plt.subplots(dim[0], dim[1], figsize=(10, 6))
        fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.94)
        axs = np.ravel(axs)
        self.lineplots = []
        for ax, label in zip(axs, np.ravel(labels)):
            if not label: continue
            line1, = ax.plot([], [], 'b-', label=label)
            line2, = ax.plot([], [], 'r--', label='MA(10)')
            self.lineplots.append((line1, line2))
            ax.set_xlabel('Epoch')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.legend()
            ax.grid(True)

        self.show_last_n = show_last_n
        self.fig = fig
        self.axs = axs

    def update(self, data_seq):
        for ax, linep, data in zip(self.axs, self.lineplots, data_seq):
            if data is None: continue
            start, end = max(0, len(data) - self.show_last_n), len(data)
            linep[0].set_data(range(start, end), data[start:end])

            if len(data) > 0:
                data = np.pad(data, (9, 0), 'edge')
                ma = moving_average(data, 10)
                n = end - start
                linep[1].set_data(range(end - n, end), ma[-n:])
            else:
                linep[1].set_data([], [])

            # Adjust the plot limits
            ax.relim()
            ax.autoscale_view()

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def update_from(self, source, keys):
        self.update([source[k] for k in keys])
    
    def poll(self):
        self.fig.canvas.flush_events()

    def save(self, filename):
        self.fig.savefig(filename, bbox_inches='tight')

