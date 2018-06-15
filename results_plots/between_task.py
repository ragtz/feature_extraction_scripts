import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'size': '30'}

matplotlib.rc('font', **font)

tasks = ['drawer', 'lamp', 'pitcher', 'bowl','average']
train_accs = [0.79, 0.78, 0.8, 0.84, 0.8]
test_accs = [0.91, 0.81, 0.79, 0.67, 0.8]

width = 0.35
train_clr = 'b'
test_clr = 'g'
ind = np.arange(len(tasks)) + width
xticks = ind + width/2

fig, ax = plt.subplots()
train_rects = ax.bar(ind, train_accs, width)
test_rects = ax.bar(ind+width, test_accs, width)

xmin, xmax = ax.get_xlim()
chance_line = ax.plot([xmin, xmax], [0.5, 0.5], ls='--', c='k', lw=5)

ax.set_title('Between Task Classification')
ax.set_xlabel('Left Out Task')
ax.set_ylabel('Accuracy')

ax.set_xticks(xticks)
ax.set_xticklabels(tasks)
ax.set_xlim([xmin, xmax])
ax.set_ylim([0, 1])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
ax.legend([train_rects[0], test_rects[0], chance_line[0]],
          ['Train', 'Test', 'Chance'],
          loc='center left',
          bbox_to_anchor=(1, 0.5))

plt.show()

