import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.normpath(os.path.join(script_dir, '..', 'img'))
os.makedirs(img_dir, exist_ok=True)

plt.plot([1,2,3])
out = os.path.join(img_dir, 'test_plot.png')
plt.savefig(out)
print('WROTE', out)
