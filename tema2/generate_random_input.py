import re
import os
import sys
import random

if len(sys.argv) == 1:
    print('Specify size of the matrix. Choose from "verysmall", "small", "medium", "large", "verylarge"')
    print('e.g.: %s small' % os.path.basename(sys.argv[0]))
    exit(1)

sizes = {
    "verysmall": (5, 10),
    "small": (10, 20),
    "medium": (20, 50),
    "large": (50, 100),
    "verylarge": (100, 1000)
}
if sys.argv[1].lower() not in sizes:
    print('Invalid matrix size. Choose from "verysmall", "small", "medium", "large", "verylarge", or a specific int')
    print('e.g.: %s small' % os.path.basename(sys.argv[0]))
    exit(2)

if re.match('^[0-9]+$', sys.argv[1]):
    n = int(sys.argv[1])
else:
    min_size = sizes[sys.argv[1]][0]
    max_size = sizes[sys.argv[1]][1]
    n = random.randint(min_size, max_size)

if len(sys.argv) > 2 and re.match('^[0-9]+$', sys.argv[2]):
    seed = int(sys.argv[2])
    print('using fixed seed...', seed)
else:
    seed = random.randint(1000, 9999)
    print('using random seed...', seed)
random.seed(seed)

matrix = []
for i in range(n):
    matrix.append([])
    for j in range(n):
        if j < i:
            matrix[-1].append(matrix[j][i])
        elif j == i:
            matrix[-1].append(0)
        else:
            matrix[-1].append(random.randint(1, 99))

fp = os.path.join('inputs', 'n%d_seed%s.json' % (n, seed))
if not os.path.isdir('inputs'):
    os.makedirs('inputs')
with open(fp, 'wt') as f:
    f.write('{\n')
    f.write('  "n": %d,\n' % n)
    f.write('  "matrix": [\n')
    for idx, line in enumerate(matrix):
        f.write('    ' + str(line))
        if idx != n - 1:
            f.write(',')
        f.write('\n')
    f.write('  ]\n')
    f.write('}')

print('Generated TSP matrix with size n=%d at %s, using the seed=%d' % (n, fp, seed))
