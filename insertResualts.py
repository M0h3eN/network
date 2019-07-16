import os
from argparse import ArgumentParser
from commons.plotRelatedFunctions.plot_compare import *

parser = ArgumentParser(description='Save plot for estimated rate detail')

parser.add_argument('-d', '--data',  action='store',
                    dest='data', help='Raw data directory')
parser.add_argument('-w', '--write', action='store',
                    dest='write', help='Output directory')
args = parser.parse_args()

# Check for writing path existence

if not os.path.exists(args.write):
    os.makedirs(args.write)

positions = ['In']
stim = ['Stim', 'NoStim']
epochs = ['Enc', 'Mem', 'Sac']
rates = ['rate_stim', 'estimated_rate_stim']
all_window_length_status = [True, False]


# al = TRUE
for p in positions:
    for s in stim:
        for r in rates:
            fr_plot(args.data, p, s, args.write, r, al=True)
            cor_plot(args.data, p, s, args.write, r, al=True)
            scatter_plot(args.data, p, s, args.write, r, al=True)
# al = False
for p in positions:
    for s in stim:
        for r in rates:
            for e in epochs:
                fr_plot(args.data, p, s, args.write, r, e, al=False)
                cor_plot(args.data, p, s, args.write, r, e, al=False)
                scatter_plot(args.data, p, s, args.write, r, e, al=False)

