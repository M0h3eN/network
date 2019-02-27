from argparse import ArgumentParser


parser = ArgumentParser(description='This is a Python program for computing Gelman-Rubin diagnostics')

parser.add_argument('-H', '--host', action='store',
                    dest='host', help='MongoDB host name')

parser.add_argument('-p', '--port',action='store',
                    dest='port', help='MongoDB port number')


args = parser.parse_args()

# Gelman-Rubin convergence statistics
from fitModel.GelmanRubin_convergence import compute_gelman_rubin_convergence
print('************ Computing GR ************')
compute_gelman_rubin_convergence(args)
