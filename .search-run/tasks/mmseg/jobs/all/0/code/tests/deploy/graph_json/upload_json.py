import pavi
import argparse


parser = argparse.ArgumentParser(description="pavi username and password")
parser.add_argument('--user', required=True)
parser.add_argument('--password', required=True)
args = parser.parse_args()
pavi.login(args.user, args.password)
writer = pavi.SummaryWriter('test_graph')
writer.add_graph_file('test.json')
print('Upload Successfully')
