import argparse

my_parser = argparse.ArgumentParser(description='List the parameters for training')
my_parser.add_argument('--Path',
						type=str,
						help='the path to save logs',
						default='log/3'
						)

my_parser.add_argument('--Dataset',
						type=str,
						help='Dataset to train, sind, roc, arxiv, nips',
						default='nips'
						)

my_parser.add_argument('--n_layer_sent',
						type=int,
						help='number of encoder layers in sentence encoder',
						default=2
						)

my_parser.add_argument('--n_heads',
						type=int,
						help='number of heads in multi head attentions',
						default=4
						)

my_parser.add_argument('--gendim',
						type=int,
						help='dimention of model',
						default=128
						)

my_parser.add_argument('--rnndim',
						type=int,
						help='dimention of pointer',
						default=64
						)

my_parser.add_argument('--multi_drop',
						type=float,
						help='dropout for multi head attentions',
						default=0.2
						)

my_parser.add_argument('--pointer_drop',
						type=float,
						help='dropout for pointer',
						default=0.3
						)

my_parser.add_argument('--rnn_dropout',
						type=float,
						help='dropout for pointer rnns',
						default=0.1
						)

my_parser.add_argument('--word_encoder_drop',
						type=float,
						help='dropout for after word ecnoder',
						default=0.3
						)

my_parser.add_argument('--informed_type',
						type=str,
						help='pointer type: informed or notinformed',
						default='informed'
						)


my_parser.add_argument('--batch_size',
						type=int,
						help='pointer type: informed or notinformed',
						default=8
						)
my_parser.add_argument('--val_batch_size',
						type=int,
						help='pointer type: informed or notinformed',
						default=8
						)
my_parser.add_argument('--prepath',
						type=str,
						help='pointer type: informed or notinformed',
						default=''
						)

# Execute the parse_args() method
args = my_parser.parse_args()


def save_args(args, path):
	with open(path,'w') as f:
		f.write('path:\t'+str(args.Path)+'\n')
		f.write('dataset:\t'+str(args.Dataset)+'\n')
		f.write('n_layer_sent:\t'+str(args.n_layer_sent)+'\n')
		f.write('n_heads:\t'+str(args.n_heads)+'\n')
		f.write('gendim:\t'+str(args.gendim)+'\n')
		f.write('rnndim:\t'+str(args.rnndim)+'\n')
		f.write('multi_drop:\t'+str(args.multi_drop)+'\n')
		f.write('pointer_drop:\t'+str(args.pointer_drop)+'\n')
		f.write('rnn_dropout:\t'+str(args.rnn_dropout)+'\n')
		f.write('word_encoder_drop:\t'+str(args.word_encoder_drop)+'\n')
		f.write('informed_type:\t'+str(args.informed_type)+'\n')
		f.write('batch_size:\t'+str(args.batch_size)+'\n')
		f.write('val_batch_size:\t'+str(args.val_batch_size)+'\n')