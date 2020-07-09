import numpy as np

import torch




def get_memory(tensor, name = "", show = True):
	"""
	Afficher / Renvoie l'usage mémoire d'un Tenseur
	"""
	m = tensor.element_size()*tensor.nelement()
	if show : print(name , "Memory Size :" , convert_bytes(m))

	return m


def categorical(y, num_classes = 10):
	y = y.numpy()
	n = y.shape[0]
	categorical = np.zeros((n, num_classes), dtype='int64')
	categorical[np.arange(n), y] = 1
	output_shape = (n, num_classes)
	categorical = np.reshape(categorical, output_shape)
	return torch.tensor(categorical)


def get_shapes(net):
	"""
	Afficher les tailles des couches du modèle.
	"""
	for key, val in net.state_dict().items():
		print(key, val.shape)



def get_number_of_parameters(model, show = True):
	"""
	Affiche / Renvoie le nombre de paramètre d'un modèle
	"""
	nb = sum(p.numel() for p in model.parameters() )#if p.requires_grad)
	if show: 
		print("Number of parameters", nb)
	return nb

def get_model_size(model, show = True):
	"""
	Affiche / Renvoie l'usage mémoire d'un modèle
	"""
	nb = get_number_of_parameters(model, False)
	size = nb*4
	if show:
		print("Model size :", convert_bytes(size, False))
	return size


def convert_bytes(size, isbytes = True):
	if isbytes :
		b = 1000.0
	else:
		b = 1024.0		
	for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
	   if size < b:
		   return "%3.1f %s" % (size, x)
	   size /= b

	return size