# ====================================	#
# Quick Draw DNN Classifier 			#
#										#
# Nicolau Tahan 		18/05/2018		#
# ====================================	#

import numpy as np
import tensorflow as tf


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


# Classe definida para juntar a label a uma imagem
class LabeledImage():
	def __init__(self, img, label):
		self.img = img
		self.label = label

# Funcao pra carregar as imagens 28x28 de numpys arquivos
def load_data():
	basic_path = 'full-numpy_bitmap-'
	labels_list = ['airplane', 'apple', 'bicycle']	# Tipos de desenho

	# Define um id de classe para cada desenho
	label_index = {
		'airplane'	: [2],
		'apple'		: [1],
		'bicycle'	: [0]
	}

	
	# Quantidade de cada desenho pra pegar
	limit = 100

	data_obj_list = []
	for label in labels_list:
		data = np.load(basic_path + label + '.npy')
		i = 0
		print("Carregando " + label + ' data')
		printProgressBar(i, limit, length= 20)
		for img in data:

			# Cria um objeto LabeledImage para unir a label e a imagem
			aux_obj = LabeledImage(img, label_index[label])
			data_obj_list.append(aux_obj)

			i = i + 1
			printProgressBar(i, limit, length= 20)
			if i >= limit:
				break

	np.random.shuffle(data_obj_list)

	return data_obj_list

# INPUT FUNCTIONS
"""	O esquema e que todos os metodos do estimador (train, evaluate, estimate)
	precisam de uma input function para interpretar os dados.

	Input functions sao funcoes que retornam ou uma tuple ou um objeto da 
	classe tf.data.Dataset que contem o set de (features, label). Ou seja
	um Tensor com as imagens e um Tensor com as labels respectivas.

	Nas funcoes abaixo uso um Dataset. Ele eh de dimensoes ((img:[28 28]), (label : [1]))
"""
def train_input_fn(obj_list, batch_size):
	features_dict = {'img' : []}
	labels_list = []

	for img_obj in obj_list:
		img_tensor = tf.convert_to_tensor(img_obj.img)
		img_tensor = tf.reshape(img_tensor, [28, 28])

		features_dict['img'].append(img_tensor)
		labels_list.append(img_obj.label)

	train_ds = tf.data.Dataset.from_tensor_slices((features_dict, labels_list))
	train_ds = train_ds.shuffle(100).repeat().batch(batch_size)

	return train_ds

def test_input_fn(obj_list, batch_size):
	features_dict = {'img' : []}
	labels_list = []

	for img_obj in obj_list:
		img_tensor = tf.convert_to_tensor(img_obj.img)
		img_tensor = tf.reshape(img_tensor, [28, 28])

		features_dict['img'].append(img_tensor)
		labels_list.append(img_obj.label)

	test_ds = tf.data.Dataset.from_tensor_slices((features_dict, labels_list))
	test_ds = test_ds.batch(batch_size)

	return test_ds


def main(args):
	data_obj_list = load_data()

	# Os dados sao dividos da forma: 70% para treinamento
	#								 20% para validacao
	#								 10% para mostrar
	corte_a = int(0.7*len(data_obj_list))
	corte_b = corte_a + int(0.2*len(data_obj_list))

	train_obj_list = data_obj_list[0 : corte_a]
	test_obj_list = data_obj_list[corte_a : corte_b]
	predict_obj_list = data_obj_list[corte_b : len(data_obj_list)]

	# FEATURE COLUMN
	""" Entao o esquema da feature column e que o estimador precisa saber como ler o Dataset que input funciton retorna
		entao precisa criar esse vetor. 

		A feature column e um vetor que, para cada feature do input, ele define a key do dicionario e as dimensoes (shape)
		desses tensores. No caso dos desenhos e apenas a imagem [28 28]. Mas poderia-se utilizar outros como media (shape [1])
		e qlq outro input com qualquer forma
	"""
	feature_column = [tf.feature_column.numeric_column(key = 'img', shape = [28, 28])]

	dnn_classifier = tf.estimator.DNNClassifier(
					feature_columns = feature_column,
					hidden_units = [16, 16],
					model_dir= '/models/quick_dnnc',
					n_classes= 3)

	
	train_spec = tf.estimator.TrainSpec(input_fn= lambda:train_input_fn(train_obj_list, 10), max_steps= 1000)
	test_spec = tf.estimator.EvalSpec(input_fn= lambda:test_input_fn(test_obj_list, 10))

	tf.estimator.train_and_evaluate(dnn_classifier, train_spec, test_spec)

	predictions = dnn_classifier.predict(input_fn= lambda:test_input_fn(predict_obj_list, 10))

	print('')
	print('')
	print('Tamanho dos Datasets:')
	print('  Treinamento\t=> ' + str(len(train_obj_list)))
	print('  Teste\t\t=> ' + str(len(test_obj_list)))

	for result ,expected in zip(predictions, predict_obj_list):
		aux_labels = ['Aviao', 'Maca', 'Bike']
		
		result_index = result['class_ids'][0]
		expected_index = expected.label[0]

		prop = result['probabilities'][result_index] * 100
		prop = str(prop)[0:4]

		print('Resultado = ' + aux_labels[result_index] + '\t Esperado = ' + aux_labels[expected_index] + '\t(' + str(prop) + ')')

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()