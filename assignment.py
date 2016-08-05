from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy

# reading in the pages
page1 = np.load('train1.npy')
page2 = np.load('train2.npy')
page3 = np.load('train3.npy')
page4 = np.load('train4.npy')

height = page1.shape[0]

# reading in the .dat files containing labels, coordinates of bounding
# boxes, boolean values
page_details1 = np.loadtxt('train1.v2.dat', dtype=str)
page_details2 = np.loadtxt('train2.dat', dtype=str)
page_details3 = np.loadtxt('train3.dat', dtype=str)
page_details4 = np.loadtxt('train4.dat', dtype=str)

# stacking the dat files of the training data
page_details = np.vstack(
	(page_details1, page_details2, page_details3, page_details4))

# bounding box coordinates for training data
coordinates = page_details[:, 1:5].astype(int)

# labels for the training data, and reshaping to horizontal vector
train_labels = page_details[:, 0].reshape((1, page_details[:, 0].shape[0]))

# loading in test pages and dat files information !! need to add more
test_page1 = np.load('test1.npy')
test_page1_1 = np.load('test1.1.npy')
test_page1_2 = np.load('test1.2.npy')
test_page1_3 = np.load('test1.3.npy')
test_page1_4 = np.load('test1.4.npy')
test_details1 = np.loadtxt('test1.dat', dtype=str)

# extracting coordinates of bounding boxes and labels (reshaped) for test data 
test_coordinates1 = test_details1[:, 1:5].astype(int)
test_labels1 = test_details1[:, 0].reshape((1, test_details1[:, 0].shape[0]))

#Second test page
test_page2 = np.load('test2.npy')
test_page2_1 = np.load('test2.1.npy')
test_page2_2 = np.load('test2.2.npy')
test_page2_3 = np.load('test2.3.npy')
test_page2_4 = np.load('test2.4.npy')
test_details2 = np.loadtxt('test2.dat', dtype=str)

# extracting coordinates of bounding boxes
test_coordinates2 = test_details2[:, 1:5].astype(int)
test_labels2 = test_details2[:, 0].reshape((1, test_details2[:, 0].shape[0]))

#loading in dictionaries
# dictionary = np.loadtxt('wordsEn.txt', dtype=str)
freq_dictionary = np.loadtxt('dictionary.txt', dtype=str)

def get_coords(index, coordinates):
	"""
	returns the top,bottom,left, and right coordinates of the bounding boxes from the coordinates array
	@param index: line number
	@param coordinates: coordinates of bounding boxes
	"""
	coords_top = height-coordinates[index, 3]
	coords_bottom = height-coordinates[index, 1]
	coords_left = coordinates[index, 0]
	coords_right = coordinates[index, 2]
	return coords_top, coords_bottom, coords_left, coords_right


def get_max_dimensions(coordinates):
	"""
	finds the maximum length and maximum width of all bounding boxes used in the training data 
	and returns their values
	@param coordinates: coordinates of bounding boxes
	"""
	lengths = []
	widths = []
	# coordinates.shape[0] returns the total number of letters used
	for i in xrange(coordinates.shape[0]):

		# get coordinates of bounding boxes for every letter
		coords_top, coords_bottom, coords_left, coords_right = get_coords(
			i, coordinates)
		length = coords_bottom - coords_top
		width = coords_right - coords_left
		lengths.append(length)
		widths.append(width)
	return max(lengths), max(widths)  # returns maximum lengths and widths


def pad_letter(org_letter, (new_rows, new_cols)):
	org_rows, org_col = org_letter.shape

	"""
	finds the maximum length and maximum width of all bounding boxes used in the training data 
	and returns their values
	@param org_letter: original letter from before padding
	@param new_rows: number of rows of desired size
	@param new_cols: number of columns of desired size
	"""

	# rows and columns to be taken from the original letter are equal to the minimum of the original
	# number and the new number
	letter_rows = min(org_rows, new_rows)
	letter_cols = min(org_col, new_cols)

	# 2d numpy array filled with zeroes (black pixels)
	padded = np.zeros((new_rows, new_cols))
	padded[:letter_rows, :letter_cols] = org_letter[:letter_rows, :letter_cols]
	return padded

def get_letters(page, coordinates, max_size):
	"""
	extracts all letters from a given page and pads them to an equal size
	@param page: page containing letters
	@param coordinates: coordinates of bounding boxes
	@param max_size: desired size of box
	"""
	letters = []
	# coordinates.shape[0] returns the total number of letters used
	for i in xrange(coordinates.shape[0]):
		# get coordinates of bounding boxes for every letter
		coords_top, coords_bottom, coords_left, coords_right = get_coords(
			i, coordinates)
		org_letter = page[coords_top:coords_bottom, coords_left:coords_right]
		padded_letter = pad_letter(org_letter, max_size)
		# flattens 2d array of a letter and appends to list
		letters.append(padded_letter.ravel())
	return np.array(letters)  # returns numpy array of all letters

def classify(train, train_labels, test, test_labels, features=None):
	"""
	performs classification using nearest neighbour classification, code from completed lab sheet
	@param train: training data
	@param train_labels: labels of the training data
	@param test: test data
	@param test_labels: labels of the test data
	@param features: range of features
	"""
   # Use all feature if no feature parameter has been supplied
	if features is None:
		features = np.arange(0, train.shape[1]) 

	# Select the desired features from the training and test data
	train = train[:, features]
	test = test[:, features]

	# Super compact implementation of nearest neighbour
	x = np.dot(test, train.transpose())
	modtest = np.sqrt(np.sum(test*test, axis=1))
	modtrain = np.sqrt(np.sum(train*train, axis=1))
	dist = x/np.outer(modtest, modtrain.transpose())  # cosine distance
	nearest = np.argmax(dist, axis=1)
	mdist = np.max(dist, axis=1).astype(int)
	label = train_labels[0, nearest]
	# print nearest
	# print label
	score = (100.0 * sum(test_labels[0, :] == label))/label.shape[0]
	return score, label

# returns principal component axes, code from completed lab sheet
def get_pca_axes(data,max_size):

	covx = np.cov(data, rowvar=0)
	N = covx.shape[0]
	w, v = scipy.linalg.eigh(covx, eigvals=(N-100, N-1))
	v = np.fliplr(v)
	axis = plt.subplot(2,2,1)
	plt.imshow(np.reshape(v[:, 0], (max_size), order='F'), cmap=cm.Greys_r)
	plt.show()
	axis = plt.subplot(2,2,2)
	axis.matshow(np.reshape(v[:, 1], (max_size), order='F'), cmap=cm.Greys_r)
	axis = plt.subplot(2,2,3)
	axis.matshow(np.reshape(v[:, 2], (max_size), order='F'), cmap=cm.Greys_r)
	axis = plt.subplot(2,2,4)
	axis.matshow(np.reshape(v[:, 3], (max_size), order='F'), cmap=cm.Greys_r)
	return v

# returns pca data, code from completed lab sheet
def get_pca_data(data, axes, train_data):
	pca_data = np.dot((data - np.mean(train_data)), axes) 
	return pca_data


def make_words(classified_labels, actual_labels):
	"""
	concatenates labels to form words using the boolean value indicating end of word
	in the dat file for test data
	@param classified_labels: labels from classifier
	@param actual_labels: actual_labels given
	"""
	words = []
	word = ''
	for i in xrange(classified_labels.shape[0]):  # labels.shape[0] = number of labels
		word += classified_labels[i]
		# boolean value 1 indicates end of word
		if (actual_labels[i, 5]) == '1':
			words.append(word)
			word = ''
	return words


def correct_errors(words):
	"""
	Error correction is performed by checking whether each word as outputted by the classifier is 
	present in the dictionary. If it is not, the algorithm searches for the words closest 
	(edit distance of 1) to the incorrect word in the dictionary, chooses the best one and replaces it.
	Returns list of replaced words
	@param words: words from classifier
	"""
	new_words = []
	for word in words:
		edit_distance_1_words = []
		edit_distance_2_words = []
		dictionary_word_found = False

		if not (word.lower() in freq_dictionary): #if the word isn't in dictionary or is a proper noun
			for dictionary_word,frequency in freq_dictionary:
				if len(dictionary_word) == len(word):
					if edit_distance(dictionary_word, word.lower()) == 1:
						edit_distance_1_words.append(dictionary_word)
					else:
						if edit_distance(dictionary_word,word.lower()) == 2:
							edit_distance_2_words.append(dictionary_word)

			if edit_distance_1_words:
				final_word = pick_best_word(edit_distance_1_words)
				dictionary_word_found = True
			else:
				if edit_distance_2_words:
					final_word = pick_best_word(edit_distance_2_words)
					dictionary_word_found = True

			if dictionary_word_found:
				if word[0].isupper():
					new_words.append(final_word.capitalize())
				else:
					new_words.append(final_word)
			else:
				new_words.append(word)
		else:
			new_words.append(word)

	for i in xrange(len(words)):
 	 	if words[i]  != new_words[i]:
 			print words[i],new_words[i]

	return new_words


def edit_distance(word1, word2):
	"""
	returns difference between a word containing a misclassified label and a dictionary word of the 
	same length
	@param dictionary_word: word found from dictionary
	@param word: word from classifier
	"""
	diff = 0
	for i in xrange(len(word1)):
		if word1[i] != word2[i]:
			# difference increased by 1 for every character that differs
			# between the two words
			diff += 1
	return diff

def pick_best_word(word_list):
	frequencies = []
	words_with_frequencies = []

	for word in word_list:
		if word in freq_dictionary:
			word,frequency = freq_dictionary[np.where(freq_dictionary == word)[0][0]]
			frequencies.append(int(frequency))
			words_with_frequencies.append((word,frequency))
	if not frequencies:
		return word_list[0]
	else:
		index_of_best_word = frequencies.index(max(frequencies))
		return words_with_frequencies[index_of_best_word][0]


def turn_to_labels(corrected_words):
	"""
	seperates a list of individual words into characters and returns the remaining list
	@param corrected_words: list of corrected words
	"""
	corrected_labels = []
	for corrected_word in corrected_words:
		for label in corrected_word:
			corrected_labels.append(label)
	return corrected_labels


def format(list):
	"""
	converts given list to 2d numpy array
	@param list: given list 
	"""
	return np.array(list).reshape((1, len(list)))


def calculate_score(correct_labels, classified_labels):
	"""
	calculates new score after error correction, using the corrected labels,
	comparing them with the actual labels
	@param correct_labels: expected labels
	@param classified_labels: labels from classifier
	"""
	new_score = (100.0 * sum(correct_labels[0, :] == classified_labels[0, :]))/classified_labels.shape[1]
	return new_score


def calculate_improvement(classified_labels, correct_labels, test_details):
	"""
	does error correction and returns the score
	@param correct_labels: expected labels 
	@param classified_labels: labels from classifier
	@param test_details: .dat file for test data used for end of word boolean value
	"""
	words = make_words(classified_labels, test_details)
	corrected_words = correct_errors(words)
	new_labels = turn_to_labels(corrected_words)
	corrected_labels = format(new_labels)
	new_score = calculate_score(correct_labels, corrected_labels)
	return new_score
