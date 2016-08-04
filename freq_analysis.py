from scipy import io
import numpy as np


#rough code to test frequency analysis
dictionary = np.loadtxt('wordsEn.txt', dtype=str)
freq_dictionary = np.loadtxt('dictionary.txt', dtype=str)

print freq_dictionary
word,frequency  = freq_dictionary[0]
print word,frequency

words = ["calf","calm","call"]
frequencies = []
word_and_frequency = []

for word in words:
	if word in freq_dictionary:
		word,frequency = freq_dictionary[np.where(freq_dictionary == word)[0][0]]
		word_and_frequency.append((word,int(frequency)))

print word_and_frequency
print word_and_frequency.index(max(word_and_frequency))
print word_and_frequency[word_and_frequency.index(max(word_and_frequency))][0]




# f = np.argsort(freq_dictionary[:,1])
# print f
# print f.shape

# fake_dict = np.array([17,197,19])

# print np.argsort(fake_dict)

# for index in f:
# 	print freq_dictionary[index]

# frequencies = []
# maybe_words = ["calf","call","calm"]

# for word in maybe_words:
# 	if word in freq_dictionary:
# 		org_index_in_freq_dictionary = np.where(freq_dictionary == word)[0][0]
# 		print word
# 		print org_index_in_freq_dictionary

# 		index_in_arr_indices = np.where(f == org_index_in_freq_dictionary)[0][0]
# 		print index_in_arr_indices

# 		frequencies.append(index_in_arr_indices)
# print frequencies
# index = max(frequencies)
# print freq_dictionary[index][0]

# row_of_word = np.where(freq_dictionary == "thinker")[0][0]

# print freq_dictionary[row_of_word][1]

# maybe_words = ["unhappy"]
# frequencies = []
# not_in_freq_dictionary = []

# for word in maybe_words:
# 	if word in freq_dictionary:
# 		index = np.where(freq_dictionary == word)
# 		row = index[0][0]

# 		frequencies.append(freq_dictionary[row][1]) #given freq in dictionary
# 		# frequencies.append(row) #ranks by index in freq_dictionary
# 	else:
# 		not_in_freq_dictionary.append(word)

# print frequencies
# print "highest rank: "
# print max(frequencies)
# print maybe_words[frequencies.index(max(frequencies))]

# # #prints out words in the dictionary at that frequency -only works if ranked by frequency and not by index
# print freq_dictionary[np.where(freq_dictionary == max(frequencies))[0][0]][0]
# # print freq_dictionary[min(frequencies)][1]

# # print 'Words not in dictionary'
# # print not_in_freq_dictionary

