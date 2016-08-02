from scipy import io
import numpy as np


#rough code to test frequency analysis
dictionary = np.loadtxt('wordsEn.txt', dtype=str)
freq_dictionary = np.loadtxt('freq.txt', dtype=str)

print dictionary.shape

maybe_words = ["this","font","foot","when","acct"]
frequencies = []
not_in_freq_dictionary = []

for word in maybe_words:
	if word in freq_dictionary:
		index = np.where(freq_dictionary == word)
		row = index[0][0]
		#frequencies.append(freq_dictionary[row][0]) #given freq in dictionary
		frequencies.append(row) #ranks by index in freq_dictionary
	else:
		not_in_freq_dictionary.append(word)

print frequencies
print "highest rank: "
print min(frequencies)

#prints out words in the dictionary at that frequency -only works if ranked by frequency and not by index
print freq_dictionary[np.where(freq_dictionary == min(frequencies))[:][0]]
print freq_dictionary[min(frequencies)][1]

print 'Words not in dictionary'
print not_in_freq_dictionary

l = ['10','10']

print min(l)

