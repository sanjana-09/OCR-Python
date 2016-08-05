from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy

from assignment import *
#maximum length and width of bounding box from training data to which all letters need to be padded
max_size = get_max_dimensions(coordinates)

#get all letters in padded form from all training pages
train1 = get_letters(page1,page_details1[:,1:5].astype(int),max_size)
train2 = get_letters(page2,page_details2[:,1:5].astype(int),max_size)
train3 = get_letters(page3,page_details3[:,1:5].astype(int),max_size)
train4 = get_letters(page4,page_details4[:,1:5].astype(int),max_size)

#get all letters in padded form from all test pages
test_data1 = get_letters(test_page1,test_coordinates1,max_size)
test_data1_1 = get_letters(test_page1_1,test_coordinates1,max_size)
test_data1_2 = get_letters(test_page1_2,test_coordinates1,max_size)
test_data1_3 = get_letters(test_page1_3,test_coordinates1,max_size)
test_data1_4 = get_letters(test_page1_4,test_coordinates1,max_size)

test_data2 = get_letters(test_page2,test_coordinates2,max_size)
test_data2_1 = get_letters(test_page2_1,test_coordinates2,max_size)
test_data2_2 = get_letters(test_page2_2,test_coordinates2,max_size)
test_data2_3 = get_letters(test_page2_3,test_coordinates2,max_size)
test_data2_4 = get_letters(test_page2_4,test_coordinates2,max_size)

#stack all letters from all training pages
train_data = np.vstack((train1,train2,train3,train4))

#get pca data
axes = get_pca_axes(train_data)

pca_train_data = get_pca_data(train_data,axes,train_data)
pca_test_data1 = get_pca_data(test_data1,axes,train_data)
pca_test_data2 = get_pca_data(test_data2,axes,train_data)

pca_test_data1_1 = get_pca_data(test_data1_1,axes,train_data)
pca_test_data1_2 = get_pca_data(test_data1_2,axes,train_data)
pca_test_data1_3 = get_pca_data(test_data1_3,axes,train_data)
pca_test_data1_4 = get_pca_data(test_data1_4,axes,train_data)

pca_test_data2_1 = get_pca_data(test_data2_1,axes,train_data)
pca_test_data2_2 = get_pca_data(test_data2_2,axes,train_data)
pca_test_data2_3 = get_pca_data(test_data2_3,axes,train_data)
pca_test_data2_4 = get_pca_data(test_data2_4,axes,train_data)


def clean_data(feature_range,number_features,sp_range):
	print_message("clean_data",number_features,sp_range)

	score_test1,classified_label_1 = classify(pca_train_data,train_labels,pca_test_data1,test_labels1, feature_range)
	score_test2,classified_label_2 = classify(pca_train_data,train_labels,pca_test_data2,test_labels2, feature_range)

	corrected_score1 = calculate_improvement(classified_label_1,test_labels1,test_details1)
	corrected_score2 = calculate_improvement(classified_label_2,test_labels2,test_details2)

	print 'classifier score for test 1 before error correction'
	print score_test1
	print 'classifier score for test 1 after error correction'
	print corrected_score1

	print 'classifier score for test 2 before error correction'
	print score_test2
	print 'classifier score for test 2 after error correction'
	print corrected_score2

def noisy_data_1(feature_range,number_features,sp_range):
	print_message("noisy data 1",number_features,sp_range)

	score_test1_1,classified_label_1_1 = classify(pca_train_data,train_labels,pca_test_data1_1,test_labels1, feature_range)
	print 'classifier score for test 1.1 before error correction'
	print score_test1_1

	score_test1_2,classified_label_1_2 = classify(pca_train_data,train_labels,pca_test_data1_2,test_labels1, feature_range)
	print 'classifier score for test 1.2 before error correction'
	print score_test1_2

	score_test1_3,classified_label_1_3 = classify(pca_train_data,train_labels,pca_test_data1_3,test_labels1, feature_range)
	print 'classifier score for test 1.3 before error correction'
	print score_test1_3

	score_test1_4,classified_label_1_4 = classify(pca_train_data,train_labels,pca_test_data1_4,test_labels1, feature_range)
	print 'classifier score for test 1.4 before error correction'
	print score_test1_4


	corrected_score1_1 = calculate_improvement(classified_label_1_1,test_labels1,test_details1)
	print 'classifier score for test 1.1 after error correction'
	print corrected_score1_1

	corrected_score1_2 = calculate_improvement(classified_label_1_2,test_labels1,test_details1)
	print 'classifier score for test 1.2 after error correction'
	print corrected_score1_2

	corrected_score1_3 = calculate_improvement(classified_label_1_3,test_labels1,test_details1)
	print 'classifier score for test 1.3 after error correction'
	print corrected_score1_3

	corrected_score1_4 = calculate_improvement(classified_label_1_4,test_labels1,test_details1)
	print 'classifier score for test 1.4 after error correction'
	print corrected_score1_4

def noisy_data_2(feature_range,number_features,sp_range):

	print_message("noisy data 2",number_features,sp_range)

	score_test2_1,classified_label_2_1 = classify(pca_train_data,train_labels,pca_test_data2_1,test_labels2, feature_range)
	print 'classifier score for test 2.1 before error correction'
	print score_test2_1

	score_test2_2,classified_label_2_2 = classify(pca_train_data,train_labels,pca_test_data2_2,test_labels2, feature_range)
	print 'classifier score for test 2.2 before error correction'
	print score_test2_2

	score_test2_3,classified_label_2_3 = classify(pca_train_data,train_labels,pca_test_data2_3,test_labels2, feature_range)
	print 'classifier score for test 2.3 before error correction'
	print score_test2_3

	score_test2_4,classified_label_2_4 = classify(pca_train_data,train_labels,pca_test_data2_4,test_labels2, feature_range)
	print 'classifier score for test 2.4 before error correction'
	print score_test2_4

	corrected_score2_1 = calculate_improvement(classified_label_2_1,test_labels2,test_details2)
	print 'classifier score for test 2.1 after error correction'
	print corrected_score2_1


	corrected_score2_2 = calculate_improvement(classified_label_2_2,test_labels2,test_details2)
	print 'classifier score for test 2.2 after error correction'
	print corrected_score2_2


	corrected_score2_3 = calculate_improvement(classified_label_2_3,test_labels2,test_details2)
	print 'classifier score for test 2.3 after error correction'
	print corrected_score2_3


	corrected_score2_4 = calculate_improvement(classified_label_2_4,test_labels2,test_details2)
	print 'classifier score for test 2.4 after error correction'
	print corrected_score2_4 

def trial(feature_range,number_features,sp_range):
	# clean_data(feature_range,number_features,sp_range)
	noisy_data_1(feature_range,number_features,sp_range)
	noisy_data_2(feature_range,number_features,sp_range)

def print_message(trial,number_features,sp_range):
	print 'Showing results for ' + trial + ' using ' + number_features + ' features from ' + sp_range

trial (xrange(1,40),"39","1 to 40")
# trial (xrange(1,11),"10","1 to 11")