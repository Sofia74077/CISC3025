import json
import nltk
import argparse
import re
import math 
from collections import defaultdict

CATEGORIES = ['crude', 'grain', 'money-fx', 'acq', 'earn']
nltk.download('punkt')

def preprocess(inputfile, outputfile):
    stemmer = nltk.PorterStemmer()

    with open(inputfile, 'r', encoding='utf-8') as f:
        data = json.load(f)  

    processed_data = []

    for item in data:
        if len(item) < 3:
            continue  

        file_id = item[0]       
        category = item[1]      
        text = item[2]          

        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = nltk.word_tokenize(text)
        stemmed_tokens = [stemmer.stem(t) for t in tokens]

        processed_data.append({
            'file_id': file_id,
            'category': category,
            'tokens': stemmed_tokens
        })

    with open(outputfile, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"Preprocessing completed! Processed {len(processed_data)} entries, result saved to {outputfile}")

def count_word(inputfile, outputfile):
    
    with open(inputfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    doc_count = {cat: 0 for cat in CATEGORIES}
    word_freq = defaultdict(lambda: [0]*5)
    
    for doc in data:
        cat = doc['category']
        cat_idx = CATEGORIES.index(cat)
        doc_count[cat] += 1
        
        for word in doc['tokens']:
            word_freq[word][cat_idx] += 1
    
    
    with open(outputfile, 'w', encoding='utf-8') as f:
        
        f.write(' '.join([str(doc_count[cat]) for cat in CATEGORIES]) + '\n')
        
        for word, freq in word_freq.items():
            f.write(f"{word} {' '.join(map(str, freq))}\n")
    print(f"Word count completed! Result saved to {outputfile}")


def feature_selection(inputfile, threshold, outputfile):
    with open(inputfile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    word_freq = {}
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        word = parts[0]
        freq = list(map(int, parts[1:6]))
        word_freq[word] = freq
    
    sorted_words = sorted(
        word_freq.items(),
        key=lambda x: sum(x[1]),
        reverse=True
    )[:threshold]
    feature_words = dict(sorted_words)
    
    total_feature_freq = [0]*5
    for freq in feature_words.values():
        for i in range(5):
            total_feature_freq[i] += freq[i]
    
    with open(outputfile, 'w', encoding='utf-8') as f:

        f.write(' '.join(map(str, total_feature_freq)) + '\n')

        for word, freq in feature_words.items():
            f.write(f"{word} {' '.join(map(str, freq))}\n")
    print(f"Feature selection completed! Result saved to {outputfile}")


def calculate_probability(word_count_file, word_dict_file, outputfile):
    
    with open(word_count_file, 'r', encoding='utf-8') as f:
        doc_count_line = f.readline().strip()
        doc_count = list(map(int, doc_count_line.split()))
    total_docs = sum(doc_count)
    
    with open(word_dict_file, 'r', encoding='utf-8') as f:
        total_feature_freq_line = f.readline().strip()
        total_feature_freq = list(map(int, total_feature_freq_line.split()))
        
        feature_words = {}
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            freq = list(map(int, parts[1:6]))
            feature_words[word] = freq
    
    V = len(feature_words)
    
   
    prior_prob = [count / total_docs for count in doc_count]
    
    
    word_prob = {}
    for word, freq in feature_words.items():
        prob = [0.0]*5
        for i in range(5):
            
            numerator = freq[i] + 1
            denominator = total_feature_freq[i] + V
            prob[i] = numerator / denominator
        word_prob[word] = prob
    
    
    with open(outputfile, 'w', encoding='utf-8') as f:
        
        f.write(' '.join(map(str, prior_prob)) + '\n')
        
        for word, prob in word_prob.items():
            f.write(f"{word} {' '.join(map(str, prob))}\n")
    print(f"Probability calculation completed! Result saved to {outputfile}")


def classify(probability_file, testset_file, outputfile):
    
    with open(probability_file, 'r', encoding='utf-8') as f:
        # Prior probabilities
        prior_line = f.readline().strip()
        prior_prob = list(map(float, prior_line.split()))
        # Word posterior probabilities
        word_post_prob = {}
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            prob = list(map(float, parts[1:6]))
            word_post_prob[word] = prob
    feature_words = set(word_post_prob.keys())
    
    with open(testset_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    result = []
    for doc in test_data:
        file_id = doc['file_id']
        tokens = doc['tokens']
        
        valid_tokens = [t for t in tokens if t in feature_words]
        
        cat_score = [0.0]*5
        for i in range(5):
            score = math.log(prior_prob[i])
            
            for word in valid_tokens:
                score += math.log(word_post_prob[word][i])
            cat_score[i] = score
        
        
        best_cat_idx = cat_score.index(max(cat_score))
        best_cat = CATEGORIES[best_cat_idx]
        result.append((file_id, best_cat))
    
    with open(outputfile, 'w', encoding='utf-8') as f:
        for file_id, cat in result:
            f.write(f"{file_id} {cat}\n")
    print(f"Classification completed! Result saved to {outputfile}")

def f1_score(testset_file, classification_result_file):
    
    with open(testset_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    true_labels = {}
    for item in test_data:
        file_id = item[0]       
        category = item[1]     
        true_labels[file_id] = category

    
    pred_labels = {}
    with open(classification_result_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            file_id, cat = line.split()
            pred_labels[file_id] = cat

    tp = {cat: 0 for cat in CATEGORIES}
    fp = {cat: 0 for cat in CATEGORIES}
    fn = {cat: 0 for cat in CATEGORIES}

    for file_id, true_cat in true_labels.items():
        pred_cat = pred_labels.get(file_id, '')
        if true_cat == pred_cat:
            tp[true_cat] += 1
        else:
            fn[true_cat] += 1
            fp[pred_cat] += 1

    f1_list = []
    for cat in CATEGORIES:
        precision = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) > 0 else 0
        recall = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list)
    return macro_f1


def main():
    ''' Main Function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-pps', '--preprocess',type=str,nargs=2,help='preprocess the dataset')
    parser.add_argument('-cw','--count_word',type=str,nargs=2,help='count the words from the corpus')
    parser.add_argument('-fs','--feature_selection',type=str,nargs=3,help='\select the features from the corpus')
    parser.add_argument('-cp','--calculate_probability',type=str,nargs=3,
                        help='calculate the posterior probability of each feature word, and the prior probability of the class')
    parser.add_argument('-cl','--classify',type=str,nargs=3,
                        help='classify the testset documents based on the probability calculated')
    parser.add_argument('-f1','--f1_score', type=str, nargs=2,
                        help='calculate the F-1 score based on the classification result.')
    opt=parser.parse_args()

    if(opt.preprocess):
        input_file = opt.preprocess[0]
        output_file = opt.preprocess[1]
        preprocess(input_file,output_file)
    elif(opt.count_word):
        input_file = opt.count_word[0]
        output_file = opt.count_word[1]
        count_word(input_file,output_file)
    elif(opt.feature_selection):
        input_file = opt.feature_selection[0]
        threshold = int(opt.feature_selection[1])
        outputfile = opt.feature_selection[2]
        feature_selection(input_file,threshold,outputfile)
    elif(opt.calculate_probability):
        word_count = opt.calculate_probability[0]
        word_dict = opt.calculate_probability[1]
        output_file = opt.calculate_probability[2]
        calculate_probability(word_count,word_dict,output_file)
    elif(opt.classify):
        probability = opt.classify[0]
        testset = opt.classify[1]
        outputfile = opt.classify[2]
        classify(probability,testset,outputfile)
    elif(opt.f1_score):
        testset = opt.f1_score[0]
        classification_result = opt.f1_score[1]
        f1 = f1_score(testset,classification_result)
        print('The F1 score of the classification result is: '+str(f1))


if __name__ == '__main__':
    import os
    main()