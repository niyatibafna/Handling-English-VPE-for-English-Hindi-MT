import copy
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
def Parse(S):
	text = S
	result = nlp.annotate(text,properties = {'annotaters': 'pos', 'outputFormat': 'conll', 'timeout': '50000',})
	Parsing = open("Parsing.txt", "w+")
	print(result, file = Parsing)
	Parsing.close()
	readParsing = open("Parsing.txt", "r")
	Parsed = []
	for line in readParsing:
		A = line.split()
		if(len(A) != 0):
			Parsed_Row = []
			Parsed_Row.append(A[1])
			Parsed_Row.append(A[5])
			Parsed_Row.append(A[6])
			Parsed_Row.append(A[3])
			Parsed_Row.append(A[2])
			Parsed.append(Parsed_Row)
	return(Parsed)
def create_features(f):
	# f = open("heyyo.txt", "r")
	global feature_file, counter
	Indicator_To_Be = ["is", "isn’t","be", "been", "was", "wasn’t", "am", "ain’t", "are", "aren’t"]
	Indicator_To_Have = ["has", "hasn’t", "have", "had"]
	Indicator_To_Do = ["does", "doesn’t", "do", "don’t", "did"]
	Indicator_Modals = ["will", "wo", "would", "may", "might", "must", "can", "ca", "could", "should"]
	Indicator_To = ["to"]
	Indicator = Indicator_To_Be + Indicator_To_Have + Indicator_To_Do +Indicator_Modals + Indicator_To	
	for line in f:
		if(line == "\n" or line == ""):
			continue
		Features = [0]*6
		Features[4] = 1
		Features[5] = 1
		Parsed = Parse(line)
		All_words = []
		for row in Parsed:
			All_words.append(row[0])
		t = -1
		stop_here = False
		for word in All_words:
			t += 1
			if(word in Indicator and stop_here == False):
				# F0
				Features[0] = 1
				#F1
				if(Parsed[t][3].startswith('VB') or Parsed[t][3].startswith('TO')):
					Features[1] = 1
				#F2
				if(Parsed[t][2] != 'aux' or Parsed[t][2] == 'xcomp'):
					Features[2] = 1
					stop_here = True #fills in the features according to this verb
				#F3
				for i in range(len(Parsed)):
					if(Parsed[i][1] != 'CD' and (int)(Parsed[i][1]) - 1 == t and Parsed[i][2] == 'dobj'):
						Features[3] = -1
				#F4
				if(t != len(Parsed)-1 and Parsed[t+1][0] == 'so'):
					Features[4] = -1
				#F5
				if(t != len(Parsed)-1 and Parsed[t+1][3] == 'NN'):
					Features[5] = -1
		S = ""
		for item in Features:
			S = S + " " +str(item)
		print(S, file = feature_file)
		print(counter)
		counter += 1
file1 = open("Cases_Ellipsis_BNC.txt", "r")
file2 = open("BNC_first_3000_non_ell.txt", "r")
feature_file = open("feature_file.txt","w")
counter = 1
create_features(file1)
create_features(file2)
