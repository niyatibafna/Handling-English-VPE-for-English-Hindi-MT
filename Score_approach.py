import copy
from pycorenlp import StanfordCoreNLP
from nltk.tokenize import sent_tokenize
nlp = StanfordCoreNLP('http://localhost:9000')
def clean(text):
	text = text.replace(" n ' t", "n't")
	text = text.replace(" ' s", "'s")
	return(text)

def Parse(S):
	whole_text = S
	# sentences = sent_tokenize(whole_text)
	# print(sentences)
	sentences = [whole_text]
	Parsed = []
	for text in sentences:
		result = nlp.annotate(text,properties = {'annotaters': 'pos', 'outputFormat': 'conll', 'timeout': '50000',})
		Parsing = open("Parsing.txt", "w+")
		print(result, file = Parsing)
		Parsing.close()
		readParsing = open("Parsing.txt", "r")
		for line in readParsing:
			A = line.split()
			if(len(A) != 0):
				Parsed_Row = []
				Parsed_Row.append(A[1]) #Word
				Parsed_Row.append(A[5]) #Root
				Parsed_Row.append(A[6]) #Dependency relation
				Parsed_Row.append(A[3]) #POS tag
				Parsed_Row.append(A[2]) #Lemma
				Parsed.append(Parsed_Row)
		readParsing.close()
	return(Parsed)
def To_Be(Parsed, row_index):
	row = Parsed[row_index]
	Parsed_l = len(Parsed)
	ellipsis = True
	parent = (int)(row[1])
	if(row[2].startswith('aux') and (Parsed[parent-1][3].startswith('VB') or Parsed[parent-1][3].startswith('RB') or Parsed[parent-1][3].startswith('IN'))):
		ellipsis = False
	if(Parsed[parent-1][3].startswith('JJ') or Parsed[parent-1][3].startswith('NN')):
		ellipsis = False
	if(row[2] == 'cop'):
		ellipsis = False
	
	for i in range(0, Parsed_l):
		if((int)(Parsed[i][1])-1 == row_index and 'obj' in Parsed[i][2] and i > row_index): 
			ellipsis = False
		if((int)(Parsed[i][1]) - 1 == row_index and Parsed[i][2].endswith('comp')): #It used to be that you coudn't...
			ellipsis = False
			# if(Parsed[i][3] == 'VBN' or Parsed[i][3] == 'VBG' or Parsed[i][3].startswith('JJ') or Parsed[i][3].startswith('NN') or Parsed[i][3].startswith('RB') or Parsed[i][3].startswith('WP')):
			# 	ellipsis = False	
		if(Parsed[i][3].startswith('VB') and Parsed[i][2] == 'xcomp'):
			ellipsis = False
		if(row[2].endswith('comp') and (int)(Parsed[i][1]) - 1 == row_index and Parsed[i][2].startswith('nsubj')): #They say that there is a dog...
			ellipsis = False
	if(ellipsis):
		return(1)
	return(0)
def To_Have(Parsed, row_index):
	Indicator_To_Be = ["is", "isn’t","be", "been", "was", "wasn’t", "am", "ain’t", "are", "aren’t"]
	Parsed_l = len(Parsed)
	row = Parsed[row_index]
	ellipsis = True
	parent = (int)(row[1])
	if(row[2].startswith('aux') and (Parsed[parent-1][3].startswith('VB') or Parsed[parent-1][3].startswith('JJ') or Parsed[parent-1][3].startswith('RB'))):# or Parsed[parent-1][3].startswith('NN'))): # has eaten
		ellipsis = False
	if(row[2]=='aux' and Parsed[parent-1][0] in Indicator_To_Be): #has been eating: this ellipsis has been handled in To_Be().
		#In the case of ellipsis: 'I haven't been' haven't is an aux child of 'been'. In 'I haven't been shopping', it is an aux child of VBG 'shopping', and so ellipsis is ruled out by above condition.
		ellipsis = False
	for i in range(row_index, Parsed_l):
		if((int)(Parsed[i][1])-1 == row_index and 'obj' in Parsed[i][2]): #has apples
			ellipsis = False
		if((int)(Parsed[i][1])-1 == row_index and Parsed[i][2] == 'xcomp'): #has to go shopping 
			ellipsis = False
		if((int)(Parsed[i][1])-1 == row_index and (Parsed[i][2].startswith('PR') or Parsed[i][2].startswith('NN'))): #has apples
			ellipsis = False
	if(ellipsis):
		return(1)
	return(0)
def To_Do(Parsed, row_index):
	Parsed_l = len(Parsed)
	row = Parsed[row_index]
	ellipsis = True
	parent = (int)(row[1])
	# print(parent)
	if(row[2]=='aux' and Parsed[parent-1][3].startswith('VB')): # does care
		ellipsis = False
	for i in range(row_index, Parsed_l):
		if('obj' in Parsed[i][2] and (int)(Parsed[i][1])-1 == row_index): #does his homework.
			ellipsis = False
	if(row[3] == 'VB'): # in infinitive form - to do.
		for i in range(row_index, Parsed_l):
			if((int)(Parsed[i][1])-1 == row_index and Parsed[i][3].casefold() == 'TO'.casefold()):
				ellipsis = False
	if(ellipsis): #if v1 is ROOT. 
		return(1)
	return(0)
def Modals(Parsed, row_index):
	Indicator_To_Be = ["is", "isn’t","be", "been", "was", "wasn’t", "am", "ain’t", "are", "aren’t"]
	Parsed_l = len(Parsed)
	row = Parsed[row_index]
	ellipsis = True
	parent = (int)(row[1])
	if(row[3].startswith('NN')):
		ellipsis = False
	if(row[2].startswith('aux') and (Parsed[parent-1][3].startswith('VB') or Parsed[parent-1][3].startswith('JJ') or Parsed[parent-1][3].startswith('IN') or Parsed[parent-1][3].startswith('CD'))): # does care, would be no fight #removed parent RB condition
		ellipsis = False
	#deal with 'would be no fight' cases where aux is child of noun
	if(row[2]=='aux' and Parsed[parent-1][0] in Indicator_To_Be):
		ellipsis = False
	for i in range(row_index, Parsed_l):
		if('obj' in Parsed[i][2] and (int)(Parsed[i][1])-1 == row_index): #does his homework.
			ellipsis = False
	if(ellipsis): #if v1 is ROOT. 
		return(1)
	return(0)
def To(Parsed, row_index):
	Parsed_l = len(Parsed)
	row = Parsed[row_index]
	ellipsis = True
	parent = (int)(row[1])
	if(row[2] != 'xcomp'):
		ellipsis = False
	# if(row[2] == 'mark'):
	# 	ellipsis = False
	# if(row[2] == 'case' and Parsed[parent-1][3].startswith('NN')):
	# 	ellipsis = False
	if(ellipsis): 
		return(1)
	return(0)
def get_verb_list(Parsed):
	Verb_list = {}
	for r in range(0, len(Parsed)):
		if(Parsed[r][3].startswith('VB')):
			Verb_list[r] = 0
	return(Verb_list)


def evaluate_scores(Verb_list, row_index, default):
	#Disadvantage backward gapping
	for key in Verb_list:
		if(key > row_index):
			Verb_list[key] -= 1
	print("Disadvantaging backward gapping: ")
	print(Verb_list)
	#Find highest score
	optimal_verb = list(Verb_list.keys())[0]
	for key in Verb_list:
		if(Verb_list[key] > Verb_list[optimal_verb]):
			optimal_verb = key
	#Collect clashes
	clashing_verbs = []
	for key in Verb_list:
		if(Verb_list[key] == Verb_list[optimal_verb]):
			clashing_verbs.append(key)
	if(len(clashing_verbs) == 1):
		return optimal_verb
	# #Favour backtracking as default
	# if(default in clashing_verbs):
	# 	return default
	#Find verb at minimum distance before, if none, then look after SOE
	closest_verb_before = -1
	closest_verb_after = -1
	for v in clashing_verbs:
		if(v < row_index and v > closest_verb_before):
			closest_verb_before = v
		if(v > row_index and v > closest_verb_after):
			closest_verb_after = v
	if(closest_verb_before != -1):
		return(closest_verb_before)
	else:
		return(closest_verb_after)
	# #Find closest verb equally among forward and backward
	# for verb in clashing_verbs:
	# 	if(abs(verb - row_index) < abs(closest_verb - row_index)):
	# 		closest_verb = verb
	# #In case of clash, favour forward ellipsis to backward ellipsis 
	# for verb in clashing_verbs:
	# 	if(abs(verb - row_index) == abs(closest_verb - row_index)):
	# 		if((verb - row_index) < 0):
	# 			return verb
	# 		else:
	# 			return closest_verb

def find_main_verb(Parsed, row_index, ellipsis):
	Verb_list = get_verb_list(Parsed)
	if(row_index in Verb_list):
		del Verb_list[row_index]
	if(ellipsis == 5):
		key = int(Parsed[row_index][1]) - 1
		if(key in Verb_list):
			del Verb_list[key]
	print(Verb_list)
	#If to_be ellipsis, keep only gerunds and past participles
	if(ellipsis == 1):
		gerunds_and_participles = []
		for key in Verb_list:
			if(Parsed[key][3] != 'VBG' and Parsed[key][3] != 'VBN'):
				gerunds_and_participles.append(key)
		for key in gerunds_and_participles:
			del Verb_list[key]
	#If to_be, to_do, modals, eliminate candidates from the same class
	same_class = []
	Indicator_To_Be = ["is", "isn’t","be", "been", "was", "wasn’t", "am", "ain’t", "are", "aren’t"]
	Indicator_To_Do = ["does", "doesn’t", "do", "don’t", "did"]
	Indicator_Modals = ["will", "wo", "would", "may", "might", "must", "can", "ca", "could", "should"]
	if(ellipsis == 1):
		same_class = Indicator_To_Be
	if(ellipsis == 3):
		same_class = Indicator_To_Do
	if(ellipsis == 4):
		same_class = Indicator_Modals
	if(ellipsis == 1 or ellipsis == 3 or ellipsis == 4):
		same_class_candidates = []
		for key in Verb_list:
			if(Parsed[key][0].casefold() in same_class):
				if(Parsed[key][0].casefold() != 'do'.casefold()):
					same_class_candidates.append(key)
		for candidate in same_class_candidates:
			del Verb_list[candidate]
	be_or_modals = []
	for key in Verb_list:
		if(Parsed[key][4] in Indicator_To_Be or Parsed[key][4] in Indicator_To_Be):
			be_or_modals.append(key)
	for v in be_or_modals:
		if(Parsed[v][0].casefold().startswith('be') and not Parsed[v][2].startswith('aux')):
			continue
		del Verb_list[v]
	if(len(Verb_list) == 0):
		return(-1)
	#Working with subject
	number = -1
	noun_subject = 'none_subject'
	proper = -1
	passive = -1
	real_index = row_index
	noun_subject_row = -1
	if(ellipsis == 5):
		real_index = int(Parsed[row_index][1]) - 1
	#backtracking for comp children verbs
	if(Parsed[real_index][2].endswith('comp') or Parsed[real_index][2].endswith('conj')):
		subject_exists = True
		while(subject_exists and noun_subject_row == -1):
			for row in Parsed:
				if((int)(row[1]) - 1 == real_index and row[2].startswith('nsubj')):
					noun_subject_row = row
					subject_exists = False
					break
			if(Parsed[real_index][3].startswith('VB') and (Parsed[real_index][2].endswith('comp') or Parsed[real_index][2].endswith('conj'))):
				real_index = int(Parsed[real_index][1]) - 1
			else:
				subject_exists = False
	#Routine for others
	else:
		for row in Parsed:
			if(int(row[1]) - 1 == real_index and row[2].startswith('nsubj')):
				noun_subject_row = row
	if(noun_subject_row != -1):
		#Find subject of adjective clause
		if(noun_subject_row[3].startswith('W') and Parsed[real_index][2] == 'acl:relcl'):
			parent_noun = int(Parsed[real_index][1]) - 1	
			noun_subject_row = Parsed[parent_noun]
		#Noun subject name
		noun_subject = noun_subject_row[0]
		#Passivity
		if(noun_subject_row[2].endswith('pass')):
			passive = 1
		else:
			passive = 0
		#Number
		#Dealing with DT-noun phrases
		if(noun_subject_row[3].startswith('DT')):
			for r in range(len(Parsed)):
				if((int)(Parsed[r][1]) - 1 == Parsed.index(noun_subject_row) and Parsed[r][2].startswith('nmod') and r < row_index):
					if(Parsed[r][3].endswith('S') or Parsed[r][4] == 'they' or Parsed[r][4] == 'we'):
						number = 1
					else:
						number = 0
					break
		if(number == -1):
			if(noun_subject_row[3].endswith('S') or noun_subject_row[4] == 'they' or noun_subject_row[4] == 'we'):
				number = 1
			else:
				number = 0
		#Conjugate nouns
		conj_exists = False
		for row in Parsed:
			if(int(row[1]) - 1 == Parsed.index(noun_subject_row) and row[2].startswith('cc') and row[4].startswith('and')):
				conj_exists = True
			if(int(row[1]) - 1 == Parsed.index(noun_subject_row) and row[2].startswith('conj') and (row[3].startswith('N') or row[3].startswith('PRP'))):
				if(conj_exists):
					number = 1
		#Properness
		if(noun_subject_row[3].endswith('NP') or noun_subject_row[3].endswith('NPS')):
			proper = 1
		else:
			proper = 0
	for each_key in Verb_list:
		real_index = each_key
		if(Parsed[each_key][4] == 'be'.casefold()):
			real_index = (int)(Parsed[each_key][1]) - 1
		verb_number = -1
		verb_noun_subject = 'none_subject'
		verb_proper = -1
		verb_passive = -1
		verb_noun_subject_row = -1
		#Backtracking for comp children verbs
		if(Parsed[real_index][2].endswith('comp') or Parsed[real_index][2].endswith('conj')):
			subject_exists = True
			while(subject_exists and verb_noun_subject_row == -1):
				for row in Parsed:
					if((int)(row[1]) - 1 == real_index and row[2].startswith('nsubj')):
						verb_noun_subject_row = row
						subject_exists = False
						break
				if(Parsed[real_index][3].startswith('VB') and (Parsed[real_index][2].endswith('comp') or Parsed[real_index][2].endswith('conj'))):
					real_index = int(Parsed[real_index][1]) - 1
				else:
					subject_exists = False
		#Routine check over all words for other verbs
		else:
			for row in Parsed:
				if(int(row[1]) - 1 == real_index and row[2].startswith('nsubj')):
					verb_noun_subject_row = row
		if(verb_noun_subject_row == -1):
			continue
		# if(each_key == 15):
		# 	print(verb_noun_subject_row)
		#Finding subject of verbs in adjective clauses: 
		if(verb_noun_subject_row[3].startswith('W') and Parsed[real_index][2] == 'acl:relcl'):
			parent_noun = int(Parsed[real_index][1]) - 1	
			verb_noun_subject_row = Parsed[parent_noun]
		#Passivity
		verb_noun_subject = verb_noun_subject_row[0]
		if(verb_noun_subject_row[2].endswith('pass')):
			verb_passive = 1
		else:
			verb_passive = 0
		#Number
		if(verb_noun_subject_row[3].startswith('DT')):
			for r in range(len(Parsed)):
				if((int)(Parsed[r][1]) - 1 == Parsed.index(verb_noun_subject_row) and Parsed[r][2].startswith('nmod')):
					if(Parsed[r][3].endswith('S') or Parsed[r][4] == 'they' or Parsed[r][4] == 'we'):
						verb_number = 1
					else:
						verb_number = 0
					break
		if(verb_number == -1):
			if(verb_noun_subject_row[3].endswith('S') or verb_noun_subject_row[4] == 'they' or verb_noun_subject_row[4] == 'we'):
				verb_number = 1
			else:
				verb_number = 0
		#Conjugate noun subjects
		conj_exists = False
		for row in Parsed:
			if(int(row[1]) - 1 == Parsed.index(verb_noun_subject_row) and row[2].startswith('cc') and row[4].startswith('and')):
				conj_exists = True
			if(int(row[1]) - 1 == Parsed.index(verb_noun_subject_row) and row[2].startswith('conj') and (row[3].startswith('N') or row[3].startswith('PRP'))):
				if(conj_exists):
					verb_number = 1
		#Properness
		if(verb_noun_subject_row[3].endswith('NNP') or verb_noun_subject_row[3].endswith('NPS')):
			verb_proper = 1
		else:
			verb_proper = 0
		#Calucating scores
		if(Parsed[each_key][0]=='prohibits'):
			print(verb_noun_subject_row[0])
			print(noun_subject_row[0])

		if(noun_subject.casefold() == verb_noun_subject.casefold()):
			Verb_list[each_key] += 2
		else:
			if(passive != -1 and verb_passive != -1 and passive != verb_passive):
				Verb_list[each_key] -= 2 #the idea is to disadvantage the non-equal guys, not advantage the rest, because some may not have nsubj
			if(number == verb_number):
				Verb_list[each_key] += 1
			if(proper ==1 and verb_proper == 1):
				Verb_list[each_key] += 1
		
	print("After looking at noun subjects: ")
	print(Verb_list)
	#Heads by _comp relation
	key = row_index
	while(Parsed[key][2].endswith('comp')):
		parent = int(Parsed[key][1]) - 1
		if(parent in Verb_list):
			Verb_list[parent] -= 3
		key = parent
	#Children by comp relation
	for each_key in Verb_list:
		ancestor = each_key
		while(Parsed[ancestor][2] == 'xcomp'):
			if(ancestor == row_index or (Parsed[ancestor][4]=='not' and int(Parsed[ancestor][1]) - 1) == row_index):
				Verb_list[each_key] -= 3
				break
			ancestor = int(Parsed[ancestor][1]) - 1
	print("After disfavouring complement clauses: ")
	print(Verb_list)
	#Auxiliary belonging to same class
	Indicator_To_Be = ["is", "isn’t","be", "been", "was", "wasn’t", "am", "ain’t", "are", "aren’t"]
	Indicator_To_Have = ["has", "hasn’t", "have", "had"]
	Indicator_To_Do = ["does", "doesn’t", "do", "don’t", "did"]
	Indicator_Modals = ["will", "wo", "would", "may", "might", "must", "can", "ca", "could", "should"]
	Indicator_To = ["to"]
	for each_key in Verb_list:
		key = each_key
		if(Parsed[key][4] == 'be'.casefold()):
			key = (int)(Parsed[each_key][1]) - 1
		for row in Parsed:
			auxiliary = 0
			if(int(row[1]) - 1 == key):
				if(row[2] == 'aux'):
					if(row[0].casefold() in Indicator_To_Be):
						auxiliary = 1
					if(row[0].casefold() in Indicator_To_Have):
						auxiliary = 2
					if(row[0].casefold() in Indicator_To_Do):
						auxiliary = 3
					if(row[4].casefold() == Parsed[row_index][4] and row[4] in Indicator_Modals):
						auxiliary = 4
				if(row[3].casefold() == 'TO'.casefold()):
					auxiliary = 5
			# print("ell ", ellipsis)
			if(auxiliary == ellipsis):
				Verb_list[each_key] += 1
	print("After looking at auxiliaries: ")
	print(Verb_list)
	#Backtracking up the dependency tree
	default = -1
	backtracking_verb = -1
	parent = int(Parsed[row_index][1]) - 1
	if(ellipsis == 1):
		while((Parsed[parent][3] != 'VBG' and Parsed[parent][3] != 'VBN') and Parsed[parent][2] != 'ROOT' and not Parsed[parent][3].startswith('JJ')):
			parent = (int)(Parsed[parent][1]) - 1
		if(Parsed[parent][3] == 'VBG' or Parsed[parent][3] == 'VBN'):
			backtracking_verb = parent
	else:
		if(ellipsis == 5):
			parent = int(Parsed[parent][1]) - 1
		while(parent != -1 and (Parsed[parent][3].startswith('VB') == False)): 
			if(parent == (int)(Parsed[parent][1]) - 1):
				break
			parent = (int)(Parsed[parent][1]) - 1
		if(Parsed[parent][3].startswith('VB')):
			backtracking_verb = parent
	if(backtracking_verb in Verb_list):
		Verb_list[backtracking_verb] += 1
		default = backtracking_verb
	print("After backtracking up dependency tree: ")
	print(Verb_list)
	main_verb = evaluate_scores(Verb_list, row_index, default)
	return main_verb	

def add_auxiliaries(Parsed, verb_index, ellipsis):
	auxiliary = []
	if(ellipsis == 2 or ellipsis == 4):
		Indicator_To_Have = ["has", "hasn’t", "have", "had"]
		Indicator_Modals = ["will", "wo", "would", "may", "might", "must", "can", "ca", "could", "should"]
		same_class_found = False
		for i in range(verb_index):
			if(int(Parsed[i][1]) - 1 == verb_index and Parsed[i][2].startswith('aux')):
				if(same_class_found):
					auxiliary.append(i)
				else:
					if((ellipsis == 2 and Parsed[i][0].casefold() in Indicator_To_Have) or (ellipsis == 4 and Parsed[i][0].casefold() in Indicator_Modals)):
						same_class_found = True
					else:
						if(ellipsis == 2 and Parsed[i][4] == "be"):
							auxiliary.append(i)
						break
	return auxiliary

def add_particles(Parsed, verb_index):
	particle = []
	for i in range(verb_index, len(Parsed)):
		if(Parsed[i][2]== 'compound:prt'):
			par = (int)(Parsed[i][1]) - 1
			while(Parsed[par][2] != 'ROOT' and not Parsed[par][3]. startswith('VB')):
				par = (int)(Parsed[par][1]) - 1
			if(par == verb_index):
				particle.append(i) 
	return particle

def add_complements(Parsed, verb_index):
	f = open("HindiConstructions.txt", "r")
	construction_verbs = f.read()
	construction_verbs = construction_verbs.split("\n")
	complement = []
	while(Parsed[verb_index][4].casefold() in construction_verbs):
		complement_found = False
		for i in range(verb_index+1, len(Parsed)):
			if(Parsed[i][2].endswith('xcomp') and int(Parsed[i][1]) - 1 == verb_index):
				complement_found = True
				for j in range(verb_index+1, i):
					if(Parsed[j][0].casefold() == 'to' and int(Parsed[j][1]) - 1):
						complement.append(j)
				complement.append(i)
				verb_index = i
				break
		if(not complement_found):
			break
	return(complement)

def supplement_main_verb(Parsed, verb_index, ellipsis):
	elliptical_verb = []
	auxiliary = add_auxiliaries(Parsed, verb_index, ellipsis)
	particle = add_particles(Parsed, verb_index)
	complement = add_complements(Parsed, verb_index)
	elliptical_verb = auxiliary + [verb_index] + particle + complement
	elliptical_verb.sort()
	return(elliptical_verb)

def add_elliptical_verb(Parsed, Output_sentence, row_index, added_words, elliptical_verb, ellipsis):
	index_of_not = row_index
	for i in range(row_index, len(Parsed)):
		if(Parsed[i][0] =='not'.casefold() and (int)(Parsed[i][1])-1 == row_index):
			index_of_not = i
	if(Parsed[row_index+1][0] == "n't"):
		row_index += 1
		index_of_not += 1
	output_sentence_index = row_index + added_words
	output_sentence_index = output_sentence_index + (index_of_not - row_index) 
	for i in range(len(elliptical_verb)):
		word = Parsed[elliptical_verb[i]][0]
		if(i == 0):
			if(ellipsis == 3 or ellipsis == 4):
				word = Parsed[elliptical_verb[i]][4]	
		Output_sentence.insert(output_sentence_index + 1, word.upper())
		output_sentence_index += 1		
	return(Output_sentence)


def convert_to_sentence(Output_sentence):
	sent = ""
	for i in range(len(Output_sentence)):
		sent = sent + Output_sentence[i] + " "
	return(sent)



def Partial_Verb_Phrase_Ellipsis(sentence, All_words, Parsed):
	global cases_ellipsis
	with_ellipsis = open("With_ellipsis.txt", "a+")
	with_elliptical_verb = open("With_elliptical_verb.txt", "a+")
	Indicator_To_Be = ["is", "isn’t","be", "been", "was", "wasn’t", "am", "ain’t", "are", "aren’t"]
	Indicator_To_Have = ["has", "hasn’t", "have", "had"]
	Indicator_To_Do = ["does", "doesn’t", "do", "don’t", "did"]
	Indicator_Modals = ["will", "wo", "would", "may", "might", "must", "can", "ca", "could", "should"]
	Indicator_To = ["to"]
	Output_sentence = copy.copy(All_words)
	added_words = 0
	row_index = -1
	elliptical_verb = []
	for word in All_words:
		ellipsis = 0
		row_index += 1
		if(word.casefold() in Indicator_To_Be):
			if(To_Be(Parsed, row_index)):
				ellipsis = 1
		if(word in Indicator_To_Have):
			if(To_Have(Parsed, row_index)):
				ellipsis = 2
		if(word.casefold() in Indicator_To_Do):
			if(To_Do(Parsed, row_index)):
				ellipsis = 3
		if(word.casefold() in Indicator_Modals):
			if(Modals(Parsed, row_index)):
				ellipsis = 4
		if(word.casefold() in Indicator_To):
			if(To(Parsed, row_index)):
				ellipsis = 5
		if(ellipsis):
			print("Ellipsis at: ", All_words[row_index])
			# print(sentence, file = with_ellipsis)
			cases_ellipsis += 1
			verb_index = find_main_verb(Parsed, row_index, ellipsis)
			if(verb_index == -1):
				continue
			elliptical_verb = supplement_main_verb(Parsed, verb_index, ellipsis)			
			Output_sentence = add_elliptical_verb(Parsed, Output_sentence, row_index, added_words, elliptical_verb, ellipsis)
			output = convert_to_sentence(Output_sentence)
			print(output, file = with_elliptical_verb)
			print("\n", file = with_elliptical_verb)
			added_words += len(elliptical_verb)
			print("Main verb: ", verb_index)
			print("Elliptical VP: ", elliptical_verb)
			print("Output sentence: ", Output_sentence)
	if(len(elliptical_verb) == 0):
		return([])
	elliptical_verb_words = []
	for w in elliptical_verb:
		elliptical_verb_words.append(Parsed[w][0])
	return(elliptical_verb_words)
	# return(Output_sentence)

# S = input()
# Parsed = Parse(S)
# t = 1
# for row in Parsed:
# 	print(t, ": ", row)
# 	t += 1
# All_words = []
# for row in Parsed:
# 	All_words.append(row[0])
# All_words_copy = copy.copy(All_words)
# cases_ellipsis = 0
# found_elliptical_verb = Partial_Verb_Phrase_Ellipsis(S, All_words_copy, Parsed)



# f1 = open('Cases_Ellipsis_WSJ_sample_1.txt', 'r')
# f2 = open('Cases_Ellipsis_WSJ_solutions_sample_1.txt', 'r')
f1 = open("Cases_Ellipsis_WSJ_200_last.txt", "r")
text = f1.read()
text = text.replace("\n\n", "\n")
text = clean(text)
text = text.split("\n")
# solutions = f2.read().split("\n")
results = []
for line in text:
	if(line == '' or line == '\n'):
		text.remove(line)
cases_ellipsis = 0
cases_no_ellipsis = 0
count = 0
for sent in text:
	if(count == 100):
		break
	Parsed = Parse(sent)
	print(sent)
	count += 1
	t = 1
	for row in Parsed:
		print(t, ": ", row)
		t += 1
	All_words = []
	for row in Parsed:
		All_words.append(row[0])
	All_words_copy = copy.copy(All_words)
	found_elliptical_verb = Partial_Verb_Phrase_Ellipsis(sent, All_words_copy, Parsed)
	print("Found verb: ", found_elliptical_verb)
	cases_ellipsis = 0
# 	results.append(found_elliptical_verb)
# results_file = open("Results_File.txt", "w")
# for result in results:
# 	print(result, file = results_file)
# total = len(results)
# correct_solutions = 0
# for i in range(total):
# 	R = results[i]
# 	S = ""
# 	for x in R:
# 		S = S + x.strip() + " "
# 	# print(S)
# 	# print(solutions[i])
# 	if(solutions[i].strip().casefold() == S.strip().casefold()):
# 		correct_solutions += 1
# 	else:
# 		print(text[i])
# 		print(solutions[i])		
# 		print(results[i])

# print(correct_solutions)
# print(total)
