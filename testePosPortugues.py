import nltk
#from nltk.corpus import floresta
from nltk.corpus import mac_morpho


texto="Joao da Silva foi a feira de manhã. Ao chegar na feira encontrou com seu amigo Pedro. Os dois trocaram informações acerca da nova ponte."
print("0")
macMorpho = nltk.corpus.mac_morpho.tagged_sents()
print("1")
sizeTraining = int(len(macMorpho) * 0.9)
print("2")
training_sentences = macMorpho[:sizeTraining]
print("3")
tagger = nltk.tag.UnigramTagger(training_sentences)
print("4")
words = nltk.word_tokenize(texto)
print("5")
print("WORDS ===============================================")
print(words)
print("6")
print("POSTAGER ===============================================")
postagger=tagger.tag(words) 
print(postagger)

