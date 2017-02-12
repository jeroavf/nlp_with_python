import nltk 


texto = "Mr. Green killed Colonel Mustard in the study with the candlestick"
frases= nltk.tokenize.sent_tokenize(texto)
tokens=nltk.word_tokenize(texto)
#print(tokens)
classes=nltk.pos_tag(tokens)
#print(classes)

entidades=nltk.chunk.ne_chunk(classes)
print(entidades)