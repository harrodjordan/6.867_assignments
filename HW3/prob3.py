import gensim 

my_corpus = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
dictionary = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')

model = gensim.models.LdaMallet('/Users/Desktop/mallet/bin/mallet', corpus=my_corpus, num_topics=100, id2word=dictionary)
model.print_topics(2)