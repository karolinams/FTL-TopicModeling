import nltk, re, gensim, spacy, csv, pyLDAvis
import gensim.corpora as corpora
import pyLDAvis.gensim  # don't skip this
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from pprint import pprint
from nltk.corpus import stopwords


# Define Stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from','go', 'make', 's', 'covid', 'would', 'corona' "covid", "19","corona", "pandemic", "covid19",\
                   "just", "don", 'amp', 'virus','like', 'cases', 'time','cummings', 'know', 'think',\
                   'need', 'did', 'going', 've','got', 'coronavirus', 'let', 'days','doing', 'didn', 'sir', 'states',\
                   'come', 'today', 'thing','look','said', 'total', 'll', 'take','see', 'self','does', 'covid_19',\
                   'getting', 'even', 'having','best','died', "people", "eid"])

# Define functions for tokenization, stopwords, bigrams, trigrams and lemmatization
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN']): # Optional: 'ADJ', 'VERB', 'ADV'
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# MAIN STARTS HERE
with open('corona.csv', 'r') as tweet_file:
    tweets_content = list()
    tweets = csv.reader(tweet_file)

    for row in tweets:
        tweets_content.append(row[3])

    # Create a list with all the contents of all tweets
    tweets_content = tweets_content
    data = tweets_content

    # Text cleaning
    for sent in data:
        sent = sent.strip()
    data = [re.sub('@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\'', '', sent) for sent in data]
    data = [re.sub('#', '', sent) for sent in data]

    # Tokenize text
    data_words = list(sent_to_words(data))

    # See tokenized Tweet example
    #print(data_words[:1], '\n')

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    #print(trigram_mod[bigram_mod[data_words[0]]])

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN'])   # Optional: 'ADJ', 'VERB', 'ADV'

    # See lemmatized Tweet example
    #print(data_lemmatized[:1])

    # Create Dictionary (unique id for each word)
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency (mapping of word_id and word_frequency)
    corpus = [id2word.doc2bow(text) for text in texts]

    # See examples
    #print(corpus[:1)
    #print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=5,   # Change the number of topics to your liking (more topics will take longer!)
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    # Print the keywords of the topics and the weightage(importance/relevance) of each keyword
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus[:100]]    # Change the number of tweets you want to analyze (more tweets will take longer)

    # Perplexity and Coherence provide a convenient measure to judge how good a given topic model is
    # Compute Model Perplexity
    #print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Topic Coherence Score
    #coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    #coherence_lda = coherence_model_lda.get_coherence()
    #print('\nCoherence Score: ', coherence_lda)

    # Visualize the topics (will save a html file in given folder)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    #print(vis.topic_order)
    pyLDAvis.save_html(vis, 'LDA_Visualization.html')
    print('\nDONE! Check your visualisation file')

