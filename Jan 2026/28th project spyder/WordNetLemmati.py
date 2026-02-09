import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

paragraph = '''AI, machine learning and deep learning are common terms in enterprise
IT and sometimes used interchangeably, especially by companies in their marketing materials.
But there are distinctions. The term AI, coined in the 1950s, refers to the simulation of human
intelligence by machines. It covers an ever-changing set of capabilities as new technologies
are developed. Technologies that come under the umbrella of AI include machine learning and
deep learning.'''

sentences = nltk.sent_tokenize(paragraph)

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [
        lemmatizer.lemmatize(word.lower())
        for word in words
        if word.lower() not in stop_words and word.isalpha()
    ]
    sentences[i] = ' '.join(words)

for s in sentences:
    print(s)
