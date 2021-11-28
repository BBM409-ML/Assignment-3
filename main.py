import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split


def get_frequencies(vectorizer, xtrain):
	y = vectorizer.fit_transform(xtrain)
	doc_array = y.toarray()

	total_freqs = dict()
	frequency_matrix = pd.DataFrame(data=doc_array, columns=vectorizer.get_feature_names_out())

	for columnName, columnData in frequency_matrix.iteritems():
		total = frequency_matrix[columnName].sum()
		total_freqs[columnName] = total

	print(total_freqs)

	return total_freqs


def get_frequencies_unigram(xtrain):
	vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b", lowercase=True)
	get_frequencies(vectorizer, xtrain)


def get_frequencies_unigram_without_stopwords(xtrain):
	vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b",
								 lowercase=True, stop_words=ENGLISH_STOP_WORDS)
	get_frequencies(vectorizer, xtrain)


def get_frequencies_bigram(xtrain):
	vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), token_pattern=r"(?u)\b\w+\b", lowercase=True)
	get_frequencies(vectorizer, xtrain)


def get_frequencies_bigram_without_stopwords(xtrain):
	vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), token_pattern=r"(?u)\b\w+\b",
								 lowercase=True, stop_words=ENGLISH_STOP_WORDS)
	get_frequencies(vectorizer, xtrain)


def get_frequencies_unigram_bigram(xtrain):
	vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", lowercase=True)
	get_frequencies(vectorizer, xtrain)


def main():
	df = pd.read_csv('emails.csv')
	value_counts = df["spam"].value_counts()

	x = df.text.values
	y = df.spam.values

	ham_count = value_counts.get(0)
	spam_count = value_counts.get(1)

	xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True)

	get_frequencies_unigram(xtrain)


main()
