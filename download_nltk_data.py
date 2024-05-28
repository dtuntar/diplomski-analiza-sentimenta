import nltk

packages = [
    'punkt',
    'averaged_perceptron_tagger',
    'vader_lexicon',
    'stopwords'
]

for package in packages:
    nltk.download(package, download_dir='/app/nltk_data')
