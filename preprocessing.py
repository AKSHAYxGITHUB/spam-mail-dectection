import re
import string
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required NLTK resources silently
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """Applies full NLP pipeline for text cleaning."""
        if not isinstance(text, str):
            text = str(text)
            
        # 1. Lowercasing
        text = text.lower()
        
        # 2. URL removal
        text = re.sub(r'http\S+|www\S+|https\S+', ' link ', text, flags=re.MULTILINE)
        
        # 3. Email removal
        text = re.sub(r'\S+@\S+', ' email ', text)
        
        # 4. Number normalization
        text = re.sub(r'\d+', ' number ', text)
        
        # 5. Emoji removal (simple regex for typical non-ascii, though we might want to keep some, 
        # let's just keep basic ascii and punctuation for now)
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # 6. Punctuation removal
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        
        # 7. Tokenization and stopword/lemmatization
        words = text.split()
        cleaned_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words and len(word) > 1
        ]
        
        return ' '.join(cleaned_words)

class DataAugmenter:
    """Implement simple data augmentation using synonym replacement (EDA style)."""
    
    def __init__(self):
        pass

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    def synonym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in stopwords.words('english')]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n: 
                break
        return new_words

    def augment_text(self, text, n=2):
        """Augments text by replacing 'n' words with synonyms."""
        words = text.split()
        if len(words) < 3:
            return text # Don't augment very short texts
        
        augmented_words = self.synonym_replacement(words, n)
        return ' '.join(augmented_words)

if __name__ == '__main__':
    preprocessor = TextPreprocessor()
    augmenter = DataAugmenter()
    
    sample_text = "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!"
    cleaned = preprocessor.clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
    
    augmented = augmenter.augment_text(cleaned)
    print(f"Augmented: {augmented}")
