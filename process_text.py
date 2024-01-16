import os
import pandas as pd
from tqdm import tqdm

import time
import re

from nltk.tokenize import sent_tokenize

from extract_text import DATA_FOLDER, TEXT_FILE

SENTENCE_MIN_WORDS = 3
SENTENCE_MIN_CHARACTERS = 10

#%% Functions
class Preprocess:

    def __init__(self):
        #The languages recorded in the file name do not match NLTK's convention
        self.language_map = {'ENG': 'english',
                             'FR': 'french',
                             'POR': 'portuguese',
                             'ESP': 'spanish',
                             'ARAB': 'arabic',
                             'RUS': 'russian'}
        
        
    def minimal_processing(self, df,  country_names, country_adjectives,
                           text_col = 'text', lang_col = 'language',
                           suffix_original = False):
        """ We do not want heavy pre-processing (e.g. stopword removal) at this stage
            as we will use sentence embedders later. 
            However, some minimal pre-processing is probably useful.
            Optionally, keep the original text in a column with specified suffix
        """
        print('Starting pre-processing')
        
        processed_texts = []
        
        df.dropna(subset = [text_col], inplace = True)
        
        # If we want to keep a column with the original text, input a suffix string
        if suffix_original:
            assert type(suffix_original) == str 
            original_col = f"{text_col}_{suffix_original}"
            df[original_col] = df[text_col]
        
        for i,row in tqdm(df.iterrows()):
            messy_text = row[text_col]
            language = self.language_map[row[lang_col]]
            # Deal with the new line character and the hyphens.
            messy_text = messy_text.replace('\n', ' ').replace('- ', '-')
            # Remove urls, e-mail.
            messy_text = re.sub(r'http\S+', '', messy_text)
            messy_text = re.sub(r'\S+@\S+(?:\.\S+)+', '', messy_text)
            
            # Replace country names and adjectives -- as embedding is multilingual
            # it doesn't matter much that we replace all with an English word
            # Would still be better for speed purposes to have them be language specific lists though
            names = [f'{n}' for n in country_names]
            patterns_names = [re.compile(name) for name in names]
            adjectives = [f'{n}' for n in country_adjectives]
            patterns_adjectives= [re.compile(adjective) for adjective in adjectives]
            for pattern in patterns_names:
                messy_text = pattern.sub("Country", messy_text)
            for pattern in patterns_adjectives:
                messy_text = pattern.sub("Country's", messy_text)

            
            
            #Remove some sentences -- need to tokenize first
            if language == 'arabic':
                tokenized_sentences = self._sentence_tokenize_arabic(messy_text)
            else:
                tokenized_sentences = sent_tokenize(messy_text, language=language)
            # Remove sentences if more than a third of the sentence is numeric.
            tokenized_sentences = [sentence for sentence in tokenized_sentences if len(sentence) > 0]
            tokenized_sentences = [sentence for sentence in tokenized_sentences
                                   if (len(''.join(re.findall(r'\d+', sentence))) / len(sentence)) < 0.333]
            # Remove short sentences (keep long as punctuation may simply be missing)
            tokenized_sentences = [sentence for sentence in tokenized_sentences
                                   if (len(sentence) >= SENTENCE_MIN_CHARACTERS and len(sentence.split()) >= SENTENCE_MIN_WORDS)]
            
            #Return the sentences as one text again
            processed_texts.append(' '.join(tokenized_sentences))
        df[text_col] = processed_texts
        return(df)
           
           
    @staticmethod
    def _sentence_tokenize_arabic(messy_text):
        """ NLTK doesn't have arabic, so a super simple one
        """
        tokenized_sentences = sent_tokenize(messy_text)
        sentences = []
        for initial_sentence in tokenized_sentences:
            sentences.extend(initial_sentence.split('.'))
        return sentences        
    
    def homogenize_length(self, df, ltarget = 100,
                          lmin = 50, #Set to False to skip
                          lmax = 120, #Set to False to skip
                          text_col = 'text',
                          index_col = 'id',
                          doc_col = 'file'):
        """ Combines texts for each document (by unique doc_col)
            untill it would exceed ltarget words.
            e.g. for ltarget=20, a doc with 10, 9 and 4 words returns
            a df with 19 words in the first row and 4 words in the second. 
            Optionally, all texts with > lmax words are split roughly in half,
            splitting at a period if one is found close to the halfway point.
            lmax is kept lower than max_length of most embedders to save memory later
            And to prevent mixed topics in overly long texts
        """
        print(f'\nStarting length homogenization with target length {ltarget}')
        
        #Add a column with the number of words 
        df['nr_words'] = df[text_col].str.findall(r'(\w+)').str.len()
        
        #We will use .agg(), so assign 'first' for all columns to keep them
        operations = {i: 'first' for i in df.columns}
        operations[text_col] = ' '.join
        operations['nr_words'] = sum
        

        
        #We will create a list of dfs to concat at the end 
        #Not the most memory efficient, but the whole set is only 1.5 GB anyway
        # & it's better than constantly updating the whole df
        dfs = []
        
        #Loop over each unique document
        for doc_id in tqdm(df[doc_col].unique()):
            ddf = df[df[doc_col] == doc_id]
            #Now, we do something a bit hack-y: 
            #we divide the cumulative wordcount by our target length
            #and group by the floor of this nr, then combine each group
            #the effect is combining all up to and including the text which exceeds ltarget
            ddf = ddf.groupby(ddf['nr_words'].cumsum() // ltarget, as_index=False
                              ).agg(operations)
            
            #The above works fine most of the time, but in some cases, very short
            #Texts are still kept separate if the next text is longer than ltarget
            #Join this to the next text block
            if lmin:
                for tries in range(2): #Do two passes only; takes a while & many are at edge of range
                    ddf.sort_index().reset_index(drop=True, inplace=True)
                    for i, row in ddf[ddf['nr_words'] < lmin].iterrows():
                        try:
                            text = " ". join([row[text_col], ddf.iloc[i+1][text_col]])
                            nr_words = row['nr_words'] + ddf.loc[ddf.index == i+1, 'nr_words']
                            ddf.loc[ddf.index == i + 1, text_col] = text
                            ddf.loc[ddf.index == i + 1, 'nr_words'] = nr_words
                        except IndexError: #Append to previous instead
                            try:
                                text = " ". join([row[text_col], ddf.iloc[i-1][text_col]])
                                nr_words = row['nr_words'] + ddf.loc[ddf.index == i-1, 'nr_words']
                                ddf.loc[ddf.index == i - 1, text_col] = text
                                ddf.loc[ddf.index == i - 1, 'nr_words'] = nr_words
                            except:
                                continue
                        #Drop the too-short row
                        ddf.drop(i, inplace=True)
                
            
            #Similarly, the above sometimes results in overly long texts,
            #e.g. if the last added text is very long.
            #These are split up again, trying to split on a "." near the middle
            if lmax:
                while len(ddf[ddf['nr_words'] > lmax]) >=1:
                    ddf.sort_index().reset_index(drop=True, inplace=True)
                    for i, row in ddf[ddf['nr_words'] > lmax].iterrows():
                        text = row[text_col]
                        middle = len(text)//2
                        split = text.rfind(".", middle-(middle//3), middle+middle//3)
                        if split == -1: #If none found, split on space instead
                            split = text.rfind(" ", middle-(middle//3), middle+middle//3)
                        if split == -1:
                            split = middle
                        
                        #Insert the split text into the original ddf
                        ddf.loc[ddf.index == i, text_col] = text[:split+1]
                        ddf.loc[ddf.index == i, 'nr_words'] = len(re.findall(r'(\w+)', text[:split+1]))
                        ddf.loc[i+0.5] = row
                        ddf.loc[ddf.index == i+0.5, text_col] = text[split+1:]
                        ddf.loc[ddf.index == i+0.5, 'nr_words'] = len(re.findall(r'(\w+)', text[split+1:]))
            
            dfs.append(ddf)
            
            
        #Return all the ddfs as one df
        out = pd.concat(dfs).reset_index(drop=True)
        print('HOMOGENIZATION DONE')
        print(out['nr_words'].describe())
        return(out)

#%% RUN
if __name__ == '__main__':
    df = pd.read_csv(os.path.join(DATA_FOLDER, TEXT_FILE), encoding = 'utf-8')
    #It's useful to have a unique ID per text block  & the language, which is in the file name
    df[['country', 'year', 'language']] = df['file'].str.split('_', expand=True) 
    df['id'] = df[['file', 'block_id']].apply(lambda x: '_'.join(x.astype(str).values), axis=1)
    
    #Read in lists of country names and country adjectives
    country_names = set()
    country_adjectives = set()
    english = pd.read_csv(os.path.join(DATA_FOLDER, "CountryNames.csv"), encoding='utf-8')
    country_names.update(english['Country'])
    country_adjectives.update(english['Adjectivals'])
    spanish= pd.read_csv(os.path.join(DATA_FOLDER, "CountryNames_Spanish.csv"), encoding='utf-8')
    country_names.update(spanish['Country_simple'])
    country_adjectives.update(spanish['Gentilicio_first'])
    country_adjectives.update(spanish['Gentilioco_alternative'])
    french = pd.read_csv(os.path.join(DATA_FOLDER, "CountryNames_French.csv"), encoding='utf-8')
    country_names.update(french['Pays'])
    country_adjectives.update(french['Gentilés_m'])
    country_adjectives.update(french['Gentilés_f'])
    arabic = pd.read_csv(os.path.join(DATA_FOLDER, "CountryNames_Arabic.csv"), encoding='utf-8')
    country_names.update(arabic['Country_Arabic'])
    
    # Do some limted pre-processing, incl. dropping empty text columns
    processor = Preprocess()
    df_preprocessed = processor.minimal_processing(df, country_names, country_adjectives)
    
    #homogenize length first to 100 words or so (default) which works well with sentence embedders
    df_100 = processor.homogenize_length(df_preprocessed)
    df_100.to_csv(os.path.join(DATA_FOLDER, 'text_100.csv'), index=False, encoding = 'utf-8')
    # #We may want to try bigger embedders also => try around just under 500
    # df_500 = processor.homogenize_length(df_preprocessed, 
    #                                      ltarget = 490, lmin = 350, lmax = 510)
    # df_500.to_csv(os.path.join(DATA_FOLDER, 'text_500.csv'), index=False, encoding = 'utf-8')
