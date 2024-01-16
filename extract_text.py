import os
import pandas as pd
from tqdm import tqdm

#PDF processing imports under main to preven dependancy issues

import win32com.client as win32
from win32com.client import constants



#Assuming the docs are saved under [DATA_FOLDER]\documents
DATA_FOLDER = r'C:\Users\siets009\OneDrive - Wageningen University & Research\UNFSS\data'
TEXT_FILE = r'texts.csv'

#%% Functions

def create_df(DATA_FOLDER): 
    """
    Loop over the data folder, convert all pdf and word-files to text.
    Returns pandas dataframe with original filename, page and text in separate cols
    """
    tdf_list = []
    DOCUMENT_FOLDER = os.path.join(DATA_FOLDER, 'documents')
    
    doc_extensions = ['.doc', '.docx']
    
    for file in tqdm(os.listdir(DOCUMENT_FOLDER)): 
        FILE_PATH = os.path.join(DOCUMENT_FOLDER, file)
        assert os.path.isfile(FILE_PATH)
        file_name, extension = os.path.splitext(file)
        
        if extension == '.pdf':
            text_dict = extract_from_pdf(FILE_PATH)
        elif extension in doc_extensions:
            if extension == '.doc':
                FILE_PATH = save_as_docx(FILE_PATH, delete_old=True)
            text_dict =  extract_from_doc(FILE_PATH)
            
        # Convert to temporary dataframe
        tdf = pd.DataFrame.from_dict(text_dict, orient = 'index', 
                                     columns = ['text'])
        tdf.reset_index(names = 'block_id', inplace=True)
        tdf['file'] = file_name
        tdf['original_type'] = extension
        tdf_list.append(tdf)
    #Create one big df from the list of tdfs
    df = pd.concat(tdf_list)
            
    return df.reset_index(drop=True)

def extract_from_pdf(FILE_PATH):
    out = {}
    pdf = pdfplumber.open(FILE_PATH) 
    for i, page in enumerate(pdf.pages):
        out[i] = page.extract_text()
    return(out)

def extract_from_doc(FILE_PATH):
    out = {}
    doc = Document(FILE_PATH)
    for i, para in enumerate(doc.paragraphs):
        out[i] = para.text
    return(out)
        

def save_as_docx(path, delete_old=False):
    # Opening MS Word
    word = win32.gencache.EnsureDispatch('Word.Application')
    doc = word.Documents.Open(path)
    doc.Activate()

    # Rename path with .docx
    new_file_abs = os.path.abspath(path)
    new_file_abs = re.sub(r'\.\w+$', '.docx', new_file_abs)

    # Save and Close
    word.ActiveDocument.SaveAs(
        new_file_abs, FileFormat=constants.wdFormatXMLDocument)
    doc.Close(False)
    
    if delete_old:
        os.remove(path)
        
    return new_file_abs

#%% RUN
if __name__ == '__main__':
    import pdfplumber
    import re
    from docx import Document
    
    df = create_df(DATA_FOLDER)
    df.to_csv(os.path.join(DATA_FOLDER, TEXT_FILE), encoding = 'utf-8', index= False)
        