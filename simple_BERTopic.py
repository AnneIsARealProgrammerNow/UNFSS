import pandas as pd
import numpy as np
import pickle
import os

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA

from extract_text import DATA_FOLDER

import time
t0 = time.time()

MODEL = 'paraphrase-multilingual-mpnet-base-v2' #'distiluse-base-multilingual-cased-v1'
OUTPUT_FOLDER = r'C:\Users\siets009\OneDrive - Wageningen University & Research\UNFSS\output\final'
NR_TOPICS = 52

def embed(texts, model):
    """ Because the embedding takes up quite some compute time, we only want to
        do it once (or once per model anyway). 
    """
    print('\nStarting embedding')
    embeddings_dir = os.path.join(DATA_FOLDER, 'embeddings')
    if not os.path.exists(embeddings_dir):
        os.mkdir(embeddings_dir)
    
    #Run the embedding
    transformer = SentenceTransformer(model)
    embeddings = transformer.encode(texts,  show_progress_bar=True)
    
    #Save as a pickle
    save_path = os.path.join(embeddings_dir, f'embedding_{model}.pkl')
    with open(save_path, 'wb') as embedding_file:
        pickle.dump(embeddings, embedding_file)
    print(f'Embeddings done and saved at {save_path}') 
    return embeddings
    
    
def run_topic_model(df, embeddings, num_topics, model,
                save = 'words_only', plot=True):
    print(f'\nStarting the topic model for {num_topics} topics')
    global nr_topics
    nr_topics = num_topics
    
    texts = df['text'].to_list()
    
    output_file = os.path.join(OUTPUT_FOLDER, f'{nr_topics}_model')
    output_df =  os.path.join(OUTPUT_FOLDER, f'{nr_topics}_df.csv')
    
    #Implementing a few of the listed best practices and suggestions
    representation_model = KeyBERTInspired()
    if nr_topics > 50: 
        n_neighbors = 12
        cluster_size = 10
        min_samples = 7
    else:
        n_neighbors = 20
        cluster_size = 14
        min_samples = 9
    dim_model = UMAP(n_neighbors=n_neighbors, n_components=16, min_dist=0.0, metric='cosine')
    # BERTopic then uses HDBScan to select clusters of similar documents
    # The hyper-paramters here can matter quite a bit but vary by nr_topics
    # Intuitively, lower nr_topics can allow for higher min_cluster_size
    # Lowering min_samples then can reduce outliers (but might lead to less "pure" topics)
    # In this case, cluser size of 15 works OK until about 50 topics; after that, they need to be smaller
    # Finally, 'leaf' instead of 'eom' reduces the chance of getting one big topic and many small ones
    hdbscan_model = HDBSCAN(min_cluster_size=cluster_size, min_samples = min_samples,
                            metric='euclidean', 
                            cluster_selection_method='leaf',
                            prediction_data=True)
    # Use the representation model in BERTopic on top of the default pipeline.
    # This help to reduce stopwords, as does ctfidf model called below
    representation_model = KeyBERTInspired()
    
    #Do the topic model
    topic_model = BERTopic(embedding_model=SentenceTransformer(model), 
                           nr_topics=nr_topics,
                           language='multilingual',
                           hdbscan_model=hdbscan_model,
                           representation_model=representation_model,
                           umap_model=dim_model,
                           ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True)
                           #seed_topic_list = list(CATEGORIES_KEYWORDS_DICT.values())
                           )
    topics, probs = topic_model.fit_transform(texts)
    
    n_unassigned = len([n for n in topic_model.topics_ if n ==-1])
    print(f"Before outlier reduction there are {n_unassigned} unassigned paragraphs")
    
    if n_unassigned >0:
        # Refine the topics by reducing outliers.
        # This is a nice idea in theory, but in practice does not seem to work well
        # -- to make it work, the threshold has to be unreasonably low
        # Also, multilingual necessitates strategy = embeddings, which works even worse
        topic_model._create_topic_vectors()
        new_topics = topic_model.reduce_outliers(texts, topic_model.topics_, strategy="embeddings", embeddings=embeddings, threshold =  0.4)
        if len([n for n in new_topics if n ==-1]) < n_unassigned:
            topic_model.update_topics(texts, topics=new_topics, representation_model=representation_model,
                                  ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True))
            print(f"After outlier reduction there are {len([n for n in topic_model.topics_ if n ==-1])} unassigned paragraphs")
        else: 
            print("Outlier reduction unsucessful")
    print('Topic model done')
    
    # Finally save.
    if save: 
        topic_model.save(output_file)
        print(f'Topic model saved at {output_file}')
        df['topic_nr'] = topic_model.topics_
        df.to_csv(output_df, encoding='utf-8')
    
    if plot:
        reduced_embeddings = plot_and_save_reduced(topic_model, embeddings, texts, df, n_neighbors)
    else:
        reduced_embeddings = None
    
    # Potentially save some additional info.
    if save:
        topic_df = get_and_save_words_per_topic(topic_model)
        if save != 'words_only':
            get_and_save_representative_pages_per_topic(texts, topic_model.representative_docs_, topic_df)
            #TODO: update?
            #get_and_save_topic_distribution(topic_model, texts, topic_df, df)
        
    return(topic_model)

def rescale(x, inplace=False):
    """ Rescale an embedding so optimization will not have convergence issues.
    """
    if not inplace:
        x = np.array(x, copy=True)
    x /= np.std(x[:, 0]) * 10000
    return x
        
        
def plot_and_save_reduced(topic_model, embeddings, paragraphs, paragraph_df, n_neighbors):
    print(f"{int(time.time() - t0)}s - Now reducing embeddings for plotting.")
    # Starting with PCA embeddings is mainly a time-saver
    #pca_embeddings = rescale(PCA(n_components=2).fit_transform(embeddings))
    
    #Slightly higher values for n_neighbours to encourage a more general view
    reduced_embeddings = UMAP(n_neighbors=30, n_components=2, min_dist=0.01,
                              metric='cosine', #init=pca_embeddings
                              ).fit_transform(embeddings)
    #Combine this with the article information
    reduced_df = pd.DataFrame(reduced_embeddings, columns = ['x', 'y'])
    combined_df = pd.concat([paragraph_df, reduced_df], axis = 1)
    #Add the topic assignment (-1 is an outlier) and save
    topic_nr_df = pd.DataFrame(topic_model.topics_, columns = ['topic_nr'])
    combined_df = pd.concat([combined_df, topic_nr_df], axis=1)
    df_path = os.path.join(OUTPUT_FOLDER, f'{nr_topics}_reduced_embeddings_per_para.csv')
    combined_df.to_csv(df_path, encoding = 'utf-8')
    
    # Finally plot.
    fig = topic_model.visualize_documents(paragraphs, reduced_embeddings=reduced_embeddings, hide_document_hover=True, hide_annotations=True)
    fig_path = os.path.join(OUTPUT_FOLDER, f'{nr_topics}_output.html')
    fig.write_html(fig_path)
    fig.show()
    print(f"{int(time.time() - t0)}s - Saved reduced embeddings at {df_path} and a plot at {fig_path}")
    return reduced_embeddings

def get_and_save_representative_pages_per_topic(paragraphs, representative_docs, topic_df):
    del representative_docs[-1] #outliers
    rdocs_df =  pd.DataFrame.from_dict(representative_docs, orient='index',
                                       columns = ['Most_representative_paragraph1', 'Most_representative_paragraph2', 'Most_representative_paragraph3'])
    combined_df = pd.concat([topic_df.reset_index(), rdocs_df], axis = 1)

    df_path = os.path.join(OUTPUT_FOLDER, f'{nr_topics}_representative_docs_per_topic.csv')
    combined_df.to_csv(df_path, encoding = 'utf-8')

    print(f"{int(time.time() - t0)}s - Saved dict with representative words per topic to {df_path}.")


def get_and_save_words_per_topic(topic_model):
    words_dict = {}
    for label_id, label in topic_model.topic_labels_.items():
        if label_id == -1:
            continue
        representative_words = [line[0] for line in topic_model.topic_representations_[label_id]]
        words_dict[label] = representative_words
    df = pd.DataFrame.from_dict(words_dict, orient='index')
    df_path = os.path.join(OUTPUT_FOLDER, f'{nr_topics}_representative_words.csv')
    df.to_csv(df_path, encoding = 'utf-8')
    print(f"{int(time.time() - t0)}s - Saved dict with representative words per topic to {df_path}.")
    
    return(df)
#%% RUN    
if __name__ == '__main__':
    df = pd.read_csv(os.path.join(DATA_FOLDER, 'text_100.csv'), encoding = 'utf-8')
    try:
        save_path = os.path.join(DATA_FOLDER, "embeddings", f'embedding_{MODEL}.pkl')
        with open(save_path, 'rb') as embedding_file:
            embeddings = pickle.load(embedding_file)
            print(f"Embedding loaded for model {MODEL}")
    except:
        embeddings = embed(df['text'].to_list(), MODEL)
        
    if type(NR_TOPICS) == int:
        topic_model = run_topic_model(df, embeddings, NR_TOPICS, MODEL, save=True)
    elif type(NR_TOPICS) == list:
        for n in NR_TOPICS:
            topic_model = run_topic_model(df, embeddings, n, MODEL, save=True)
    else:
        print("Invalid TOPIC_NR input. Either input a single int or a list of ints")
    

    
    