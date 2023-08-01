import os
import re
import arxiv
import pandas as pd
import requests
import PyPDF2
from tqdm import tqdm

def query_arxiv(query, num_papers):
    all_data = []
    query = query.strip('\"')
    search = arxiv.Search(
        query=query,
        max_results=int(num_papers),
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    for result in search.results():
        temp = ["", "", "", "", ""]
        temp[0] = result.title
        temp[1] = result.published
        temp[2] = result.entry_id
        temp[3] = result.summary
        temp[4] = result.pdf_url
        all_data.append(temp)
    column_names = ['Title', 'Date', 'Id', 'Summary', 'URL']
    df = pd.DataFrame(all_data, columns=column_names)
    return df

def download_papers(paper_titles, pdf_urls, save_path):
    paths = []
    for pdf_url, title in tqdm(zip(pdf_urls, paper_titles), total = len(pdf_urls)):
        if not os.path.exists(save_path + title + ".pdf"):
            response = requests.get(pdf_url)
            title = title.replace('/', '_')
            with open(save_path + title + ".pdf", 'wb') as file:
                file.write(response.content)
        paths.append(save_path + title + ".pdf")
    return paths

def pdf_to_text(file_path, save_path, title):
    title = title.replace('/','_')
    #print(title)
    # create a pdf file object
    pdfFileObj = open(file_path, 'rb')

    # create a pdf reader object
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    # create a page object
    file = open(save_path + title + ".txt", "w")
    for page_num in range(len(pdfReader.pages)):
        pageObj = pdfReader.pages[page_num]

        # extract text from page
        text = pageObj.extract_text()

        # save to a text file for later use
        # copy the path where the script and pdf is placed
        file.writelines(text)
    # closing the text file object
    file.close()

    # closing the pdf file object
    pdfFileObj.close()
    return save_path + title + ".txt"

def extract_introduction(full_file_path, save_path, title):
    title = title.replace('/', '_')
    with open(full_file_path) as f:
        lines = f.readlines()
    start = -1
    end = -1
    for i, line in enumerate(lines):
        if re.match(r".*i\s*ntroduction.*", line.lower()):
            if start == -1:
                start = i
        elif re.match(r".*(\d|(ii?))\s*([.]\s*)?((m\s*ethod)|(r\s*elated work).*)", line.lower()):
            end = i
            break
    f.close()
    if -1 in [start, end]:
        raise Exception("Problem in finding the Introduction")
    parsed_introduction = parser(lines[start+1:end+1])
    with open(save_path+title+".txt", "w") as f:
        f.write(parsed_introduction)

    return lines[start], lines[end]

def parser(text_lines):
    text = ""
    for line in text_lines:
        parsed_line = line.replace("\n", " ")
        parsed_line = parsed_line.replace("-", "")
        text += parsed_line

    return text

def list_files(directory):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    return file_names

def create_csv(path):
    abstract_folder_path, introduction_folder_path = path+"abstracts/", path+'introductions/'
    titles = []
    abstracts = []
    introductions = []
    abstracts_files = list_files(abstract_folder_path)
    introduction_files = list_files(introduction_folder_path)
    for file_path in abstracts_files:
        if not file_path in introduction_files:
            print("this abstract has no match: "+file_path)
        else:
            titles.append(file_path)
            with open(abstract_folder_path+file_path, 'r') as abstract_file:
                abstract = abstract_file.read()
                abstracts.append(abstract)
            with open(introduction_folder_path+file_path, 'r') as introduction_file:
                introduction = introduction_file.read()
                introductions.append(introduction)
    df = pd.DataFrame({'titles': titles, 'abstract': abstracts, 'introduction': introductions})
    df.to_csv(path+"abstracts_introductions.cvs")
    return df

def main():
    #topic = input("Enter the topic you need to search for : ")
    #num_papers = input("Enter the max number of papers: ")
    save_path = '/Users/wrystrn/Documents/ANLP/final_project/'
    # query = "noisy labels identification using neural networks ensemble"
    query = "quantum computing"
    if not os.path.exists(save_path+query):
        os.mkdir(save_path+query)
    if not os.path.exists(save_path+query+"/pdfs/"):
        os.mkdir(save_path+query+"/pdfs/")
    if not os.path.exists(save_path + query + "/text/"):
        os.mkdir(save_path + query + "/text/")
    if not os.path.exists(save_path+query+"/introductions/"):
        os.mkdir(save_path+query+"/introductions/")
    if not os.path.exists(save_path+query+"/abstracts/"):
        os.mkdir(save_path+query+"/abstracts/")
    num_papers = 1800
    print(query)
    df = query_arxiv(query, num_papers)
    abstracts = df['Summary']
    file_paths = download_papers(df['Title'], df['URL'], save_path+query+"/pdfs/")
    titles = []
    starts = []
    ends = []
    failed = []
    for file_path, title, abstract in tqdm(zip(file_paths, df['Title'], abstracts), total = len(file_paths)):
        title = title.replace('/', '_')
        try:
            text_path = pdf_to_text(file_path, save_path + query + "/text/", title)
            start, end = extract_introduction(text_path, save_path+query+'/introductions/', title)
            with open(save_path + query + '/abstracts/' + title + ".txt", "w") as f:
                f.write(abstract)
            titles.append(title)
            starts.append(start)
            ends.append(end)
        except:
            failed.append(title)

    for file_path in failed: #the code failed to process these files/extract introduction - the list is automatically created
        os.remove(save_path + query + "/pdfs/" + file_path + ".pdf")
    # additional_failed = ['Tooth Instance Segmentation from Cone-Beam CT Images through Point-based Detection and Gaussian Disentanglement',
    #                      'Semantic Representation and Inference for NLP',
    #                      'AssemblyNet: A Novel Deep Decision-Making Process for Whole Brain MRI Segmentation',
    #                      'Bootstrapping Deep Neural Networks from Approximate Image Processing Pipelines',
    #                      'Neural Graph Embedding Methods for Natural Language Processing',
    #                      'The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks',
    #                      'Logic Tensor Networks for Semantic Image Interpretation',
    #                      'EGC: Image Generation and Classification via a Diffusion Energy-Based Model',
    #                      'Going Deeper into Semi-supervised Person Re-identification',
    #                      'Learning Two Layer Rectified Neural Networks in Polynomial Time',
    #                      'Robust Hierarchical Graph Classification with Subgraph Attention',
    #                      'Exploring the importance of context and embeddings in neural NER models for task-oriented dialogue systems',
    #                      'Super-resolution and denoising of fluid flow using physics-informed convolutional neural networks without high-resolution labels',
    #                      'Deep tree-ensembles for multi-output prediction',
    #                      'Exploring difference in public perceptions on HPV vaccine between gender groups from Twitter using deep learning',
    #                      'Learning to Rectify for Robust Learning with Noisy Labels','Data Augmentation of Wearable Sensor Data for Parkinson\'s Disease Monitoring using Convolutional Neural Networks',
    #                      'PAL : Pretext-based Active Learning',
    #                      'Task-specific Word Identification from Short Texts Using a Convolutional Neural Network',
    #                      'Elucidating Meta-Structures of Noisy Labels in Semantic Segmentation by Deep Neural Networks',
    #                      'Image Classification with Deep Learning in the Presence of Noisy Labels: A Survey',
    #                      'Balanced Symmetric Cross Entropy for Large Scale Imbalanced and Noisy Data'] #the code extracted the introduction unsuccesfully - this list is manually created
    # for file_path in additional_failed:
    #     os.remove(save_path + query + "/pdfs/" + file_path + ".txt")
    #     os.remove(save_path + query + "/text/" + file_path + ".txt")
    #     os.remove(save_path + query + "/introductions/" + file_path + ".txt")
    #     os.remove(save_path + query + "/abstracts/" + file_path + ".txt")

    print(len(failed))
    # print(len(additional_failed))
    df = pd.DataFrame({'titles': titles, 'starts': starts, 'ends': ends})
    df.to_csv(save_path+query+'/ExtractedIntroductionsData.csv')


if __name__ == "__main__":
    main()
