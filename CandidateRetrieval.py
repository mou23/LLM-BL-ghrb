from utils import load_file, OpenaiClient, save_file, process_text,get_rel_file,combmnz, EmbeddingModel
from Configure import Configure
from tqdm import tqdm
import os
import codecs
import glob
import multiprocessing as mp
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-v3")
text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=510,
                    chunk_overlap=0,  # number of tokens overlap between chunks
                    length_function=lambda x: len(tokenizer.tokenize(x)),
                    separators=['\n\n', '\n', ' ', '']
                )

def read_codes(source,suffix='java'):
    files = glob.glob(f'{source}/**/*.{suffix}', recursive=True)
    codes = {}
    for i in files:
        if os.path.getsize(i): # and 'org/apache' in i:
            abs_path = os.path.relpath(i,source)
            abs_path = abs_path.replace('\\', '/')
            with codecs.open(i,encoding='utf-8',errors='ignore') as f:
                content = f.read()
                codes[abs_path] = content
        # elif os.path.getsize(i) and 'py' in i:
        #     abs_path = os.path.relpath(i,source)
        #     with codecs.open(i,encoding='utf-8',errors='ignore') as f:
        #         content = f.read()
        #         codes[abs_path] = content
    return codes


class CandidateRetrival:
    def __init__(self,args) -> None:
        self.project = args.project
        self.group = args.group
        self.args = args
        # self.S = Subjects()
        self.gitrepo = self.args.gitrepo
        self.file_type = 'Java' if self.group == 'Apache' else 'Python'
        self.output_dir = self.args.output_dir
        self.enhanced_report_path = self.args.report_path
        self.version2paths_path = self.args.version2paths_path
        self.version2shas_path = self.args.version2shas_path
        self.candidate_path = self.args.candidate_path
        self.candidate_list_path = self.args.candidate_list_path
        self.candidate_list_csv_path = self.args.candidate_list_csv_path
        self.class_method_names_path = self.args.class_method_names_path
        self.filtered_path = self.args.filtered_path
        self.rel_path = self.args.rel_path
        self.codename_path = self.args.codename_path
        self.candidate_name_path = self.args.candidate_name_path
        self.typelist = self.args.typelist
        
    def _extract_class_method(self,report):
        class_method_names = load_file(self.class_method_names_path)
        classnames = []
        methodnames = []
        report = re.sub(r'[^\w]', ' ', report)
        report = re.sub(r'\s+', ' ', report)
        # report = report.lower()
        report = report.split(' ')
        for i in class_method_names['classname']:
            if i in report:
                classnames.append(i)
        for i in class_method_names['methodname']:
            if i in report:
                methodnames.append(i)
        return ' '.join(classnames),' '.join(methodnames)
       
    def enhanced_report(self):
        reports = load_file(self.enhanced_report_path)        
        if 'r_summ' not in reports.columns:
            reports['r_summ']=""
        if 'r_cause' not in reports.columns:
            reports['r_cause']= ""
        r_summ_list = []
        r_cause_list = []
        file_method_list = []
        api_key = ''
        # api_base = ""
        agent = OpenaiClient(api_key=api_key)
        for _,r in tqdm(reports.iterrows(),total=len(reports)):
            r_summary,possible_cause=r['r_summ'],r['r_cause']
            if r_summary != "":
                r_summ_list.append(r_summary)
                r_cause_list.append(possible_cause)
                continue
            classnames,methodnames = self._extract_class_method(r.summary+' '+r.description)
            id = r["id"]
            summary = r.summary
            description = r.description
            prompt = "Here is a bug report, containing title and description.\ntitle: {}\ndescription: {}\n\nYou are a Software Test Engineer. Analyze the bug report above, summarize it and describe the functionality executed by the program when the bug is triggered. Only provide the keywords, separated by commas.\nSummary: \nFunctionality: "
            messages = [{'role': 'user','content': prompt.format(summary,description)}]
            r_summary,possible_cause = agent.summarize(model_name='gpt-4o-mini', messages=messages,temperature=0.2)
            if r_summary=="" or possible_cause=="":
                print(f'report {id} output missing')
            r_summ_list.append(r_summary)
            r_cause_list.append(possible_cause)
            file_method_list.append((classnames + ' '+ methodnames).strip())
        reports['r_summ'] = r_summ_list
        reports['r_cause'] = r_cause_list
        reports['file_method'] = file_method_list
        save_file(self.enhanced_report_path,reports)
            
    def generate_token_candidate(self):
        reports = load_file(self.enhanced_report_path)
        if os.path.exists(self.candidate_path):
            candidate_dict = load_file(self.candidate_path)
        else:
            candidate_dict = {}

        versions = list(set(reports['version'].tolist()))

        for version in tqdm(versions,total=len(versions),desc=f"{self.project}"):
            codes = read_codes(self.gitrepo, 'java')

            files = list(codes.keys())
            content = list(codes.values())

            with mp.Pool(mp.cpu_count()) as pool:
                content = pool.map(process_text, content)
            
            tokenized_corpus = [i.split(' ') for i in content]
            bm25 = BM25Okapi(tokenized_corpus)
            
            v_reports = reports[reports['version']==version]
            for _,r in v_reports.iterrows():
                id = r['id']
                if id not in candidate_dict:
                    candidate_dict[id]={}
                
                origin = process_text(r.summary+' '+r.description).split()
                score = bm25.get_scores(origin)
                index = np.argsort(-score)
                candidate_dict[id]['origin']=[files[i] for i in index[:100]]
                candidate_dict[id]['origin_score']=[score[i] for i in index[:100]]
                
                r_summ = process_text(r.file_method+' '+r.r_summ).split()
                score = bm25.get_scores(r_summ)
                index = np.argsort(-score)
                candidate_dict[id]['summ']=[files[i] for i in index[:100]]
                candidate_dict[id]['summ_score']=[score[i] for i in index[:100]]
                
                r_cause = process_text(r.file_method+' '+r.r_cause).split()
                score = bm25.get_scores(r_cause)
                index = np.argsort(-score)
                candidate_dict[id]['cause']=[files[i] for i in index[:100]]
                candidate_dict[id]['cause_score']=[score[i] for i in index[:100]]
                
                r_add = process_text(r.file_method+' '+r.r_summ+ ' '+ r.r_cause).split()
                score = bm25.get_scores(r_add)
                index = np.argsort(-score)
                candidate_dict[id]['add']=[files[i] for i in index[:100]]
                candidate_dict[id]['add_score']=[score[i] for i in index[:100]]
        save_file(self.candidate_path,candidate_dict)
        
    def generate_sen_candidate(self):
        reports = load_file(self.enhanced_report_path)
        if os.path.exists(self.candidate_path):
            candidate_dict = load_file(self.candidate_path)
        else:
            candidate_dict = {}
        versions = list(set(reports['version'].tolist()))
        with tqdm(total=len(reports),desc=f"{self.project}") as pbar:
            for version in versions:
                codes = read_codes(self.gitrepo, 'java')
                
                content = []
                files = []
                for f, code in codes.items():
                    chunks = text_splitter.split_text(code)
                    if chunks:
                        content.append(chunks)
                        files.append(f)

                corpus_emb = list(map(encode, content))
                v_reports = reports[reports['version']==version]
                for _,r in v_reports.iterrows():
                    id = r['id']
                    if id not in candidate_dict:
                        candidate_dict[id]={}
                    origin = text_splitter.split_text(f"{r.summary}. {r.description}")
                    report_emb = encode(origin)
                    score = list(map(lambda x: cosine_similarity(report_emb,x).max(),corpus_emb))
                    index = np.argsort(-np.array(score))
                    candidate_dict[id]['origin_sen']=[files[i] for i in index[:100]]
                    candidate_dict[id]['origin_sen_score']=[score[i] for i in index[:100]]
                    origin = [r.file_method+' '+r.r_summ]
                    report_emb = encode(origin)
                    score1 = list(map(lambda x: cosine_similarity(report_emb,x).max(),corpus_emb))
                    index = np.argsort(-np.array(score1))
                    candidate_dict[id]['summ_sen']=[files[i] for i in index[:100]]
                    candidate_dict[id]['summ_sen_score']=[score1[i] for i in index[:100]]
                    origin = [r.file_method+' '+r.r_cause]
                    report_emb = encode(origin)
                    score2 = list(map(lambda x: cosine_similarity(report_emb,x).max(),corpus_emb))
                    index = np.argsort(-np.array(score2))
                    candidate_dict[id]['cause_sen']=[files[i] for i in index[:100]]
                    candidate_dict[id]['cause_sen_score']=[score2[i] for i in index[:100]]
                    score = np.maximum.reduce([np.array(score1),np.array(score2)])
                    index = np.argsort(-score)
                    candidate_dict[id]['add_sen']=[files[i] for i in index[:100]]
                    candidate_dict[id]['add_sen_score']=[score[i] for i in index[:100]]
                    pbar.update(1)
        save_file(self.candidate_path,candidate_dict)
    
    def generate_candidates(self,n=50):
        reports = load_file(self.enhanced_report_path)
        if os.path.exists(self.candidate_path):
            candidate_dict = load_file(self.candidate_path)
        else:
            candidate_dict = {}

        versions = list(set(reports['version'].tolist()))
        with tqdm(total=len(reports), desc=f"{self.project}") as pbar:
            for version in versions:
                codes = read_codes(self.gitrepo, 'java')
                files = list(codes.keys())
                raw_content = list(codes.values())

                # Token-based preprocessing for BM25
                with mp.Pool(mp.cpu_count()) as pool:
                    processed_content = pool.map(process_text, raw_content)
                tokenized_corpus = [i.split(' ') for i in processed_content]
                bm25 = BM25Okapi(tokenized_corpus)

                # Sentence-based preprocessing for embeddings
                model = EmbeddingModel()
                chunked_content = []
                file_refs = []
                for f, code in codes.items():
                    chunks = text_splitter.split_text(code)
                    if chunks:
                        chunked_content.append(chunks)
                        file_refs.append(f)
                corpus_emb = list(map(model.encode, chunked_content))

                # Per-report scoring
                v_reports = reports[reports['version'] == version]
                for _, r in v_reports.iterrows():
                    id = r['id']
                    if id not in candidate_dict:
                        candidate_dict[id] = {}

                    # --- Token-level BM25 Scoring ---
                    origin = process_text(r.summary + ' ' + r.description).split()
                    score = bm25.get_scores(origin)
                    index = np.argsort(-score)
                    candidate_dict[id]['origin'] = [files[i] for i in index[:n]]
                    candidate_dict[id]['origin_score'] = [score[i] for i in index[:n]]

                    r_summ = process_text(r.file_method + ' ' + r.r_summ).split()
                    score = bm25.get_scores(r_summ)
                    index = np.argsort(-score)
                    candidate_dict[id]['summ'] = [files[i] for i in index[:n]]
                    candidate_dict[id]['summ_score'] = [score[i] for i in index[:n]]

                    r_cause = process_text(r.file_method + ' ' + r.r_cause).split()
                    score = bm25.get_scores(r_cause)
                    index = np.argsort(-score)
                    candidate_dict[id]['cause'] = [files[i] for i in index[:n]]
                    candidate_dict[id]['cause_score'] = [score[i] for i in index[:n]]

                    r_add = process_text(r.file_method + ' ' + r.r_summ + ' ' + r.r_cause).split()
                    score = bm25.get_scores(r_add)
                    index = np.argsort(-score)
                    candidate_dict[id]['add'] = [files[i] for i in index[:n]]
                    candidate_dict[id]['add_score'] = [score[i] for i in index[:n]]

                    # --- Sentence-level Embedding Scoring ---
                    origin = text_splitter.split_text(f"{r.summary}. {r.description}")
                    report_emb = model.encode(origin)
                    score = list(map(lambda x: cosine_similarity(report_emb, x).max(), corpus_emb))
                    index = np.argsort(-np.array(score))
                    candidate_dict[id]['origin_sen'] = [file_refs[i] for i in index[:n]]
                    candidate_dict[id]['origin_sen_score'] = [score[i] for i in index[:n]]

                    origin = [r.file_method + ' ' + r.r_summ]
                    report_emb = model.encode(origin)
                    score1 = list(map(lambda x: cosine_similarity(report_emb, x).max(), corpus_emb))
                    index = np.argsort(-np.array(score1))
                    candidate_dict[id]['summ_sen'] = [file_refs[i] for i in index[:n]]
                    candidate_dict[id]['summ_sen_score'] = [score1[i] for i in index[:n]]

                    origin = [r.file_method + ' ' + r.r_cause]
                    report_emb = model.encode(origin)
                    score2 = list(map(lambda x: cosine_similarity(report_emb, x).max(), corpus_emb))
                    index = np.argsort(-np.array(score2))
                    candidate_dict[id]['cause_sen'] = [file_refs[i] for i in index[:n]]
                    candidate_dict[id]['cause_sen_score'] = [score2[i] for i in index[:n]]

                    combined_score = np.maximum.reduce([np.array(score1), np.array(score2)])
                    index = np.argsort(-combined_score)
                    candidate_dict[id]['add_sen'] = [file_refs[i] for i in index[:n]]
                    candidate_dict[id]['add_sen_score'] = [combined_score[i] for i in index[:n]]

                    pbar.update(1)

        save_file(self.candidate_path, candidate_dict)
        model.close()
     
    def generate_list_df(self):
        reports = load_file(self.enhanced_report_path)
        candidate_dict = load_file(self.candidate_path)
        list_res = []
        rel_res = []
        for _,r in reports.iterrows():
            q_id = r['id']
            tmp_f = []
            for voter,v in candidate_dict[q_id].items():
                if voter.endswith('score'):
                    continue
                for i,s in enumerate(v[:50]):
                    list_res.append((q_id,voter,s,(i+1),candidate_dict[q_id][f"{voter}_score"][i],self.project))
                tmp_f.extend(v[:50])
            tmp_f = list(set(tmp_f))
            # print(list_res)
            for i in tmp_f:
                rel_i = i if self.file_type=='Python' else get_rel_file(i)
                if rel_i in r.files:
                    rel_res.append((q_id,0,i,1))
        df = pd.DataFrame(list_res,columns=['Query','Voter','Item','Rank','Score','Dataset'])
        df['Item'],namelist = pd.factorize(df['Item'])
        namelist = namelist.tolist()
        save_file(self.codename_path,namelist)
        df.to_pickle(self.candidate_list_path)
        df.to_csv(self.candidate_list_csv_path,index=False,header=False)
        new_rel_res = []
        for i in rel_res:
            new_rel_res.append((i[0],i[1],namelist.index(i[2]),i[3]))
        df = pd.DataFrame(new_rel_res,columns=['Query',0,'Item','Relevance'])
        df.to_csv(self.rel_path,index=False,header=False)

    def rank_aggregation(self):
        df = load_file(self.candidate_list_path)
        flag = df['Voter'].apply(lambda x: x in self.typelist)
        df = df[flag]
        df.to_csv(self.candidate_list_csv_path,index=False,header=False)
        codenames = load_file(self.codename_path)
        candidate_ids = combmnz(self.candidate_list_csv_path,f"{self.output_dir}/tmp",self.args.merge_type)

        candidate_names = {}
        for k,v in candidate_ids.items():
            # print(k,v)
            candidate_names[str(k)] = [codenames[i] for i in v]
        save_file(self.candidate_name_path,candidate_names)
    
    def generate_filter(self):
        reports = load_file(self.enhanced_report_path)
        # print(len(reports))
        codenames = load_file(self.candidate_name_path)
        filter_list = [] 
        for _,r in reports.iterrows():
            print(r['id'])
            files = r['files']
            if isinstance(files, str):
                files = [files]
            candidate_file = codenames[str(r["id"])]
            if self.file_type == 'Java':
                candidate_file = [get_rel_file(i) for i in candidate_file]
            for f in files:
                # print("f",f)
                if f in candidate_file[:self.args.N]:
                    filter_list.append(r["id"])
                    break
        print(f"{self.project}: {len(filter_list)}")
        save_file(self.filtered_path,filter_list)
        
    # =========================test================================
    def candidate_file_select(self,query,version):
        source_path = self.S.getPath_source(self.group, self.project, version)
        if self.group == 'Python':
            codes = read_codes(source_path,'py')
        else:
            codes = read_codes(source_path)
        content = list(codes.values())
        files = list(codes.keys())
        if self.group == 'Python':
            rel_files = list(files)
        else:
            rel_files = list(map(get_rel_file,files))
        with mp.Pool(mp.cpu_count()) as pool: # , timing(f"Processing {self.verName}") as t
            content = pool.map(process_text,content)
        query = process_text(query).split(' ')
        tokenized_corpus = [i.split(' ') for i in content]
        bm25 = BM25Okapi(tokenized_corpus)
        score = bm25.get_scores(query)
        index = np.argsort(-score)
        print([rel_files[i] for i in index[:30]])

def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title='argument-parser')
    group.add_argument('-s','--stage', type=int, default=2,
                       help='stage')
    group.add_argument('-g','--group', type=str, default='Apache',
                       help='Apache or Python')
    group.add_argument('-p','--project', type=str, default='HTTPCLIENT',
                       help='project name')
    group.add_argument('-m','--merge_type', type=int, default=3,
                       help='merge_type')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    # # S = Subjects()
    # _group = args.group # 'Python' # Apache
    # _project = args.project # 'HIVE' # HTTPCLIENT, JCR,
    group = 'Apache'
    project = "aspectj"
    gitrepo = f'../dataset/{project}'
    xml_report_path = f'../dataset/{project}-updated-data.xml' 
    
    _report_type = 'all' # 'origin','origin_sen','summ','summ_sen','add','add_sen','all','all_sen'
    _merge_type = 9
    if args.stage == 0:
        # ===============candidate retrieval=============================
        conf = Configure(group,project,gitrepo,xml_report_path,report_type=_report_type,merge_type=_merge_type)
        cand_ret = CandidateRetrival(conf)
        # cand_ret.enhanced_report() #uncomment
        cand_ret.generate_token_candidate()
    elif args.stage == 1:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-v3")
        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=510,
                            chunk_overlap=0,  # number of tokens overlap between chunks
                            length_function=lambda x: len(tokenizer.tokenize(x)),
                            separators=['\n\n', '\n', ' ', '']
                        )
        model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v3")
        pool=model.start_multi_process_pool()
        def encode(texts):
            emb=model.encode_multi_process(texts,pool)
            return emb
            # ===============candidate retrieval=============================
        conf = Configure(group,project,gitrepo,xml_report_path,report_type=_report_type,merge_type=_merge_type)
        cand_ret = CandidateRetrival(conf)
        cand_ret.generate_sen_candidate()
        model.stop_multi_process_pool(pool)
        
    elif args.stage == 2:
        for report_type in ['all'] if _report_type is None else [_report_type]: # 'origin','origin_sen','summ','summ_sen','cause','cause_sen',
            # for group in (S.groups if _group is None else [_group]): #S.groups
            #     for project in (S.projects[group] if _project is None else [_project]):
                # ===============candidate retrieval=============================
            conf = Configure(group,project,gitrepo,xml_report_path,report_type=report_type,merge_type=_merge_type)
            cand_ret = CandidateRetrival(conf)
            cand_ret.generate_list_df()
            cand_ret.rank_aggregation()
            cand_ret.generate_filter()
            print(f"{report_type} finished.")
            print('=='*20)