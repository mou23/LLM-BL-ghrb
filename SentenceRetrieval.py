import os
from Configure import Configure
from utils import load_file, save_file, process_text, remove_prefix, get_rel_file, remove_comments, EmbeddingModel
import tiktoken
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
import random
import codecs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from pandarallel import pandarallel
import argparse
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-v3")
text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=510,
                    chunk_overlap=0,  # number of tokens overlap between chunks
                    length_function=lambda x: len(tokenizer.tokenize(x)),
                    separators=['\n\n', '\n', ' ', '']
                )

class SentenceRetrieval:
    def __init__(self,args) -> None:
        self.args = args
        self.project = args.project
        self.group = args.group
        self.gitrepo = args.gitrepo
        self.OutputPATH = args.output_dir
        self.file_type = 'Java' if args.group == 'Apache' else 'Python'
        self.tokenizer = tiktoken.get_encoding('cl100k_base')
        # self.origin_candidate = load_file(self.args.candidate_path)
        self.candidate = load_file(self.args.candidate_name_path)
        self.filtered_path = self.args.filtered_path
        self.report_path = self.args.report_path
        self.prompt_path = self.args.prompt_path

    def tiktoken_len(self,text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def split_by_len(self,x,N):
        t_id = self.tokenizer.encode(x)[:N]
        return self.tokenizer.decode(t_id)
    
    def candidate_select_strategy(self,report,fcontent,_type='Java'):
        if self.args.sen_type=='snip':
            spidx = []
            content = []
            q = process_text(report).split()
            for i,c in enumerate(fcontent):
                chunks = self.text_splitter.split_text(c)                
                for snippet in chunks:
                    spidx.append(i)
                    content.append(snippet)
            tokenized_corpus = [process_text(i).split()  for i in content]
            bm25 = BM25Okapi(tokenized_corpus)
            score = bm25.get_scores(q)
            index = np.argsort(-score)
            selected = []
            unique = []
            for i in index:
                if spidx[i] not in unique:
                    unique.append(spidx[i])
                    selected.append((spidx[i],content[i]))
            sorted_selected = sorted(selected,key=lambda x: x[0])
            sorted_selected = [i[1] for i in sorted_selected]
            return sorted_selected
        elif self.args.sen_type=='tok':
            def selectd_sentence(query,lines):
                lines_with_counts = []
                for no, line in enumerate(lines):
                    pline = process_text(line)
                    if pline:
                        shared_words_count = sum(1 for i in pline.split() if i in query)
                        # different 
                        lines_with_counts.append((no,shared_words_count/len(pline)))
                sorted_lines = sorted(lines_with_counts, key=lambda x: x[1], reverse=True)
                c = '\n'.join([lines[i[0]] for i in sorted_lines])
                summ_c = self.split_by_len(c,self.args.sniplen)
                k = len(summ_c.splitlines())-1
                sorted_lines = sorted(sorted_lines[:k], key=lambda x: x[0])
                return '\n'.join([lines[i[0]] for i in sorted_lines])
            # fcontent = [remove_prefix(i) for i in fcontent]
            report_tokens = process_text(report)
            report_tokens = list(set(report_tokens.split()))
            selected = [selectd_sentence(report_tokens,i.splitlines()) for i in fcontent]
            return selected
        elif self.args.sen_type=='tok_rm':
            def selectd_sentence(query,lines):
                lines_with_counts = []
                for no, line in enumerate(lines):
                    pline = process_text(line)
                    if pline:
                        shared_words_count = sum(1 for i in pline.split() if i in query)
                        # different 
                        lines_with_counts.append((no,shared_words_count/len(pline)))
                sorted_lines = sorted(lines_with_counts, key=lambda x: x[1], reverse=True)
                c = '\n'.join([lines[i[0]] for i in sorted_lines])
                summ_c = self.split_by_len(c,self.args.sniplen)
                k = len(summ_c.splitlines())-1
                sorted_lines = sorted(sorted_lines[:k], key=lambda x: x[0])
                return '\n'.join([lines[i[0]] for i in sorted_lines])
            if self.file_type == 'Python':
                fcontent = [remove_comments(self.file_type)(i) for i in fcontent]
            else:
                fcontent = [remove_comments(self.file_type)(i) for i in fcontent]
                fcontent = [remove_prefix(self.file_type)(i) for i in fcontent]
            report_tokens = process_text(report)
            report_tokens = list(set(report_tokens.split()))
            selected = [selectd_sentence(report_tokens,i.splitlines()) for i in fcontent]
            return selected
        elif self.args.sen_type=='tok2':
            def selectd_sentence(query,lines):
                lines_with_counts = []
                for no, line in enumerate(lines):
                    pline = process_text(line)
                    if pline:
                        shared_words_count = sum(1 for i in pline.split() if i in query)
                        lines_with_counts.append((no,shared_words_count))
                sorted_lines = sorted(lines_with_counts, key=lambda x: x[1], reverse=True)
                c = '\n'.join([lines[i[0]] for i in sorted_lines])
                summ_c = self.split_by_len(c,self.args.sniplen)
                k = len(summ_c.splitlines())-1
                sorted_lines = sorted(sorted_lines[:k], key=lambda x: x[0])
                return '\n'.join([lines[i[0]] for i in sorted_lines])

            report_tokens = process_text(report)
            report_tokens = list(set(report_tokens.split()))
            selected = [selectd_sentence(report_tokens,i.splitlines()) for i in fcontent]
            return selected
        elif self.args.sen_type=='tok2_rm':
            def selectd_sentence(query,lines):
                lines_with_counts = []
                for no, line in enumerate(lines):
                    pline = process_text(line)
                    if pline:
                        shared_words_count = sum(1 for i in pline.split() if i in query)
                        lines_with_counts.append((no,shared_words_count))
                sorted_lines = sorted(lines_with_counts, key=lambda x: x[1], reverse=True)
                c = '\n'.join([lines[i[0]] for i in sorted_lines])
                summ_c = self.split_by_len(c,self.args.sniplen)
                k = len(summ_c.splitlines())-1
                sorted_lines = sorted(sorted_lines[:k], key=lambda x: x[0])
                return '\n'.join([lines[i[0]] for i in sorted_lines])
            if self.file_type == 'Python':
                fcontent = [remove_comments(self.file_type)(i) for i in fcontent]
            else:
                fcontent = [remove_comments(self.file_type)(i) for i in fcontent]
                fcontent = [remove_prefix(self.file_type)(i) for i in fcontent]
            report_tokens = process_text(report)
            report_tokens = list(set(report_tokens.split()))
            selected = [selectd_sentence(report_tokens,i.splitlines()) for i in fcontent]
            return selected
        elif self.args.sen_type=='base2':
            spidx = []
            content = []
            for i,c in enumerate(fcontent):
                c = remove_prefix(_type)(c)
                c = self.split_by_len(c,self.args.sniplen)
                content.append(c)
            return content
        elif self.args.sen_type=='base':
            spidx = []
            content = []
            for i,c in enumerate(fcontent):
                c = self.split_by_len(c,self.args.sniplen)
                content.append(c)
            return content
        elif self.args.sen_type=='rand':
            content = []
            for i,c in enumerate(fcontent):
                lines = c.splitlines()
                line_no = list(range(len(lines)))
                line_len = [self.tiktoken_len(line) for line in lines]
                random.shuffle(line_no)
                cal_len = 0
                selected = []
                for no in line_no:
                    if cal_len+line_len[no]<self.args.sniplen:
                        cal_len+=line_len[no]
                        selected.append(no)
                    else:
                        break
                selected.sort()
                content.append('\n'.join([lines[no] for no in selected]))
            return content

                
    def get_report_prompt(self,r):
        id = r['id']
        summary = r['summary']
        description = r['description']
        r_summ = r.r_summ
        r_cause = r.r_cause
        vername = r['vername']
        files = r['files']
        candidate_file = self.candidate[str(id)][:self.args.N]
        
        all_contexts = []
        source_path = self.S.getPath_source(self.group, self.project, vername)
        for i in candidate_file:
            with codecs.open(os.path.join(source_path,i),encoding='utf-8',errors='ignore') as f:
                content = f.read()
                all_contexts.append(content)
        
        if self.args.prompt_type == 'old':
            report = summary+' '+description
        else:
            if self.args.report_type=='add' or self.args.report_type=='add_sen' or self.args.report_type=='all' or self.args.report_type=='all_sen':
                report = summary+' '+description+' '+r_summ + ' '+r_cause
            elif self.args.report_type=='origin' or self.args.report_type=='origin_sen':
                report = summary+' '+description
            elif self.args.report_type=='summ' or self.args.report_type=='summ_sen':
                report = summary+' '+description +' '+r_summ
        contents = self.candidate_select_strategy(report,all_contexts,self.file_type)
        
        filecontent= ''
        for i,f, c in zip(range(len(candidate_file)),candidate_file,contents):

            filecontent+=f"[{i}]: {get_rel_file(f) if self.file_type=='Java' else f}\n{{\n{c}}}\n"

        
        summary = summary.replace('\n',' ')
        description = self.split_by_len(description,300).replace('\n',' ')
        prompt = [{'role': 'system','content': "You are a code review expert, an intelligent assistant that can rank files based on their relevancy to the bug report."},
                    {'role': 'user','content': f"Here is the bug report. It contains title and description.\ntitle: {summary}\ndescription: {description}"},
                    {'role': 'assistant','content': 'Received the bug report.'},
                    {'role': 'user','content': f'Based on this bug report, please rank the {len(candidate_file)} {self.file_type} files, each indicated by number identifier []:\n{filecontent}\nRank the {len(candidate_file)} files above based on their relevance to the report in descending order. The output format should be [] > [], e.g., [0] > [1]. Only response the ranking results, without any other textual explanation.'}
                    ]

        return (id,{'version':vername,'names':candidate_file,'prompt':prompt,'files':files})
        
    def generate_tok_prompt(self):
        reports = load_file(self.report_path)
        filtered = load_file(self.filtered_path)
        flag = reports['id'].apply(lambda x: x in filtered)
        reports = reports[flag]
        self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.args.sniplen,
                chunk_overlap=0,  # number of tokens overlap between chunks
                length_function=self.tiktoken_len,
                separators=['\n\n', '\n', ' ', '']
            )
        # pandarallel.initialize(progress_bar=True)
        results = reports.parallel_apply(self.get_report_prompt,axis=1)

        # 过滤器
        prompts = dict(results.tolist())
        save_file(self.prompt_path,prompts)
        print(f"{self.project} prompt processed.")

    def candidate_sen_select_strategy(self,report,fcontent,_type='Java'):
        model = EmbeddingModel()
        def selectd_sentence(query_emb,lines):
            line_embs = model.encode(lines)
            scores = cosine_similarity(query_emb,line_embs).max(axis=0).tolist()
            lines_with_counts = list(enumerate(scores))
            sorted_lines = sorted(lines_with_counts, key=lambda x: x[1], reverse=True)
            c = '\n'.join([lines[i[0]] for i in sorted_lines])
            summ_c = self.split_by_len(c,self.args.sniplen)
            k = len(summ_c.splitlines())-1
            sorted_lines = sorted(sorted_lines[:k], key=lambda x: x[0])
            return '\n'.join([lines[i[0]] for i in sorted_lines])
        if self.args.sen_type == 'sen_rm':
            if self.file_type == 'Python':
                fcontent = [remove_comments(self.file_type)(i) for i in fcontent]
            else:
                fcontent = [remove_comments(self.file_type)(i) for i in fcontent]
                fcontent = [remove_prefix(self.file_type)(i) for i in fcontent]
        report_emb = model.encode(report)
        selected = [selectd_sentence(report_emb,i.splitlines()) for i in fcontent]
        return selected
        
    def get_sen_prompt(self,r):
        id = r['id']
        summary = r['summary']
        description = r['description']
        r_summ = r.r_summ
        r_cause = r.r_cause
        files = r['files']
        version = r['version']
        candidate_file = self.candidate[str(id)][:self.args.N]
        
        all_contexts = []
        for i in candidate_file:
            file_path = os.path.join(self.gitrepo, i)
            with codecs.open(file_path, encoding='utf-8', errors='ignore') as f:
                content = f.read()
                all_contexts.append(content)
        chunks = [summary]
        if description:
            desc_chunks = text_splitter.split_text(description)
        else:
            desc_chunks = []
        chunks.extend(desc_chunks)
        if self.args.prompt_type != 'old':
            if self.args.report_type=='summ' or self.args.report_type=='summ_sen':
                chunks.append(r_summ)
            elif self.args.report_type=='all' or self.args.report_type=='all_sen':
                chunks.append(r_summ)
                chunks.append(r_cause)
        contents = self.candidate_sen_select_strategy(chunks,all_contexts,self.file_type)
        
        filecontent= ''
        for i,f, c in zip(range(len(candidate_file)),candidate_file,contents):
            filecontent+=f"[{i}]: {get_rel_file(f) if self.file_type=='Java' else f}\n{{\n{c}}}\n"

        
        summary = summary.replace('\n',' ')
        description = self.split_by_len(description,300).replace('\n',' ')
        prompt = [{'role': 'system','content': "You are a code review expert, an intelligent assistant that can rank files based on their relevancy to the bug report."},
                    {'role': 'user','content': f"Here is the bug report. It contains title and description.\ntitle: {summary}\ndescription: {description}"},
                    {'role': 'assistant','content': 'Received the bug report.'},
                    {'role': 'user','content': f'Based on this bug report, please rank the {len(candidate_file)} {self.file_type} files, each indicated by number identifier []:\n{filecontent}\nRank the {len(candidate_file)} files above based on their relevance to the report in descending order. The output format should be [] > [], e.g., [0] > [1]. Only response the ranking results, without any other textual explanation.'}
                    ]

        return (id,{'version':version,'names':candidate_file,'prompt':prompt,'files':files})
    
    def generate_sen_prompt(self):
        reports = load_file(self.report_path)
        filtered = load_file(self.filtered_path)
        flag = reports['id'].apply(lambda x: x in filtered)
        reports = reports[flag]
        print(len(reports))
        prompts = {}
        prompt_name = self.prompt_path.split('/')[-1]
        for _,r in tqdm(reports.iterrows(),total=len(reports),desc=f"{prompt_name}"):
            id,prompt = self.get_sen_prompt(r)
            prompts[id]= prompt
        # project_20_sen_all_200_prompts.json
        save_file(self.prompt_path,prompts)
        print(f"{self.prompt_path} saved.")
        

def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title='argument-parser')
    group.add_argument('-g','--group', type=str, default='Apache',
                       help='Apache or Python')
    group.add_argument('-p','--project', type=str, default='HTTPCLIENT',
                       help='project name')
    group.add_argument('-l','--length', type=int, default=100,
                       help='project name')
    group.add_argument('-m','--merge_type', type=int, default=3,
                       help='rank aggregation type')
    group.add_argument('-s','--sen_type', type=str, default='sen',
                       help='content reduction type')
    group.add_argument('-r','--report_type', type=str, default='all_sen',
                       help='report type')
    group.add_argument('-o','--prompt_type', type=str, default='old',
                       help='use only report or enhanced report')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    S = Subjects()
    _group = args.group # 'Python' # Apache
    _project = args.project # 'HIVE' # HTTPCLIENT, JCR, 
    _report_type = args.report_type # 'origin','origin_sen','summ','summ_sen','add','add_sen','all','all_sen'
    _merge_type = args.merge_type
    _sen_type = args.sen_type
    _sniplen = args.length
    _prompt_type = args.prompt_type
    if _sen_type != 'sen' and _sen_type != 'sen_rm':
        _group = None
        _project = None
        for group in (S.groups if _group is None else [_group]): #S.groups
            for project in (S.projects[group] if _project is None else [_project]):
            # ===============candidate retrieval=============================
                conf = Configure(group,project,report_type=_report_type,merge_type=_merge_type,sen_type=_sen_type,sniplen = _sniplen,prompt_type=_prompt_type)
                sentence_ret = SentenceRetrieval(conf)
                sentence_ret.generate_tok_prompt()
    else:
        # _group = None
        # _project = None
        
        model = SentenceTransformer("msmarco-distilbert-base-v3")
        pool=model.start_multi_process_pool()
        def encode(texts):
            emb=model.encode_multi_process(texts,pool)
            return emb
        for group in (S.groups if _group is None else [_group]): #S.groups
            for project in (S.projects[group] if _project is None else [_project]):
            # ===============candidate retrieval=============================
                # for sniplen in [50,100,150,200,250] if _sniplen == -1 else [_sniplen]:
                for report_type in ['all_sen'] if _report_type is None else [_report_type]:
                    conf = Configure(group,project,report_type=report_type,merge_type=_merge_type,sen_type=_sen_type,sniplen = _sniplen,prompt_type=_prompt_type)
                    if os.path.exists(conf.prompt_path):
                        continue
                    sentence_ret = SentenceRetrieval(conf)
                    sentence_ret.generate_sen_prompt()
        model.stop_multi_process_pool(pool)