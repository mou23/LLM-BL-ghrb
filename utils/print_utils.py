from text_process_utils import process_text
import tiktoken
from common_utils import load_file, get_rel_file
import codecs
import glob
import os
from commons.Subjects import Subjects

# -----------------------sentence score print-------------------------------------

def split_by_len(x,N):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    t_id = tokenizer.encode(x)[:N]
    return tokenizer.decode(t_id)
    

def print_sentence_score(report,fcontent):
    def selectd_sentence(query,lines):
        lines_with_counts = []
        for no, line in enumerate(lines):
            pline = process_text(line)
            if pline:
                shared_words_count = sum(1 for i in pline.split() if i in query)
                lines_with_counts.append((no,shared_words_count))
        score = [i[1] for i in lines_with_counts]
        max_score,min_score = max(score),min(score)
        print('\n'.join([f"{lines[i[0]]} score: {(i[1]- min_score)/(max_score - min_score):.2f}" for i in lines_with_counts]))
        print('='*20)
        sorted_lines = sorted(lines_with_counts, key=lambda x: x[1], reverse=True)
        c = '\n'.join([lines[i[0]] for i in sorted_lines])
        summ_c = split_by_len(c,300)
        k = len(summ_c.splitlines())-1
        sorted_lines = sorted(sorted_lines[:k], key=lambda x: x[0])
        return '\n'.join([f"{lines[i[0]]} score: {(i[1]- min_score)/(max_score - min_score):.2f}" for i in sorted_lines])
        
    report_tokens = process_text(report)
    report_tokens = list(set(report_tokens.split()))
    selected = selectd_sentence(report_tokens,fcontent.splitlines())
    return selected
    
def print_id_sentence(group,project,id):
    S = Subjects()
    reports = load_file(f'raw/{project}_reports.pkl')
    selected = reports[reports['id']==id].iloc[0]
    source_path = S.getPath_source(group, project, selected.vername)
    rel2abs = {}
    print(f"summary: {selected.summary}\ndescription:{selected.description}")
    print('='*20)
    if group == 'Apache':
        files = glob.glob(f'{source_path}/**/*.java', recursive=True)
        for i in files:
            if os.path.getsize(i) and 'org/apache' in i:
                abs_path = os.path.relpath(i,source_path)
                rel_path = get_rel_file(abs_path)
                if rel_path not in rel2abs:
                    rel2abs[rel_path] = [abs_path]
                else:
                    rel2abs[rel_path].append(abs_path)
    for i in selected.files:            
        if i in rel2abs:
            with codecs.open(os.path.join(source_path,rel2abs[i][0]),encoding='utf-8',errors='ignore') as f:
                content = f.read()
            print(content)
            print('='*20)
            lines = print_sentence_score(f"{selected.summary} {selected.description} {selected.r_summ} {selected.r_cause}",content)
            print(lines)
            print('='*20)

# dataframe print
def overall(df,groupname,metric_list,total=1):
    grouped_sum = df.groupby(groupname,sort=False)[metric_list].sum().reset_index()
    for _,r in grouped_sum.iterrows():
        print("{},{},{},{}".format(r[groupname],f"{r[metric_list[0]]}({r[metric_list[0]]/total*100:.1f}%)",f"{r[metric_list[1]]}({r[metric_list[1]]/total*100:.1f}%)",f"{r[metric_list[2]]}({r[metric_list[2]]/total*100:.1f}%)"))