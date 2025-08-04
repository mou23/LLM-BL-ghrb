import os
from tqdm import tqdm
from utils import OpenaiClient, load_file, save_file
from Configure import Configure
import argparse

class ChatReranker():
    def __init__(self, args):
        
        self.model_name = args.model_name # 'gpt-3.5-turbo-0125'
        self.log_interval = args.log_interval
        self.group = args.group
        self.project = args.project
        self.OutputPATH = args.output_dir
        self.file_type = 'Java' if args.group == 'Apache' else 'Python'
        self.args = args
        self.log = open(f'{self.OutputPATH}/{self.project}_logs.txt','a')
        self.program = args.program
        self.prompt_path = args.prompt_path
        # self.report_path = args.report_path
        self.filtered_path = args.filtered_path
        self.res_list_path = args.res_list_path
        self.candidate_name_path = args.candidate_name_path
    
    def finish(self):
        self.log.close()
        
    def writeres(self,dir,id,res):
        with open(os.path.join(dir,f"{id}.txt"),'w') as f:
            for i in res:
                f.write(f'{i[0]}\t{i[1]}\t{i[2]}\t{i[3]}\n')
               
        
    def do_inference(self):
        prompts = load_file(self.prompt_path)
        print(len(prompts))
        for id,v in tqdm(prompts.items(),total= len(prompts),desc=f"{self.project}-{self.program}"):
            version = v['version']
            prompt = v['prompt']
            nameids = v['names']
            bgfiles = v['files']
            result_llm_path = os.path.join(self.OutputPATH, self.project,u'%s_%s_%s'%(self.program,self.project, version),f"{id}_llm.txt")
            result_dir = os.path.join(self.OutputPATH, self.project,u'%s_%s_%s'%(self.program,self.project, version))
            
            if not os.path.exists(result_dir):
                os.makedirs(result_dir,exist_ok=True)

            if os.path.exists(result_llm_path) and os.path.getsize(result_llm_path)>0:
                continue
                
            if 'gpt' in self.args.model:
                api_key = ""
                api_base = ""
                client = OpenaiClient(api_key)
                files,output = client.chat(len(nameids),self.model_name,prompt,1,temperature=0.2)
            elif self.args.model == 'llama3_70B' or self.args.model == 'codellama_34B' or self.args.model == 'codellama_70B':
                api_key = ""
                api_base = ""
                client = OpenaiClient(api_key,api_base)
                files,output = client.chat(len(nameids),self.model_name,prompt,0.1,3,temperature=0.2)
            elif self.args.model == 'llama3_8B':
                api_key = ""
                api_base = ""
                client = OpenaiClient(api_key,api_base)
                files,output = client.chat(len(nameids),self.model_name,prompt,0.1,3,temperature=0.2,stop='<|eot_id|>')
            elif self.args.model == 'codellama_7B':
                api_key = ""
                api_base = ""
                client = OpenaiClient(api_key,api_base)
                files,output = client.chat(len(nameids),self.model_name,prompt,0.1,3,temperature=0.2)
            # if len(files)!= 0 and len(files)!=len(nameids):
            #     for i in range(len(nameids)):
            #         if i not in files:
            #             files.append(i)
            #     print(f"{verName}: report {id} rank error.")
            #     self.log.write(f"{self.project}-{verName}-{self.program}-{id}:{output}\n")
            #     self.log.flush()
            if len(files)== 0:
                print(f"{version}: report {id} output none.")
                for out in output: 
                    self.log.write(f"{self.project}-{version}-{self.program}-{id}: output none.\n{out}")
                    self.log.flush()
                continue
            elif output:
                for out in output: 
                    self.log.write(f"{self.project}-{version}-{self.program}-{id}: output error.\n{out}")
                    self.log.flush()
            save_file(result_llm_path,files)
            

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
    S = Subjects()
    args = get_args()
    _group = args.group # 'Python' # Apache
    _project = args.project # 'HIVE' # HTTPCLIENT, JCR, 
    _report_type = args.report_type # 'origin','origin_sen','summ','summ_sen','add','add_sen','all','all_sen'
    _merge_type = args.merge_type
    _sen_type = args.sen_type
    _sniplen = args.length
    _prompt_type = args.prompt_type
    
    for report_type in ['all_sen'] if _report_type is None else [_report_type]: #'base2','base','tok','sen',
    # for merge_type in [3,2] if _merge_type is None else [_merge_type]:
        # for sen_type in ['sen_rm','tok_rm'] if _sen_type is None else [_sen_type]:
        for sniplen in [100] if _sniplen is None else [_sniplen]: # 50,150,
            for group in (S.groups if _group is None else [_group]): #S.groups
                for project in (S.projects[group] if _project is None else [_project]):
                # ===============candidate retrieval=============================
                    try:
                        conf = Configure(group,project,'gpt4o_mini',report_type=report_type,merge_type=_merge_type,N=20,sen_type=_sen_type,sniplen=sniplen,prompt_type = _prompt_type)
                        if not os.path.exists(conf.prompt_path):
                            print(f"{conf.prompt_path} not exists.")
                            continue
                        reranker = ChatReranker(conf)
                        reranker.do_inference()
                        # reranker.finish()
                    except Exception as e:
                        print(f"{conf.program}-{project}: {e}")
                        continue