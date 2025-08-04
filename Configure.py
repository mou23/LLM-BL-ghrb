import os
# from commons.Subjects import Subjects

class Configure():
    def __init__(self,group,project,bug_id,gitrepo,model ='gpt4o_mini',report_type='all_sen' ,sen_type='sen',log_interval=1,N=20,sniplen=100,merge_type=3,prompt_type='old'):
        self.group = group
        self.project = project
        self.gitrepo = gitrepo
        self.bug_id = bug_id

        self.output_dir = 'expresults/'
        if not os.path.exists(os.path.join(self.output_dir,self.project,self.bug_id)):
            os.makedirs(os.path.join(self.output_dir,self.project,self.bug_id),exist_ok=True)
        self.report_path = os.path.join(self.output_dir,self.project,self.bug_id,'reports.pkl')
        self.answer_path = os.path.join(self.output_dir,self.project,self.bug_id,f'answer.json')
        self.id2version_path = os.path.join(self.output_dir,self.project,self.bug_id,'id2version.json')
        self.version2shas_path = os.path.join(self.output_dir,project,self.bug_id,'version2shas.json')
        self.version2paths_path = os.path.join(self.output_dir,project,self.bug_id,'version2paths.json')
        self.shas_path = os.path.join(self.output_dir,project,self.bug_id,'shas.json')
        self.class_method_names_path = os.path.join(self.output_dir,project,self.bug_id,'class_method_names.json')
        self.version2relpaths_path = os.path.join(self.output_dir,project,self.bug_id,'version2relpaths.json')
        self.version2sha_rel_path = os.path.join(self.output_dir,project,self.bug_id,'version2sha_rels.json')
        self.filter_version_path = os.path.join(self.output_dir,project,self.bug_id,'filter_versions.txt')
        self.filter_report_path = os.path.join(self.output_dir,project,self.bug_id,'filter_report.pkl')
        
        self.log_interval = log_interval
        self.N = N
        self.sniplen = sniplen
        self.model = model
        self.sen_type = sen_type
        self.report_type = report_type
        self.merge_type = merge_type
        self.prompt_type = prompt_type
        
        self.program = f"{self.report_type}_{self.merge_type}"
        # CandidateRetrival
        
        self.candidate_path = os.path.join(self.output_dir,self.project,self.bug_id,'candidate.pkl')
        self.candidate_list_path = os.path.join(self.output_dir,self.project,self.bug_id,'candidate_df.pkl')
        self.codename_path = os.path.join(self.output_dir,self.project,self.bug_id,'codenames.json')
        self.candidate_name_path = os.path.join(self.output_dir,f'{self.project}_{self.bug_id}_{self.report_type}_{self.merge_type}_candidate_names.json')
        self.filtered_path = os.path.join(self.output_dir,f'{self.project}_{self.bug_id}_{self.report_type}_{self.merge_type}_filter.json')
        # the two files only used for test
        self.candidate_list_csv_path = os.path.join(self.output_dir,self.project,self.bug_id,f'{self.project}_candidate.csv')
        self.rel_path = os.path.join(self.output_dir,self.project,self.bug_id,f'{self.project}_candidate_rel.csv')
        # SentenceRetrieval
        self.prompt_path = os.path.join(self.output_dir,self.project,self.bug_id,f"{self.project}_{self.report_type}_{self.merge_type}_{self.N}_{self.prompt_type}_{self.sen_type}_{self.sniplen}_prompts.json")
        # ChatReranker
        self.res_list_path = os.path.join(self.output_dir,self.project,self.bug_id,self.group,self.project,u'%s.csv'%(self.program))
        
        
        
        if self.merge_type == 9:
            if self.report_type == 'summ_sen':
                self.typelist = ['summ_sen']
            elif self.report_type == 'summ':
                self.typelist = ['summ']
            elif self.report_type == 'origin':
                self.typelist = ['origin']
            elif self.report_type == 'origin_sen':
                self.typelist = ['origin_sen']
            elif self.report_type == 'cause':
                self.typelist = ['cause']
            elif self.report_type == 'cause_sen':
                self.typelist = ['cause_sen']
            elif self.report_type == 'all':
                self.typelist = ['origin','summ','cause']
            elif self.report_type == 'all_sen':
                self.typelist = ['origin','summ','cause','origin_sen','summ_sen','cause_sen']
        else:
            if self.report_type == 'summ_sen':
                self.typelist = ['origin','summ','origin_sen','summ_sen']
            elif self.report_type == 'summ':
                self.typelist = ['origin','summ']
            elif self.report_type == 'origin':
                self.typelist = ['origin']
            elif self.report_type == 'origin_sen':
                self.typelist = ['origin','origin_sen']
            elif self.report_type == 'add':
                self.typelist = ['origin','add']
            elif self.report_type == 'add_sen':
                self.typelist = ['origin','add','origin_sen','add_sen']
            elif self.report_type == 'all':
                self.typelist = ['origin','summ','cause']
            elif self.report_type == 'all_sen':
                self.typelist = ['origin','summ','cause','origin_sen','summ_sen','cause_sen']
                
        if self.model == 'llama3_70B':
            self.model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
        elif self.model == 'llama3_8B':
            self.model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
        elif self.model == 'codellama_7B':
            self.model_name = 'codellama/CodeLlama-7b-Instruct-hf'
        elif self.model == 'codellama_34B':
            self.model_name = 'CodeLlama-34b-Instruct-hf'
        elif self.model == 'codellama_70B':
            self.model_name = 'CodeLlama-70b-Instruct-hf'
        elif self.model == 'gpt3':
            self.model_name = 'gpt-3.5-turbo-0125' # 'gpt-3.5-turbo'
        elif self.model == 'gpt4o_mini':
            self.model_name = 'gpt-4o-mini'