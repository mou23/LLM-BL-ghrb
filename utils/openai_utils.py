import re
import time
from openai import OpenAI

class OpenaiClient:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)
    
    def extract_file(self,text,file_num):
        res = [int(num) for num in re.findall(r'\[(\d+)\]', text)]
        filtered = []
        flag = False
        for i in res:
            if i < file_num and i not in filtered:
                filtered.append(i)
            else:
                flag = True
        # print(f"Extracted files: {filtered}, Flag: {flag}")
        return filtered,flag

    def chat(self, file_num, model_name, messages, timeout=3, times=1, temperature=0.2):
        files = []
        outputs = []
        res = []
        while(times>0):
            try:
                response = self.client.chat.completions.create(model=model_name, messages=messages, temperature=temperature)
                output = response.choices[0].message.content
                # print("output",output)
            except Exception as e:
                print(e)
                time.sleep(timeout)
                times-=1
                continue
            files,flag = self.extract_file(output,file_num)
            if flag:
                outputs.append(output)
            time.sleep(timeout)
            times-=1
            if files:
                res.append(files)
        return res,outputs
    
    def extract_summary_cause(self,text):
        # Extract summary
        summary, possible_cause = "",""
                
        summary_start = text.find("Summary") + len("Summary")
        summary_start = text.find(":",summary_start)
        # summary_end = text.find("Possible Causes")
        summary_end = text.find("Functionality")
        summary = text[summary_start+1:summary_end].strip()

        # Extract possible cause
        # cause_start = text.find("Possible Causes") + len("Possible Causes")
        cause_start = text.find("Functionality") + len("Functionality")
        cause_start = text.find(":",cause_start)
        cause_end = text.find("Without access to the actual",cause_start)
        if cause_end==-1:
            possible_cause = text[cause_start+1:].strip()
        else:
            possible_cause = text[cause_start+1:cause_end].strip()
        return summary, possible_cause
    
    def summarize(self,model_name,messages,timeout=3,times = 1,temperature=0.2):
        r_summary,possible_cause = "",""
        while(times>0):
            try:
                response = self.client.chat.completions.create(model=model_name, messages=messages,temperature=temperature)
                output = response.choices[0].message.content
            except Exception as e:
                print(e)
                time.sleep(timeout)
                times-=1
                continue
            r_summary,possible_cause  = self.extract_summary_cause(output)
            time.sleep(timeout)
            times-=1
            if r_summary!="" and possible_cause!="":
                break
        return r_summary, possible_cause