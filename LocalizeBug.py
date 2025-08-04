import os
import sys
import xml.etree.ElementTree as ET
from CandidateRetrieval import CandidateRetrival
from ChatReranker import ChatReranker
from Configure import Configure
from PrepareData import Prepare
from SentenceRetrieval import SentenceRetrieval
from utils import checkout_commit

def get_bug_data(xml_report_path):
    bugs = []
    tree = ET.parse(xml_report_path)
    root = tree.getroot()
    for element in root.findall(".//table"):
        bug_id = element[1].text
        summary = element[2].text
        description = element[3].text
        buggy_commit = element[4].text
        fixed_files = (element[5].text).split('.java')

        bug_data = {
            "id": bug_id,
            "summary": summary,
            "description": description,
            "version": buggy_commit,
            "files": [(file + '.java').strip() for file in fixed_files[:-1]]
        }
        bugs.append(bug_data)

    return bugs


if __name__ == "__main__":
    # group = 'Apache'
    # project = "dubbo"
    # gitrepo = f'ghrb/{project}'
    # xml_report_path = f'ghrb/{project}.xml'
    group = sys.argv[1]
    project = sys.argv[2]
    gitrepo = f'ghrb/{project}'
    xml_report_path = f'ghrb/{project}.xml'
    bugs = get_bug_data(xml_report_path)
    _merge_type = 9
    _report_type = 'all_sen'

    for bug in bugs:
        conf = Configure(group, project, bug['id'], gitrepo, report_type=_report_type, merge_type=_merge_type)
        checkout_suceed = checkout_commit(gitrepo, bug['version'])
        if not checkout_suceed:
            sys.exit(1)
        print(f"Processing bug {bug['id']} in {project}...")
        process = Prepare(conf)
        process.prepare_report(bug)
        process.prepare_id2version()
        process.prepare_version_codes()
        process.extract_classname_methodname()
        process.get_rel_paths()
        process.prepare_report_flag()

        print(f"Retrieving candidates...")

        cand_ret = CandidateRetrival(conf)
        cand_ret.enhanced_report()
        cand_ret.generate_candidates()
        
        print(f"Preparing ranked list...")
        cand_ret.generate_list_df()
        cand_ret.rank_aggregation()
        cand_ret.generate_filter()

        print(f"Identifying relevant lines...")
        if os.path.exists(conf.prompt_path):
            continue
        sentence_ret = SentenceRetrieval(conf)
        sentence_ret.generate_sen_prompt()

        print(f"Reranking...")
        if not os.path.exists(conf.prompt_path):
            print(f"{conf.prompt_path} not exists.")
            continue
        reranker = ChatReranker(conf)
        reranker.do_inference()