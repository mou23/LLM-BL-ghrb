import os
import pandas as pd
import multiprocessing as mp

from Configure import Configure
from utils import (save_file, load_file, process_text, ls_tree,
                   extractShaAndPath, cat_file_blob,
                   parse_java_file)



class Prepare:
    def __init__(self, conf):
        self.output_dir = os.path.join(conf.output_dir, 'raw')

        self.project = conf.project
        self.group = conf.group
        self.report_path = conf.report_path
        
        self.gitrepo = conf.gitrepo
        # self.xml_report_path = conf.xml_report_path

        self.id2version_path = conf.id2version_path
        self.answer_path = conf.answer_path
        self.version2shas_path = conf.version2shas_path
        self.version2paths_path = conf.version2paths_path
        self.shas_path = conf.shas_path
        self.class_method_names_path = conf.class_method_names_path
        self.version2relpaths_path = conf.version2relpaths_path
        self.version2sha_rel_path = conf.version2sha_rel_path
        self.filter_version_path = conf.filter_version_path
        self.filter_report_path = conf.filter_report_path


    def prepare_report(self,report):
        df = pd.DataFrame([report])
        # print(df.head())
        df['summary'] = df['summary'].fillna('')
        df['description'] = df['description'].fillna('')
        with mp.Pool(mp.cpu_count()) as pool:
            df['processed_summ'] = list(pool.map(process_text, df['summary']))
            df['processed_desc'] = list(pool.map(process_text, df['description']))
        df.to_pickle(self.report_path)

    def prepare_id2version(self):
        reports = load_file(self.report_path)
        id2version = {r['id']: r['version'] for _, r in reports.iterrows()}
        id2answer = {r['id']: len(r['files']) for _, r in reports.iterrows()}
        save_file(self.id2version_path, id2version)
        save_file(self.answer_path, id2answer)

    def prepare_version_codes(self):
        reports = load_file(self.report_path)
        versions = reports['version']
        ver2paths, ver2shas, all_shas = {}, {}, []
        for v in versions:
            filesha_path = ls_tree(self.gitrepo, v)
            filesha_path = [f for f in filesha_path if f.endswith('.java')]# or f.endswith('.py')]
            sha_list, paths = extractShaAndPath(filesha_path)
            ver2paths[v], ver2shas[v] = paths, sha_list
            all_shas.extend([s for s in sha_list if s not in all_shas])
        save_file(self.version2paths_path, ver2paths)
        save_file(self.version2shas_path, ver2shas)
        save_file(self.shas_path, all_shas)

    def _extract_classname_methodname(self, sha):
        content = cat_file_blob(self.gitrepo, sha)
        # if self.group == 'Python':
        #     return parse_python_file(sha, content)
        return parse_java_file(sha, content)

    def extract_filenames(self):
        version2paths = load_file(self.version2paths_path)
        filenames = set(os.path.basename(p).split('.')[0] for paths in version2paths.values() for p in paths)
        return list(filenames)

    def extract_classname_methodname(self):
        shas = load_file(self.shas_path)
        methodnames, classnames = [], self.extract_filenames()
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(self._extract_classname_methodname, shas)
            for c, m in results:
                if c: classnames.extend(c.split())
                if m: methodnames.extend(m.split())
        save_file(self.class_method_names_path, {
            'classname': list(set(classnames)),
            'methodname': list(set(methodnames))
        })

    def _get_rel_path(self, path):
        # for prefix in ["/org/", "/java/", "src/"]:
        #     if prefix in path:
        #         rel = path.split(prefix)[-1].replace('/', '.')
        #         return rel
        # print(f"[Warning] Package name issue in: {path}")
        return path.replace('\\', '/')

    def get_rel_paths(self):
        version2paths = load_file(self.version2paths_path)
        version2shas = load_file(self.version2shas_path)
        version2relpaths = {}
        for version in version2paths:
            rels = [self._get_rel_path(p) for p in version2paths[version]]
            version2relpaths[version] = rels
        save_file(self.version2relpaths_path, version2relpaths)

        version2sha_rel = {}
        for version in version2shas:
            sha2rels, rel2sha = {}, {}
            for sha, rel in zip(version2shas[version], version2relpaths[version]):
                sha2rels.setdefault(sha, []).append(rel)
                rel2sha.setdefault(rel, []).append(sha)
            version2sha_rel[version] = {'sha2rel': sha2rels, 'rel2sha': rel2sha}
        save_file(self.version2sha_rel_path, version2sha_rel)

    def prepare_report_flag(self):
        reports = load_file(self.report_path)
        version2sha_rel = load_file(self.version2sha_rel_path)

        bugflag, id2methods, id2shas, filter_versions = [], {}, {}, {}
        for version in set(reports['version']):
            rel2sha = version2sha_rel.get(version, {}).get('rel2sha', {})
            tmpreports = reports[reports['version'] == version]
            ver_bugs = []
            for _, r in tmpreports.iterrows():
                matched_methods, matched_shas = [], []
                for f in r['files']:
                    if f in rel2sha:
                        matched_methods.append(f)
                        matched_shas.extend(rel2sha[f])
                if matched_methods:
                    bugflag.append(r['id'])
                    ver_bugs.append(r['id'])
                    id2methods[r['id']] = matched_methods
                    id2shas[r['id']] = matched_shas
            if ver_bugs:
                filter_versions[version] = ver_bugs

        reports['flag'] = reports['id'].isin(bugflag)
        reports.to_pickle(self.report_path)

        filtered = reports[reports['flag']].copy()
        filtered['files'] = filtered['id'].apply(lambda x: id2methods.get(x, []))
        filtered['shas'] = filtered['id'].apply(lambda x: id2shas.get(x, []))
        filtered.to_pickle(self.filter_report_path)
        save_file(self.filter_version_path, filter_versions)



if __name__ == "__main__":
        group = 'Apache'
        project = "aspectj"
        gitrepo = f'../dataset/{project}'
        xml_report_path = f'../dataset/{project}-updated-data.xml'

        print("=" * 20 + project + "=" * 20)
        conf = Configure(group, project, gitrepo, xml_report_path)
        process = Prepare(conf)
        process.prepare_reports()
        process.prepare_id2version()
        process.prepare_version_codes()
        process.extract_classname_methodname()
        process.get_rel_paths()
        process.prepare_report_flag()